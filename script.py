# ========== 导入依赖库 ==========
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
import streamlit as st

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

# ========== 页面配置 ==========
st.set_page_config(
    page_title="电影数据分析系统",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ========== 1. 数据加载与预处理模块 ==========
@st.cache_data(show_spinner="正在加载数据...")
def load_and_preprocess(data_path="./"):
    movies = pd.read_csv(os.path.join(data_path, "movies.csv"))
    ratings = pd.read_csv(os.path.join(data_path, "ratings.csv"))
    tags = pd.read_csv(os.path.join(data_path, "tags.csv"))

    # 预处理
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce")
    movies["title_clean"] = movies["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True)
    movies["genres_list"] = movies["genres"].str.split("|")

    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings["year"] = ratings["datetime"].dt.year

    return movies, ratings, tags


# ========== 2. 数据分析统计模块 ==========
class DataAnalyzer:
    def __init__(self, movies, ratings, tags):
        self.movies = movies
        self.ratings = ratings
        self.tags = tags

    def rating_distribution(self):
        dist = self.ratings["rating"].value_counts().sort_index()
        stats = {
            "平均分": round(self.ratings["rating"].mean(), 3),
            "中位数": self.ratings["rating"].median(),
            "标准差": round(self.ratings["rating"].std(), 3),
            "最低分": self.ratings["rating"].min(),
            "最高分": self.ratings["rating"].max()
        }
        return dist, stats

    def genre_distribution(self):
        all_genres = []
        for genres in self.movies["genres_list"].dropna():
            all_genres.extend(genres)
        genre_counts = pd.Series(Counter(all_genres)).sort_values(ascending=False)
        return genre_counts

    def genre_rating_compare(self):
        movie_ratings = self.ratings.groupby("movieId")["rating"].agg(["mean", "count"])
        merged = self.movies.merge(movie_ratings, left_on="movieId", right_index=True)
        genre_stats = {}
        for _, row in merged.iterrows():
            for g in row["genres_list"]:
                if g not in genre_stats:
                    genre_stats[g] = []
                genre_stats[g].append(row["mean"])
        result = {g: round(np.mean(scores), 3) for g, scores in genre_stats.items()}
        return pd.Series(result).sort_values(ascending=False)

    def year_trend(self):
        movie_by_year = self.movies.groupby("year").size().sort_index()
        rating_by_year = self.ratings.groupby("year").size().sort_index()
        avg_rating_by_year = self.ratings.groupby("year")["rating"].mean()
        return movie_by_year, rating_by_year, avg_rating_by_year

    def user_behavior(self):
        user_stats = self.ratings.groupby("userId").agg(
            rating_count=("rating", "count"),
            avg_rating=("rating", "mean")
        )
        return user_stats

    def top_movies_by_rating(self, min_votes=50, top_n=10):
        movie_stats = self.ratings.groupby("movieId").agg(
            avg_rating=("rating", "mean"),
            vote_count=("rating", "count")
        )
        filtered = movie_stats[movie_stats["vote_count"] >= min_votes]
        top = filtered.sort_values("avg_rating", ascending=False).head(top_n)
        return top.merge(self.movies[["movieId", "title"]], on="movieId")

    def top_movies_by_popularity(self, top_n=10):
        popular = self.ratings.groupby("movieId").size().sort_values(ascending=False).head(top_n)
        popular.name = "rating_count"
        return popular.to_frame().merge(self.movies[["movieId", "title"]], on="movieId")

    def long_tail_analysis(self):
        vote_counts = self.ratings.groupby("movieId").size().sort_values(ascending=False).values
        cumulative = np.cumsum(vote_counts)
        total = cumulative[-1]
        return vote_counts, cumulative / total

    def tag_popularity(self, top_n=20):
        return self.tags["tag"].value_counts().head(top_n)


# ========== 3. 电影推荐算法模块 ==========
@st.cache_resource
def build_recommender(movies, ratings):
    return MovieRecommender(movies, ratings)


class MovieRecommender:
    def __init__(self, movies, ratings):
        self.movies = movies
        self.ratings = ratings
        self.movie_idx = None
        self.user_movie_matrix = None
        self.item_sim = None
        self.user_sim = None
        self._build_matrix()

    def _build_matrix(self):
        self.user_movie_matrix = self.ratings.pivot(
            index="userId", columns="movieId", values="rating"
        ).fillna(0)
        self.movie_idx = {mid: i for i, mid in enumerate(self.user_movie_matrix.columns)}

    def popularity_recommend(self, top_n=10):
        popular = self.ratings.groupby("movieId").size().sort_values(ascending=False).head(top_n)
        return self.movies[self.movies["movieId"].isin(popular.index)][["movieId", "title", "genres"]]

    def rating_recommend(self, min_votes=50, top_n=10):
        movie_stats = self.ratings.groupby("movieId").agg(
            avg_rating=("rating", "mean"),
            vote_count=("rating", "count")
        )
        filtered = movie_stats[movie_stats["vote_count"] >= min_votes]
        top = filtered.sort_values("avg_rating", ascending=False).head(top_n)
        return self.movies[self.movies["movieId"].isin(top.index)][["movieId", "title", "genres"]]

    def content_based_recommend(self, movie_id, top_n=10):
        target = self.movies[self.movies["movieId"] == movie_id]
        if len(target) == 0:
            return pd.DataFrame()
        target_genres = set(target.iloc[0]["genres_list"])
        scores = []
        for _, row in self.movies.iterrows():
            if row["movieId"] == movie_id:
                continue
            movie_genres = set(row["genres_list"]) if isinstance(row["genres_list"], list) else set()
            jaccard = len(target_genres & movie_genres) / len(target_genres | movie_genres) if len(
                target_genres | movie_genres) > 0 else 0
            scores.append((row["movieId"], jaccard))
        scores.sort(key=lambda x: x[1], reverse=True)
        top_ids = [s[0] for s in scores[:top_n]]
        return self.movies[self.movies["movieId"].isin(top_ids)][["movieId", "title", "genres"]]

    def item_based_cf(self, movie_id, top_n=10):
        if self.item_sim is None:
            self.item_sim = cosine_similarity(self.user_movie_matrix.T)
        if movie_id not in self.movie_idx:
            return pd.DataFrame()
        idx = self.movie_idx[movie_id]
        sim_scores = list(enumerate(self.item_sim[idx]))
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, s in sim_scores[1:top_n + 1]]
        top_ids = [self.user_movie_matrix.columns[i] for i in top_indices]
        return self.movies[self.movies["movieId"].isin(top_ids)][["movieId", "title", "genres"]]

    def user_based_cf(self, user_id, top_n=10):
        if self.user_sim is None:
            self.user_sim = cosine_similarity(self.user_movie_matrix)
        user_list = list(self.user_movie_matrix.index)
        if user_id not in user_list:
            return pd.DataFrame()
        u_idx = user_list.index(user_id)
        sim_scores = list(enumerate(self.user_sim[u_idx]))
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        similar_users = [user_list[i] for i, s in sim_scores[1:21]]
        user_ratings = defaultdict(float)
        user_sim_sum = defaultdict(float)
        for sim_u in similar_users:
            sim = sim_scores[user_list.index(sim_u)][1]
            u_ratings = self.ratings[self.ratings["userId"] == sim_u]
            for _, row in u_ratings.iterrows():
                user_ratings[row["movieId"]] += row["rating"] * sim
                user_sim_sum[row["movieId"]] += sim
        watched = set(self.ratings[self.ratings["userId"] == user_id]["movieId"])
        candidates = {mid: user_ratings[mid] / user_sim_sum[mid] for mid in user_ratings if mid not in watched}
        sorted_cand = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_ids = [mid for mid, _ in sorted_cand]
        return self.movies[self.movies["movieId"].isin(top_ids)][["movieId", "title", "genres"]]

    def hybrid_recommend(self, user_id, top_n=10):
        cf_rec = self.user_based_cf(user_id, top_n=top_n * 2)
        if cf_rec.empty:
            return self.popularity_recommend(top_n)
        movie_stats = self.ratings.groupby("movieId")["rating"].mean()
        cf_rec["score"] = cf_rec["movieId"].map(movie_stats) * 0.3 + 0.7
        return cf_rec.sort_values("score", ascending=False).head(top_n)[["movieId", "title", "genres"]]


# ========== 4. 主程序 - 页面渲染 ==========
def main():
    # 侧边栏导航
    st.sidebar.title("🎬 电影数据分析系统")
    page = st.sidebar.radio(
        "功能导航",
        ["📊 数据概览", "📈 数据分析", "🖼️ 可视化图表", "🎯 电影推荐", "👤 用户画像"]
    )

    # 加载数据
    movies, ratings, tags = load_and_preprocess()
    analyzer = DataAnalyzer(movies, ratings, tags)
    recommender = build_recommender(movies, ratings)

    # ========== 页面1: 数据概览 ==========
    if page == "📊 数据概览":
        st.title("数据集基本信息")
        st.markdown("---")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("电影总数", f"{len(movies):,}")
        col2.metric("评分总数", f"{len(ratings):,}")
        col3.metric("用户总数", f"{ratings['userId'].nunique():,}")
        col4.metric("标签总数", f"{len(tags):,}")

        st.info(f"⏰ 数据时间跨度：{ratings['datetime'].min().year} - {ratings['datetime'].max().year}")

        st.subheader("数据样例预览")
        tab1, tab2, tab3 = st.tabs(["电影数据", "评分数据", "标签数据"])
        with tab1:
            st.dataframe(movies.head(10), use_container_width=True)
        with tab2:
            st.dataframe(ratings.head(10), use_container_width=True)
        with tab3:
            st.dataframe(tags.head(10), use_container_width=True)

    # ========== 页面2: 数据分析 ==========
    elif page == "📈 数据分析":
        st.title("数据分析统计")
        st.markdown("---")

        analysis_type = st.selectbox(
            "选择分析维度",
            ["评分分布统计", "电影类型分布", "各类型评分对比", "高分电影榜",
             "热门电影榜", "用户行为统计", "年份趋势分析", "热门标签"]
        )

        if analysis_type == "评分分布统计":
            dist, stats = analyzer.rating_distribution()
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("评分分布")
                st.bar_chart(dist)
            with col2:
                st.subheader("统计指标")
                st.json(stats)

        elif analysis_type == "电影类型分布":
            genre_counts = analyzer.genre_distribution()
            st.subheader("各类型电影数量")
            st.bar_chart(genre_counts)

        elif analysis_type == "各类型评分对比":
            genre_ratings = analyzer.genre_rating_compare()
            st.subheader("各类型平均评分")
            st.bar_chart(genre_ratings)

        elif analysis_type == "高分电影榜":
            min_votes = st.slider("最低评分人数", 10, 200, 50)
            top_n = st.slider("显示数量", 5, 30, 10)
            top_rated = analyzer.top_movies_by_rating(min_votes=min_votes, top_n=top_n)
            st.subheader(f"评分最高 Top{top_n}（至少{min_votes}人评分）")
            st.dataframe(
                top_rated[["title", "avg_rating", "vote_count"]],
                use_container_width=True,
                hide_index=True
            )

        elif analysis_type == "热门电影榜":
            top_n = st.slider("显示数量", 5, 30, 10)
            top_pop = analyzer.top_movies_by_popularity(top_n=top_n)
            st.subheader(f"最热门 Top{top_n}")
            st.dataframe(
                top_pop[["title", "rating_count"]],
                use_container_width=True,
                hide_index=True
            )

        elif analysis_type == "用户行为统计":
            user_stats = analyzer.user_behavior()
            col1, col2, col3 = st.columns(3)
            col1.metric("用户总数", len(user_stats))
            col2.metric("人均评分数", round(user_stats['rating_count'].mean(), 2))
            col3.metric("平均评分", round(user_stats['avg_rating'].mean(), 3))

            st.subheader("用户平均评分分布")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(user_stats["avg_rating"], bins=30, color="#64b5cd", edgecolor="white")
            ax.set_xlabel("平均评分")
            ax.set_ylabel("用户数量")
            st.pyplot(fig)
            plt.close()

        elif analysis_type == "年份趋势分析":
            m_year, r_year, avg_year = analyzer.year_trend()
            st.subheader("每年上映电影数量")
            st.line_chart(m_year)
            st.subheader("每年评分数量与平均评分")
            chart_data = pd.DataFrame({
                "评分数量": r_year,
                "平均评分": avg_year
            })
            st.line_chart(chart_data)

        elif analysis_type == "热门标签":
            top_tags = analyzer.tag_popularity(top_n=20)
            st.subheader("最热门的20个标签")
            st.bar_chart(top_tags)

    # ========== 页面3: 可视化图表 ==========
    elif page == "🖼️ 可视化图表":
        st.title("可视化图表")
        st.markdown("---")

        dist, stats = analyzer.rating_distribution()
        genre_counts = analyzer.genre_distribution()
        genre_ratings = analyzer.genre_rating_compare()
        m_year, r_year, avg_year = analyzer.year_trend()
        top_rated = analyzer.top_movies_by_rating()
        top_pop = analyzer.top_movies_by_popularity()
        user_stats = analyzer.user_behavior()
        vote_counts, cum_ratio = analyzer.long_tail_analysis()
        top_tags = analyzer.tag_popularity()

        # 评分分布
        st.subheader("1. 电影评分分布")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        dist.plot(kind="bar", color="#4c72b0", ax=ax1)
        ax1.set_xlabel("评分")
        ax1.set_ylabel("数量")
        st.pyplot(fig1)
        plt.close()

        # 类型分布
        st.subheader("2. 各类型电影数量分布")
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        genre_counts.plot(kind="bar", color="#55a868", ax=ax2)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig2)
        plt.close()

        # 类型评分对比
        st.subheader("3. 各类型电影平均评分对比")
        fig3, ax3 = plt.subplots(figsize=(12, 7))
        genre_ratings.plot(kind="bar", color="#c44e52", ax=ax3)
        ax3.set_ylim(3, 4.5)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig3)
        plt.close()

        # 高分榜
        st.subheader("4. 评分最高的Top10电影")
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        ax4.barh(top_rated["title"][::-1], top_rated["avg_rating"][::-1], color="#8172b3")
        ax4.set_xlabel("平均评分")
        ax4.set_xlim(4, 5)
        st.pyplot(fig4)
        plt.close()

        # 长尾分布
        st.subheader("5. 长尾分布与累计占比")
        fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 5))
        ax5a.plot(range(len(vote_counts)), vote_counts, color="#c44e52")
        ax5a.set_title("长尾分布图")
        ax5a.set_xlabel("电影排名")
        ax5a.set_ylabel("评分次数")
        ax5b.plot(range(len(cum_ratio)), cum_ratio, color="#55a868")
        ax5b.axhline(y=0.8, color="gray", linestyle="--", label="80%线")
        ax5b.set_title("累计评分占比")
        ax5b.set_xlabel("电影数量")
        ax5b.set_ylabel("累计评分占比")
        ax5b.legend()
        st.pyplot(fig5)
        plt.close()

        # 流行度 vs 评分
        st.subheader("6. 流行度与评分关系散点图")
        movie_stats = ratings.groupby("movieId").agg(
            avg_rating=("rating", "mean"),
            vote_count=("rating", "count")
        )
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        ax6.scatter(movie_stats["vote_count"], movie_stats["avg_rating"],
                    alpha=0.5, s=10, color="#4c72b0")
        ax6.set_xlabel("评分人数（流行度）")
        ax6.set_ylabel("平均评分")
        ax6.set_xscale("log")
        st.pyplot(fig6)
        plt.close()

        # 热门标签
        st.subheader("7. 最热门的20个标签")
        fig7, ax7 = plt.subplots(figsize=(12, 6))
        top_tags[::-1].plot(kind="barh", color="#8172b3", ax=ax7)
        ax7.set_xlabel("出现次数")
        st.pyplot(fig7)
        plt.close()

    # ========== 页面4: 电影推荐 ==========
    elif page == "🎯 电影推荐":
        st.title("电影推荐系统")
        st.markdown("---")

        rec_method = st.radio(
            "选择推荐算法",
            ["流行度推荐", "高评分推荐", "基于内容推荐", "物品协同过滤", "用户协同过滤", "混合推荐"],
            horizontal=True
        )

        top_n = st.slider("推荐数量", 5, 20, 10)

        if rec_method == "流行度推荐":
            result = recommender.popularity_recommend(top_n=top_n)
            st.success(f"为您推荐最热门的 {top_n} 部电影")

        elif rec_method == "高评分推荐":
            min_votes = st.slider("最低评分人数", 10, 200, 50)
            result = recommender.rating_recommend(min_votes=min_votes, top_n=top_n)
            st.success(f"为您推荐评分最高的 {top_n} 部电影（至少{min_votes}人评价）")

        elif rec_method == "基于内容推荐":
            movie_list = movies[["movieId", "title"]].values.tolist()
            selected = st.selectbox(
                "选择一部电影，基于类型推荐相似电影",
                movie_list,
                format_func=lambda x: f"{x[1]} (ID: {x[0]})"
            )
            result = recommender.content_based_recommend(selected[0], top_n=top_n)
            st.success(f"基于《{selected[1]}》的类型为您推荐")

        elif rec_method == "物品协同过滤":
            movie_list = movies[["movieId", "title"]].values.tolist()
            selected = st.selectbox(
                "选择一部电影，基于用户行为推荐相似电影",
                movie_list,
                format_func=lambda x: f"{x[1]} (ID: {x[0]})"
            )
            result = recommender.item_based_cf(selected[0], top_n=top_n)
            st.success(f"看过《{selected[1]}》的人还喜欢")

        elif rec_method == "用户协同过滤":
            max_user = ratings["userId"].max()
            user_id = st.number_input("输入用户ID", min_value=1, max_value=int(max_user), value=1, step=1)
            result = recommender.user_based_cf(int(user_id), top_n=top_n)
            st.success(f"为用户 {user_id} 推荐的电影")

        elif rec_method == "混合推荐":
            max_user = ratings["userId"].max()
            user_id = st.number_input("输入用户ID", min_value=1, max_value=int(max_user), value=1, step=1)
            result = recommender.hybrid_recommend(int(user_id), top_n=top_n)
            st.success(f"为用户 {user_id} 的混合推荐结果")

        if not result.empty:
            st.dataframe(result, use_container_width=True, hide_index=True)
        else:
            st.warning("未找到推荐结果")

    # ========== 页面5: 用户画像 ==========
    elif page == "👤 用户画像":
        st.title("用户画像分析")
        st.markdown("---")

        max_user = ratings["userId"].max()
        user_id = st.number_input("输入要分析的用户ID", min_value=1, max_value=int(max_user), value=1, step=1)

        user_ratings = ratings[ratings["userId"] == user_id]

        if len(user_ratings) == 0:
            st.error("该用户不存在")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("评分总数", len(user_ratings))
            col2.metric("平均评分", round(user_ratings["rating"].mean(), 3))
            col3.metric("评分标准差", round(user_ratings["rating"].std(), 3) if len(user_ratings) > 1 else "-")

            st.subheader("评分最高的5部电影")
            top_rated = user_ratings.sort_values("rating", ascending=False).head(5)
            merged = top_rated.merge(movies[["movieId", "title", "genres"]], on="movieId")
            st.dataframe(merged[["title", "rating", "genres"]], use_container_width=True, hide_index=True)

            st.subheader("偏好类型分布")
            genre_count = defaultdict(int)
            merged_all = user_ratings.merge(movies[["movieId", "genres_list"]], on="movieId")
            for _, row in merged_all.iterrows():
                if isinstance(row["genres_list"], list):
                    for g in row["genres_list"]:
                        genre_count[g] += 1

            genre_series = pd.Series(genre_count).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            genre_series.plot(kind="barh", color="#55a868", ax=ax)
            ax.set_xlabel("观看数量")
            st.pyplot(fig)
            plt.close()

            st.subheader("评分分布")
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            user_ratings["rating"].hist(bins=10, color="#4c72b0", edgecolor="white", ax=ax2)
            ax2.set_xlabel("评分")
            ax2.set_ylabel("电影数量")
            st.pyplot(fig2)
            plt.close()


if __name__ == "__main__":
    main()