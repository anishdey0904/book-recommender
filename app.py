import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="wide"
)

# ---------------------------------------------------
# CSS (FIXED ✅)
# ---------------------------------------------------
st.markdown("""
<style>

/* GLOBAL */
.stApp{
    background:#F7F5F2;
}

.block-container{
    max-width:90rem;
    padding-top:2rem;
}

/* HEADER */
.header{
    text-align:center;
    margin-bottom:2rem;
}
.header h1{
    font-size:3.5rem;
    font-weight:800;
    color:#111827;
}
.header p{
    color:#6B7280;
}

/* INPUT */
.stTextInput input{
    border-radius:16px;
    padding:14px;
}

/* BUTTON */
.stButton button{
    width:100%;
    height:3rem;
    border-radius:16px;
    background:#111827;
    color:white;
}

.book-card{
    background:white;
    border-radius:20px;
    padding:12px;

    height:300px;

    display:flex;
    flex-direction:column;
    justify-content:space-between;

    border:1px solid #E5E7EB;
    margin-bottom:7px;

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease;
            
    position:relative;   /* IMPORTANT */
    z-index:1;
}

.book-card:hover{
    transform:scale(1.05);

    box-shadow:
        0 12px 24px rgba(0,0,0,0.12),
        0 20px 40px rgba(0,0,0,0.08);

    z-index:9999;
}
            
/* IMAGE */
.book-cover{
    height:180px;   /* ✅ smaller image */
    
    display:flex;
    justify-content:center;
    align-items:center;

    margin-bottom:8px;
}

.book-cover img{
    max-height:100%;
    max-width:100%;
    object-fit:contain;
    border-radius:10px;
}

.book-title{
    font-size:0.95rem;
    font-weight:700;
    line-height:1.3;

    display:-webkit-box;
    -webkit-line-clamp:2;      /* Show max 2 lines */
    -webkit-box-orient:vertical;

    overflow:hidden;
    text-overflow:ellipsis;

    min-height:40px;           /* Reserve space for 2 lines */
}
            /* Tooltip */
.book-title:hover::after{
    content: attr(data-title);

    position:absolute;
    left:0;
    top:100%;

    z-index:9999;

    background:#111827;
    color:white;

    padding:8px 10px;
    border-radius:8px;

    width:max-content;
    max-width:250px;

    white-space:normal;
    word-wrap:break-word;

    box-shadow:0 4px 12px rgba(0,0,0,0.15);
}
            
.book-author{
    font-size:0.85rem;
    margin-top:4px;
}

.book-meta{
    color:#D97706;
    font-size:0.85rem;
    margin-top:10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    books = pd.read_csv("Books.csv")
    ratings = pd.read_csv("Ratings.csv")
    users = pd.read_csv("Users.csv")

    rated_books = ratings.merge(books, on="ISBN")

    num = rated_books.groupby("Book-Title")["Book-Rating"].count().reset_index()
    num.rename(columns={"Book-Rating": "total_ratings"}, inplace=True)

    avg = rated_books.groupby("Book-Title")["Book-Rating"].mean().reset_index()
    avg.rename(columns={"Book-Rating": "avg_ratings"}, inplace=True)

    popularity_df = num.merge(avg, on="Book-Title")

    popularity_df = popularity_df[popularity_df["total_ratings"] >= 250] \
        .sort_values("avg_ratings", ascending=False) \
        .head(50)

    popularity_df = popularity_df.merge(books, on="Book-Title") \
        .drop_duplicates("Book-Title")

    # Collaborative filtering
    active = rated_books.groupby("User-ID").count()["Book-Rating"]
    active = active[active > 200].index

    filtered = rated_books[rated_books["User-ID"].isin(active)]

    famous = filtered.groupby("Book-Title").count()["Book-Rating"]
    famous = famous[famous >= 50].index

    final = filtered[filtered["Book-Title"].isin(famous)]

    pivot = final.pivot_table(
        index="Book-Title",
        columns="User-ID",
        values="Book-Rating"
    ).fillna(0).astype(np.float32)

    similarity = cosine_similarity(pivot)

    return books, users, ratings, popularity_df, pivot, similarity


books, users, ratings, popularity_df, pvt_table, similarity_scores = load_data()

# ---------------------------------------------------
# RECOMMEND FUNCTION
# ---------------------------------------------------
def recommend(book):
    if book not in pvt_table.index:
        return []

    idx = np.where(pvt_table.index == book)[0][0]

    similar = sorted(
        list(enumerate(similarity_scores[idx])),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    result = []

    for i in similar:
        title = pvt_table.index[i[0]]

        temp = books[
            books["Book-Title"] == title
        ].drop_duplicates("Book-Title")

        if len(temp) == 0:
            continue

        rating_info = popularity_df[
            popularity_df["Book-Title"] == title
        ]

        avg_rating = (
            rating_info["avg_ratings"].values[0]
            if len(rating_info) > 0
            else np.nan
        )

        total_ratings = (
            rating_info["total_ratings"].values[0]
            if len(rating_info) > 0
            else 0
        )

        import math
        if math.isnan(avg_rating):
            avg_rating = 'No Ratings'
        else:
            avg_rating = round(avg_rating, 2)

        result.append({
            "title": temp["Book-Title"].values[0],
            "author": temp["Book-Author"].values[0],
            "image": temp["Image-URL-M"].values[0],
            "avg_ratings": avg_rating,
            "total_ratings": total_ratings
        })

    return result


# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
if "selected_book" not in st.session_state:
    st.session_state.selected_book = None

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("""
<div class="header">
    <h1>📚 Book Recommender</h1>
    <p>Discover your next favorite read</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# STATS
# ---------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Books", f"{len(books):,}")
c2.metric("Users", f"{len(users):,}")
c3.metric("Ratings", f"{len(ratings):,}")

st.divider()

# ---------------------------------------------------
# SEARCH (IMPROVED ✅)
# ---------------------------------------------------
st.subheader("Find Similar Books")

query = st.text_input("Type a book name", 
                      placeholder="e.g. Harry Potter and the Sorcerer's Stone")

if st.button("✨ Recommend"):
    if query.strip():
        matches = [
            b for b in pvt_table.index
            if query.lower() in b.lower()
        ]

        if matches:
            matches = sorted(matches, key=len)
            st.session_state.selected_book = matches[0]
        else:
            st.error("Book not found")
    else:
        st.warning("Enter a book name")

# ---------------------------------------------------
# RECOMMENDATIONS
# ---------------------------------------------------
if st.session_state.selected_book:
    st.success(f"Showing results for: {st.session_state.selected_book}")

    with st.spinner("Finding similar books..."):
        recs = recommend(st.session_state.selected_book)

    if recs:
        cols = st.columns(5)
        for col, b in zip(cols, recs):
            with col:
                st.markdown(f"""
                <div class="book-card">
                    <div class="book-cover">
                        <img src="{b['image']}" onerror="this.src='https://via.placeholder.com/150'">
                    </div>
                    <div class="book-title" data-title="{b['title']}">{b['title']}</div>
                    <div class="book-author">{b['author']}</div>
                    <div class="book-meta">
                    ⭐ {b['avg_ratings']}<br>
                    👥 {int(b['total_ratings'])}
                    
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No recommendations available")

st.divider()

# ---------------------------------------------------
# POPULAR BOOKS
# ---------------------------------------------------
st.subheader("🔥 Popular Books")

for i in range(0, len(popularity_df), 5):
    cols = st.columns(5)
    subset = popularity_df.iloc[i:i+5]

    for col, (_, row) in zip(cols, subset.iterrows()):
        with col:
            st.markdown(f"""
            <div class="book-card">
                <div class="book-cover">
                    <img src="{row['Image-URL-M']}" onerror="this.src='https://via.placeholder.com/150'">
                </div>
                <div class="book-title" data-title="{row['Book-Title']}">{row['Book-Title']}</div>
                <div class="book-author">{row['Book-Author']}</div>
                <div class="book-meta">
                    ⭐ {row['avg_ratings']:.2f}<br>
                    👥 {int(row['total_ratings'])}
                </div>
            </div>
            """, unsafe_allow_html=True)