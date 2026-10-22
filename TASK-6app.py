import streamlit as st
import pandas as pd
from scipy.sparse import load_npz
from pathlib import Path

st.set_page_config(page_title="Stock Recommendation System",  layout="wide")
st.title(" Stock Recommendation System")
st.write("Select a stock and generate similar-stock recommendations using TF-IDF cosine similarity.")

DATA_FILE = "indian_stock_prediction_recommendation_5000.csv"
SIMILARITY_FILE = "stock_similarity_matrix.npz"

@st.cache_data
def load_dataset(path):
    return pd.read_csv(path).reset_index(drop=True)

@st.cache_resource
def load_similarity(path):
    return load_npz(path)

missing = [f for f in (DATA_FILE, SIMILARITY_FILE) if not Path(f).exists()]
if missing:
    st.error("Missing required file(s): " + ", ".join(missing))
    st.info("Place both files in the same folder as app.py. Run your Part 2 script first to create stock_similarity_matrix.npz.")
    st.stop()

df = load_dataset(DATA_FILE)
similarity = load_similarity(SIMILARITY_FILE)

required = {"company_symbol", "ticker"}
if not required.issubset(df.columns):
    st.error("CSV must include company_symbol and ticker columns.")
    st.stop()
if similarity.shape[0] != len(df):
    st.error(f"CSV has {len(df)} rows but matrix has {similarity.shape[0]} rows. Re-run Part 2 on this CSV.")
    st.stop()

work = df.copy()
work["_row_index"] = work.index
if "date" in work.columns:
    work["_date"] = pd.to_datetime(work["date"], errors="coerce")
    latest = work.sort_values("_date").groupby("company_symbol", sort=True).tail(1)
else:
    latest = work.groupby("company_symbol", sort=True).tail(1)
latest = latest.sort_values("company_symbol")

symbols = latest["company_symbol"].astype(str).tolist()
if not symbols:
    st.warning("No stock symbols found in the CSV.")
    st.stop()

col1, col2 = st.columns([3, 1])
with col1:
    selected = st.selectbox("Choose a stock", symbols)
with col2:
    top_n = st.selectbox("Number of recommendations", [3, 5, 10], index=1)

if st.button("Generate Recommendations", type="primary"):
    selected_row = latest[latest["company_symbol"].astype(str) == selected].iloc[0]
    query_idx = int(selected_row["_row_index"])
    candidate_indices = latest["_row_index"].to_numpy(dtype=int)
    scores = similarity[query_idx, candidate_indices].toarray().ravel()

    result = pd.DataFrame({
        "Company": latest["company_symbol"].astype(str).to_numpy(),
        "Ticker": latest["ticker"].astype(str).to_numpy(),
        "Similarity Score": scores,
        "_row_index": candidate_indices
    })
    result = result[result["Company"] != selected]
    result = result.sort_values("Similarity Score", ascending=False).head(int(top_n))

    for col, label in [("recommendation_label", "Recommendation"), ("technical_signal", "Technical Signal"), ("Close", "Latest Close")]:
        if col in df.columns:
            result[label] = result["_row_index"].map(df[col])

    result = result.drop(columns=["_row_index"])
    result["Similarity Score"] = result["Similarity Score"].round(4)
    if "Latest Close" in result.columns:
        result["Latest Close"] = pd.to_numeric(result["Latest Close"], errors="coerce").round(2)

    st.subheader(f"Top stocks similar to {selected}")
    st.caption("Similarity is based on text features, not a forecast or personalized investment advice.")
    st.dataframe(result, use_container_width=True, hide_index=True)
    st.subheader("Similarity score chart")
    st.bar_chart(result.set_index("Company")[["Similarity Score"]])

st.divider()
st.caption("Educational project only. Text similarity does not establish whether a stock is a good investment.")
