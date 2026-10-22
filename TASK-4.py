import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz
file_path = "indian_stock_prediction_recommendation_5000.csv"
df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)



if "clean_text" in df.columns:
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
else:
    text_columns = [
        col for col in [
            "company_symbol",
            "ticker",
            "prediction_label",
            "recommendation_label",
            "technical_signal"
        ]
        if col in df.columns
    ]

    df["clean_text"] = (
        df[text_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
tfidf = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf.fit_transform(df["clean_text"])

print("\nTF-IDF VECTORIZATION")
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
print("Number of Features:", len(tfidf.get_feature_names_out()))
print("\nFirst 20 TF-IDF Features:")
print(tfidf.get_feature_names_out()[:20])


similarity_matrix = cosine_similarity(
    tfidf_matrix,
    dense_output=False
)

print("\n COSINE SIMILARITY")
print("Similarity Matrix Shape:", similarity_matrix.shape)

# Save the complete similarity matrix efficiently
save_npz("stock_similarity_matrix.npz", similarity_matrix)

# Save a small preview of the similarity matrix for inspection
preview = cosine_similarity(tfidf_matrix[:10], tfidf_matrix)
pd.DataFrame(preview).to_csv(
    "stock_similarity_preview.csv",
    index=False
)

# Display similarities for the first stock record
first_similarities = cosine_similarity(
    tfidf_matrix[0:1],
    tfidf_matrix
).flatten()

# Exclude the stock record itself
first_similarities[0] = -1

top_indices = np.argsort(first_similarities)[::-1][:5]

print("\nTop 5 Similar Records to the First Record:")
for rank, idx in enumerate(top_indices, start=1):
    print(
        f"{rank}. {df.iloc[idx].get('company_symbol', 'Unknown')} "
        f"| Similarity: {first_similarities[idx]:.4f}"
    )

print("\nFiles saved:")
print("1. stock_similarity_matrix.npz")
print("2. stock_similarity_preview.csv")