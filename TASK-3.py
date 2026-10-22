import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

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