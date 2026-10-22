
import pandas as pd
import numpy as np
from scipy.sparse import load_npz

# Load the dataset and similarity matrix from Part 2
df = pd.read_csv("indian_stock_prediction_recommendation_5000.csv")
similarity_matrix = load_npz("stock_similarity_matrix.npz")

def recommend(item_name, top_n=5):
    """
    Return the top N most similar stocks to the selected stock.
    item_name can be a company symbol or ticker, e.g. RELIANCE or RELIANCE.NS.
    """

    item_name = item_name.strip().upper()

    # Find the selected company
    symbols = df["company_symbol"].astype(str).str.upper()
    tickers = df["ticker"].astype(str).str.upper()

    matches = (symbols == item_name) | (tickers == item_name)

    if not matches.any():
        print(f"Stock '{item_name}' not found in the dataset.")
        return pd.DataFrame()

    # Select the latest available record for the requested company
    selected_indices = df.index[matches]
    query_idx = selected_indices[-1]

    query_symbol = df.loc[query_idx, "company_symbol"]

    # Use the latest record for each other company
    latest_indices = (
        df.reset_index()
          .sort_values("date")
          .groupby("company_symbol")["index"]
          .last()
          .tolist()
    )

    # Calculate similarity against each company's latest record
    scores = similarity_matrix[query_idx, latest_indices].toarray().ravel()

    results = pd.DataFrame({
        "row_index": latest_indices,
        "similarity_score": scores
    })

    # Add company details
    results["company_symbol"] = results["row_index"].map(
        df["company_symbol"]
    )
    results["ticker"] = results["row_index"].map(df["ticker"])

    # Exclude the selected company itself
    results = results[
        results["company_symbol"] != query_symbol
    ]

    # Sort by similarity and return top N stocks
    results = results.sort_values(
        "similarity_score", ascending=False
    ).head(top_n)

    # Include available recommendation labels
    if "recommendation_label" in df.columns:
        results["recommendation_label"] = results["row_index"].map(
            df["recommendation_label"]
        )

    return results[
        [
            "company_symbol",
            "ticker",
            "similarity_score",
            *(
                ["recommendation_label"]
                if "recommendation_label" in results.columns
                else []
            )
        ]
    ].reset_index(drop=True)


# Test with at least three different stocks
test_stocks = ["RELIANCE", "TCS", "INFY"]

for stock in test_stocks:
    print(f"\nRecommendations for {stock}:")
    print(recommend(stock, top_n=5).to_string(index=False))
