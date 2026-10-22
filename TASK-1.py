import pandas as pd
df = pd.read_csv(r"C:\Users\ANSHUMAN\OneDrive\Desktop\gen-ai assignment-20\indian_stock_prediction_recommendation_5000.csv")

print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe(include="all"))

print("\nText Columns Used for Recommendations:")
text_columns = ["company_symbol", "ticker", "recommendation_label", "technical_signal"]
print(text_columns)

print("\nText Column Data Types:")
print(df[text_columns].dtypes)
