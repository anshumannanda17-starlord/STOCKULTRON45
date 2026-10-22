
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download("stopwords", quiet=True)

# 1. Load the uploaded stock dataset
df = pd.read_csv("indian_stock_prediction_recommendation_5000.csv")

# 2. Select the text column for preprocessing
text_column = "technical_signal"

# 3. Handle missing values
df[text_column] = df[text_column].fillna("")

# Load English stopwords
stop_words = set(stopwords.words("english"))

# 4. Define the preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = str(text).lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# 5. Store cleaned text in a new column
df["clean_text"] = df[text_column].apply(preprocess_text)

# 6. Display original and cleaned text
print("Original and Cleaned Text:")
print(df[[text_column, "clean_text"]].head(10))

# 7. Save the cleaned dataset
df.to_csv(
    "indian_stock_prediction_recommendation_5000_cleaned.csv",
    index=False
)

print("\nDataset shape:", df.shape)
print("Cleaned dataset saved successfully!")
