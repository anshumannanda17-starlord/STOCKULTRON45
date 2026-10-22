
# Stock Recommendation System

A Python-based stock recommendation project using TF-IDF vectorization,
cosine similarity, and a Streamlit interface.

## Features
- Stock dataset loading and preprocessing
- TF-IDF text vectorization
- Cosine similarity computation
- Top-N similar-stock recommendations
- Streamlit user interface

## Requirements
Python 3.10 or newer.

## Installation
```bash
python -m pip install -r requirements.txt
```

## Generate the dataset
```bash
python build_real_indian_stock_dataset.py
```

## Run the Streamlit application
```bash
python -m streamlit run app.py
```

## Dataset
The dataset contains historical market data downloaded from Yahoo Finance.
The generated recommendation labels are for educational use only.

## Disclaimer
This project is for educational purposes and does not provide financial advice.
