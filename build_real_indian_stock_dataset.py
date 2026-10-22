"""
Real Indian Stock Prediction + Recommendation Dataset Builder
============================================================
Creates up to 5,000 REAL historical daily stock records from Yahoo Finance.

Output:
  indian_stock_prediction_recommendation_5000.csv

Important:
- Run with internet access.
- Yahoo Finance data may be delayed, unavailable, or rate-limited.
- The latest date is the latest available trading session, not necessarily today's
  unfinished/live price.
- This is for educational research, not investment advice.
- Buy/Hold/Avoid training labels are rule-based labels derived from subsequent
  5-trading-day returns; they are not analyst recommendations.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# 100 real Indian listed companies. Yahoo Finance symbols use the .NS suffix.
# Some symbols may change or be unavailable; the script reports what downloaded.
SYMBOLS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL",
    "ITC", "TCS", "SBIN", "LICI", "HINDUNILVR",
    "LT", "BAJFINANCE", "HCLTECH", "MARUTI", "SUNPHARMA",
    "KOTAKBANK", "AXISBANK", "TITAN", "NTPC", "ONGC",
    "TATAMOTORS", "TATASTEEL", "M&M", "POWERGRID", "ULTRACEMCO",
    "ASIANPAINT", "WIPRO", "ADANIENT", "ADANIPORTS", "COALINDIA",
    "JSWSTEEL", "NESTLEIND", "TECHM", "BAJAJFINSV", "GRASIM",
    "INDUSINDBK", "HINDALCO", "DRREDDY", "CIPLA", "EICHERMOT",
    "APOLLOHOSP", "DIVISLAB", "BPCL", "BRITANNIA", "HEROMOTOCO",
    "TATACONSUM", "SHRIRAMFIN", "HDFCLIFE", "SBILIFE", "TRENT",
    "BEL", "HAL", "IRCTC", "DLF", "PIDILITIND",
    "DMART", "GODREJCP", "SIEMENS", "INDIGO", "VEDL",
    "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "IDFCFIRSTB",
    "FEDERALBNK", "CHOLAFIN", "MUTHOOTFIN", "BSE", "MCX",
    "PAYTM", "ZOMATO", "JIOFIN", "POLICYBZR", "NAUKRI",
    "PERSISTENT", "COFORGE", "MPHASIS", "LTIM", "OFSS",
    "TORNTPHARM", "ALKEM", "LUPIN", "AUROPHARMA", "BIOCON",
    "DABUR", "MARICO", "COLPAL", "BERGEPAINT", "KANSAINER",
    "AMBUJACEM", "ACC", "SHREECEM", "TVSMOTOR", "BAJAJ-AUTO",
    "ASHOKLEY", "IOC", "GAIL", "PETRONET", "INDUSTOWER"
]
TICKERS = [s + ".NS" for s in SYMBOLS]
PERIOD = "1y"
ROWS_REQUESTED = 5000
OUTPUT = "indian_stock_prediction_recommendation_5000.csv"


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build():
    print(f"Downloading real historical prices for {len(TICKERS)} NSE symbols...")
    raw = yf.download(
        TICKERS,
        period=PERIOD,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False
    )

    frames = []
    for symbol in TICKERS:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if symbol not in raw.columns.get_level_values(0):
                    continue
                df = raw[symbol].copy()
            else:
                # Fallback for a single ticker response
                df = raw.copy()

            needed = {"Open", "High", "Low", "Close", "Volume"}
            if not needed.issubset(df.columns):
                continue

            df = df.dropna(subset=["Close"]).copy()
            if len(df) < 35:
                print(f"Skipping {symbol}: insufficient price history")
                continue

            df.index = pd.to_datetime(df.index)
            df["ticker"] = symbol
            df["date"] = df.index.date
            close = df["Close"]
            volume = df["Volume"]

            # Features available at the close of the given date
            df["daily_return_pct"] = close.pct_change() * 100
            df["return_5d_pct"] = close.pct_change(5) * 100
            df["return_20d_pct"] = close.pct_change(20) * 100
            df["sma_5"] = close.rolling(5).mean()
            df["sma_20"] = close.rolling(20).mean()
            df["sma_50"] = close.rolling(50).mean()
            df["price_vs_sma20_pct"] = (close / df["sma_20"] - 1) * 100
            df["volatility_20d_pct"] = (
                df["daily_return_pct"].rolling(20).std() * np.sqrt(252)
            )
            df["rsi_14"] = rsi(close, 14)
            df["volume_sma_20"] = volume.rolling(20).mean()
            df["volume_ratio_20d"] = volume / df["volume_sma_20"]
            df["high_low_range_pct"] = (df["High"] - df["Low"]) / close * 100

            # Target: return over the NEXT five trading sessions.
            # This target is shifted backward so features on date t predict t+5.
            df["future_5d_return_pct"] = (close.shift(-5) / close - 1) * 100
            df["prediction_label"] = np.select(
                [
                    df["future_5d_return_pct"] > 2,
                    df["future_5d_return_pct"] < -2
                ],
                ["UP", "DOWN"],
                default="SIDEWAYS"
            )

            # Educational label from subsequent returns, not a real analyst rating.
            df["recommendation_label"] = np.select(
                [
                    df["future_5d_return_pct"] >= 3,
                    df["future_5d_return_pct"] <= -3
                ],
                ["Buy", "Avoid"],
                default="Hold"
            )

            # A transparent current-date heuristic score for recommendation demos.
            # It does not use future return; future labels are only for supervised training.
            score = (
                0.35 * df["return_5d_pct"].clip(-10, 10).fillna(0)
                + 0.25 * df["return_20d_pct"].clip(-20, 20).fillna(0)
                + 0.20 * df["price_vs_sma20_pct"].clip(-10, 10).fillna(0)
                + 0.20 * ((df["rsi_14"].fillna(50) - 50) / 5).clip(-10, 10)
            )
            df["technical_signal"] = np.select(
                [score >= 2.0, score <= -2.0],
                ["Positive", "Negative"],
                default="Neutral"
            )

            df["company_symbol"] = symbol.replace(".NS", "")
            frames.append(df.reset_index(drop=True))
            print(f"Downloaded {symbol}: {len(df)} rows")
        except Exception as exc:
            print(f"Skipping {symbol}: {exc}")

    if not frames:
        raise RuntimeError(
            "No data downloaded. Check internet access, install/update yfinance, "
            "and try again later."
        )

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["date", "ticker"])
    out = out.dropna(subset=[
        "daily_return_pct", "return_5d_pct", "return_20d_pct",
        "sma_20", "rsi_14", "future_5d_return_pct"
    ])

    # Use most recent eligible observations, then cap at 5,000 rows.
    out = out.tail(ROWS_REQUESTED).copy()

    columns = [
        "date", "company_symbol", "ticker", "Open", "High", "Low", "Close",
        "Volume", "daily_return_pct", "return_5d_pct", "return_20d_pct",
        "sma_5", "sma_20", "sma_50", "price_vs_sma20_pct",
        "volatility_20d_pct", "rsi_14", "volume_sma_20", "volume_ratio_20d",
        "high_low_range_pct", "future_5d_return_pct", "prediction_label",
        "recommendation_label", "technical_signal"
    ]
    out = out[columns].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["future_5d_return_pct"])
    out.to_csv(OUTPUT, index=False)

    print("\nDATASET CREATED")
    print(f"File: {OUTPUT}")
    print(f"Rows saved: {len(out):,} (maximum requested: {ROWS_REQUESTED:,})")
    print(f"Companies represented: {out['company_symbol'].nunique()}")
    print(f"Date range: {out['date'].min()} to {out['date'].max()}")
    print("\nPrediction labels:")
    print(out["prediction_label"].value_counts().to_string())
    print("\nRecommendation labels:")
    print(out["recommendation_label"].value_counts().to_string())
    print("\nFirst five rows:")
    print(out.head().to_string(index=False))
    print(
        "\nNote: The latest rows cannot have a future 5-day target until five "
        "more trading sessions have elapsed. This training file therefore ends "
        "five trading sessions before the latest downloaded close for each stock. "
        "Use the most recent features without future labels for live inference."
    )


if __name__ == "__main__":
    build()
