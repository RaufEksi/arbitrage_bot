"""
xgb_hft_model.py -- XGBoost Directional Prediction Model for HFT.

Trains a multi-class XGBoost classifier to predict short-term price
breakouts using Order Flow Imbalance (OFI) and Order Book Imbalance (OBI)
features with rolling Z-score normalization.

Usage:
    python xgb_hft_model.py
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import warnings

# Suppress only XGBoost feature name warnings (not all warnings)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


def prepare_data(
    csv_path: str = "data/btcusdt_ofi_data.csv",
    horizon: int = 10,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Loads raw tick data and engineers features + forward-looking labels.

    Args:
        csv_path: Path to the bookTicker CSV data.
        horizon: Number of ticks to look ahead for label generation.
        threshold: Minimum price movement (USDT) to classify as Buy/Sell.

    Returns:
        DataFrame with features and labels, warmup rows dropped.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. Feature Engineering
    print("Computing features (OFI, OBI, Z-Scores, Volatility)...")
    df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2.0
    df["spread"] = df["ask_price"] - df["bid_price"]

    # OBI (Order Book Imbalance)
    bid_qty = df["bid_qty"]
    ask_qty = df["ask_qty"]
    df["obi"] = (bid_qty - ask_qty) / (bid_qty + ask_qty + 1e-8)

    # Exponential Moving Averages
    df["ofi_ema"] = df["ofi"].ewm(span=20, adjust=False).mean()
    df["obi_ema"] = df["obi"].ewm(span=20, adjust=False).mean()

    # Rolling Z-scores (window=100)
    z_window = 100
    ofi_mean = df["ofi_ema"].rolling(window=z_window, min_periods=1).mean()
    ofi_std = df["ofi_ema"].rolling(window=z_window, min_periods=1).std().replace(0, 1e-8)
    df["ofi_z"] = (df["ofi_ema"] - ofi_mean) / ofi_std

    obi_mean = df["obi_ema"].rolling(window=z_window, min_periods=1).mean()
    obi_std = df["obi_ema"].rolling(window=z_window, min_periods=1).std().replace(0, 1e-8)
    df["obi_z"] = (df["obi_ema"] - obi_mean) / obi_std

    # Volatility
    df["volatility"] = df["mid_price"].rolling(window=z_window, min_periods=1).std().fillna(0)

    # 2. Target Generation (forward-looking labels)
    print(f"Generating labels (horizon={horizon} ticks, threshold={threshold} USDT)...")
    df["future_mid"] = df["mid_price"].shift(-horizon)
    df["future_bid"] = df["bid_price"].shift(-horizon)
    df["future_ask"] = df["ask_price"].shift(-horizon)

    # Label: 1 (Buy) if price rises by threshold, 2 (Sell) if drops, 0 (Hold) otherwise
    conditions = [
        (df["future_mid"] > df["mid_price"] + threshold),
        (df["future_mid"] < df["mid_price"] - threshold),
    ]
    choices = [1, 2]
    df["label"] = np.select(conditions, choices, default=0)

    # 3. Drop warmup + lookahead tail to prevent data leakage
    initial_len = len(df)
    df = df.iloc[z_window:-horizon].reset_index(drop=True)
    print(f"Dropped {z_window} warmup + {horizon} tail rows. Valid: {len(df):,} / {initial_len:,}")

    return df


def train_and_evaluate(df: pd.DataFrame) -> None:
    """
    Trains XGBoost on chronological split and evaluates on out-of-sample data.
    Saves the model and generates an equity curve plot.
    """
    # Features and target
    features = ["ofi", "obi", "ofi_ema", "obi_ema", "ofi_z", "obi_z", "volatility", "spread"]
    X = df[features]
    y = df["label"]

    # Chronological train/test split (80/20) — NO shuffling to prevent leakage
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    df_test = df.iloc[split_idx:].copy().reset_index(drop=True)

    print(f"\nChronological split: train={len(X_train):,}, test={len(X_test):,}")

    unique_classes, counts = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(["Hold", "Buy", "Sell"], counts))
    print(f"Training class distribution: {class_dist}")

    # Compute sample weights to handle class imbalance
    print("Computing balanced class weights...")
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    # Initialize XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        objective="multi:softmax",
        num_class=3,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )

    print("Training XGBoost model...")
    model.fit(X_train, y_train, sample_weight=sample_weights)

    print("Evaluating on out-of-sample test set...")
    y_pred = model.predict(X_test)

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT (OUT-OF-SAMPLE)")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=["Hold (0)", "Buy (1)", "Sell (2)"]))

    # Vectorized backtest (raw model performance)
    df_test["pred"] = y_pred

    # Buy PnL: enter at ask, exit at future_bid
    buy_pnl = np.where(df_test["pred"] == 1, df_test["future_bid"] - df_test["ask_price"], 0)

    # Sell PnL: enter at bid, exit at future_ask
    sell_pnl = np.where(df_test["pred"] == 2, df_test["bid_price"] - df_test["future_ask"], 0)

    df_test["strategy_pnl"] = buy_pnl + sell_pnl

    total_trades = (df_test["pred"] > 0).sum()
    gross_profit = df_test["strategy_pnl"].sum()

    print("=" * 50)
    print("VECTORIZED BACKTEST (No Commission)")
    print("=" * 50)
    print(f"Total predicted trades: {total_trades:,}")
    if total_trades > 0:
        win_rate = (df_test["strategy_pnl"] > 0).sum() / total_trades * 100
        print(f"Win rate: {win_rate:.2f}%")
        print(f"Avg PnL per trade: {gross_profit / total_trades:.4f} USDT")
    print(f"Gross PnL (spread-adjusted): {gross_profit:.2f} USDT")

    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(df_test["strategy_pnl"].cumsum(), color="purple", label="XGBoost Strategy PnL")
    plt.title("XGBoost Vectorized Backtest — Equity Curve (Out-of-Sample)", fontsize=14)
    plt.xlabel("Ticks", fontsize=12)
    plt.ylabel("Cumulative PnL (USDT)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("xgb_equity_curve.png")
    print("-> Equity curve saved to 'xgb_equity_curve.png'")

    # Save model for live trading
    os.makedirs("models", exist_ok=True)
    model_path = "models/xgb_best.json"
    model.save_model(model_path)
    print(f"-> Model saved to '{model_path}' (ready for live_trader.py)")


if __name__ == "__main__":
    # Parameters: 10-tick lookahead, 1.0 USDT threshold
    # At BTC ~$70K, 1.0 USDT is a reasonable noise filter
    data = prepare_data("data/btcusdt_ofi_data.csv", horizon=10, threshold=1.0)
    train_and_evaluate(data)
