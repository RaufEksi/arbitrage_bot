import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import warnings

# Suppress annoying warnings
warnings.filterwarnings('ignore')

def prepare_data(csv_path="data/btcusdt_ofi_data.csv", horizon=10, threshold=1.0):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Feature Engineering
    print("Calculating Technical Features (OFI, OBI, Z-Scores, Volatility)...")
    df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2.0
    df['spread'] = df['ask_price'] - df['bid_price']
    
    # OBI (Order Book Imbalance)
    bid_qty = df['bid_qty']
    ask_qty = df['ask_qty']
    df['obi'] = (bid_qty - ask_qty) / (bid_qty + ask_qty + 1e-8)
    
    # Fast EMA calculations
    df['ofi_ema'] = df['ofi'].ewm(span=20, adjust=False).mean()
    df['obi_ema'] = df['obi'].ewm(span=20, adjust=False).mean()
    
    # Rolling Z-scores (Window=100)
    z_window = 100
    ofi_mean = df['ofi_ema'].rolling(window=z_window, min_periods=1).mean()
    ofi_std = df['ofi_ema'].rolling(window=z_window, min_periods=1).std().replace(0, 1e-8)
    df['ofi_z'] = (df['ofi_ema'] - ofi_mean) / ofi_std
    
    obi_mean = df['obi_ema'].rolling(window=z_window, min_periods=1).mean()
    obi_std = df['obi_ema'].rolling(window=z_window, min_periods=1).std().replace(0, 1e-8)
    df['obi_z'] = (df['obi_ema'] - obi_mean) / obi_std
    
    # Volatility
    df['volatility'] = df['mid_price'].rolling(window=z_window, min_periods=1).std().fillna(0)
    
    # 2. Target Generation (Lookahead Window)
    print(f"Generating forward-looking labels (Horizon={horizon} ticks, Threshold={threshold} USDT)...")
    df['future_mid'] = df['mid_price'].shift(-horizon)
    df['future_bid'] = df['bid_price'].shift(-horizon)
    df['future_ask'] = df['ask_price'].shift(-horizon)
    
    # Conditions for labeling
    # 1 (Buy): Future mid price goes up by at least threshold
    # 2 (Sell): Future mid price goes down by at least threshold
    # 0 (Hold): Sideways
    conditions = [
        (df['future_mid'] > df['mid_price'] + threshold),
        (df['future_mid'] < df['mid_price'] - threshold)
    ]
    choices = [1, 2]
    df['label'] = np.select(conditions, choices, default=0)
    
    # 3. Clean NaNs (Warmup + Shift leakage prevention)
    initial_len = len(df)
    df = df.iloc[z_window:-horizon].reset_index(drop=True)
    print(f"Dropped {z_window} warmup rows and {horizon} tail rows. Valid rows: {len(df)} / {initial_len}")
    
    return df

def train_and_evaluate(df):
    # Features X and Target y
    features = ['ofi', 'obi', 'ofi_ema', 'obi_ema', 'ofi_z', 'obi_z', 'volatility', 'spread']
    X = df[features]
    y = df['label']
    
    # Chronological Train/Test Split (80/20) - NO RANDOM SHUFFLING (Leakage protection)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    df_test = df.iloc[split_idx:].copy().reset_index(drop=True)
    
    print(f"\nTime Series Split: Training on first {len(X_train):,} samples, Testing on last {len(X_test):,} samples.")
    
    unique_classes, counts = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(['Hold', 'Buy', 'Sell'], counts))
    print(f"Class distribution in Train: {class_dist}")
    
    # Compute Class Weights to fix extreme "Hold" bias
    print("Computing class weights to handle imbalance...")
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    # Initialize XGBoost Classifier
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        objective='multi:softmax',
        num_class=3,
        random_state=42,
        tree_method='hist', # fast histogram algorithm
        n_jobs=-1
    )
    
    print("Training XGBoost Model (This might take a minute)...")
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    print("Evaluating Model on Out-of-Sample Test Set...")
    y_pred = model.predict(X_test)
    
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT (OOS TEST SET)")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=['Hold (0)', 'Buy (1)', 'Sell (2)']))
    
    # =======================================================
    # Vektörel Backtest (Ham Model Performansı)
    # =======================================================
    df_test['pred'] = y_pred
    
    # Buy PnL: Modeli dinleyip Ask'tan aldık, Horizon (10) saniye sonra Bid'den (future_bid) sattık
    buy_pnl = np.where(df_test['pred'] == 1, df_test['future_bid'] - df_test['ask_price'], 0)
    
    # Sell PnL: Modeli dinleyip Bid'den açığa sattık, Horizon (10) saniye sonra Ask'tan (future_ask) yerine koyduk
    sell_pnl = np.where(df_test['pred'] == 2, df_test['bid_price'] - df_test['future_ask'], 0)
    
    df_test['strategy_pnl'] = buy_pnl + sell_pnl
    
    total_trades = (df_test['pred'] > 0).sum()
    gross_profit = df_test['strategy_pnl'].sum()
    
    print("="*50)
    print("VECTORIZED BACKTEST RESULTS (No Commission)")
    print("="*50)
    print(f"Total Predicted Trades: {total_trades:,}")
    if total_trades > 0:
        win_rate = (df_test['strategy_pnl'] > 0).sum() / total_trades * 100
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average PnL per trade: {gross_profit / total_trades:.4f} USDT")
    print(f"Gross PnL (Spread maliyeti dahil): {gross_profit:.2f} USDT")
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df_test['strategy_pnl'].cumsum(), color='purple', label='XGBoost Strategy PnL')
    plt.title('XGBoost Vectorized Backtest Equity Curve (Out-of-Sample Test Set)', fontsize=14)
    plt.xlabel('Ticks', fontsize=12)
    plt.ylabel('Cumulative PnL (USDT)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('xgb_equity_curve.png')
    print("-> Saved equity curve exactly representing model edges as 'xgb_equity_curve.png'.")
    
    # Save Model for Live Trading
    model_path = "models/xgb_best.json"
    model.save_model(model_path)
    print(f"-> Model başarıyla {model_path} konumuna kaydedildi. (live_trader.py için hazır!)")

if __name__ == "__main__":
    # Parametreleri ayarlayabilirsiniz (Örn: 10 tick sonram %1.0 hareket threshold'u)
    # Bitcoin fiyati 70,000$ civarinda, 1$ (1.0) iyi bir gurultu esigidir.
    data = prepare_data("data/btcusdt_ofi_data.csv", horizon=10, threshold=1.0)
    train_and_evaluate(data)
