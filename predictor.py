import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def calculate_technical_indicators(df):
    """
    Computes SMA (5, 20, 50), RSI, MACD, Bollinger Bands, ATR,
    momentum (ROC), EMA ratio, and price acceleration.
    """
    df = df.copy()
    
    # 1. Simple Moving Averages
    df["sma_5"] = df["close"].rolling(window=5).mean()
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    
    # 2. Relative Strength Index (RSI) - smoothed wilder style
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Wilder smoothing equivalent to ewm (com=13)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10) # Avoid division by zero
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # 3. MACD (Moving Average Convergence Divergence)
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    
    # 4. Bollinger Bands (20 day, 2 standard dev)
    df["bb_middle"] = df["close"].rolling(window=20).mean()
    df["bb_std"] = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * 2)
    df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * 2)

    # 5. Derived features for ML
    df["return_1d"] = df["close"].pct_change(1)
    df["return_2d"] = df["close"].pct_change(2)
    df["return_3d"] = df["close"].pct_change(3)
    df["return_5d"] = df["close"].pct_change(5)
    df["dist_sma5"] = df["close"] / (df["sma_5"] + 1e-10) - 1
    df["dist_sma20"] = df["close"] / (df["sma_20"] + 1e-10) - 1
    df["dist_sma50"] = df["close"] / (df["sma_50"] + 1e-10) - 1
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (bb_range + 1e-10)
    df["volume_ma20"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_ma20"] + 1e-10)
    
    # 6. NEW: Momentum / Rate-of-Change (10 days)
    df["momentum_10"] = df["close"].pct_change(10)
    
    # 7. NEW: Average True Range (ATR) — proxy for volatility
    # For gold (single price series), use rolling std of daily returns as ATR proxy
    df["atr"] = df["return_1d"].rolling(window=14).std()
    
    # 8. NEW: EMA ratio (close vs 20-day EMA) — mean-reversion signal
    ema_20 = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_ratio"] = df["close"] / (ema_20 + 1e-10) - 1
    
    # 9. NEW: Price acceleration (change in daily return) — detects trend reversals
    df["price_accel"] = df["return_1d"] - df["return_2d"]
    
    # 10. NEW: MACD histogram slope (momentum of momentum)
    df["macd_hist_slope"] = df["macd_hist"].diff()
    
    # 11. NEW: RSI slope (rate of RSI change)
    df["rsi_slope"] = df["rsi"].diff()
    
    return df


def predict_future_prices(df, days_to_predict=5):
    """
    Uses Linear Regression on the last 30 trading days
    to predict the closing prices of the next N days.
    """
    # Use last 30 days of data for local trend line
    hist_len = min(30, len(df))
    df_slice = df.tail(hist_len).copy()
    
    X = np.arange(hist_len).reshape(-1, 1)
    y = df_slice["close"].values.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next N days
    future_X = np.arange(hist_len, hist_len + days_to_predict).reshape(-1, 1)
    predictions = model.predict(future_X).flatten()
    
    # Slope (coefficient) indicating the local trend
    slope = model.coef_[0][0]
    
    # Calculate R-squared of the fit
    r_squared = model.score(X, y)
    
    return predictions, slope, r_squared


def predict_future_prices_rf(df, days_to_predict=5, params=None):
    """
    Uses Random Forest Regressor trained on technical features
    to predict the next N days' closing prices via return forecasting.
    Falls back to Linear Regression if data is insufficient.
    """
    FEATURE_COLS = [
        "rsi", "rsi_slope",
        "macd", "macd_signal", "macd_hist", "macd_hist_slope",
        "return_1d", "return_2d", "return_3d", "return_5d",
        "momentum_10", "price_accel", "atr", "ema_ratio",
        "dist_sma5", "dist_sma20", "dist_sma50",
        "bb_position", "volume_ratio"
    ]
    
    # Add macro indicators if they exist in df
    macro_cols = ["dxy", "us10y", "vix", "brent", "dji", "eurusd", "xagusd", "real_yield", "days_to_fed", "days_to_cpi", "days_to_nfp"]
    for col in macro_cols:
        if col in df.columns:
            FEATURE_COLS.append(col)
    
    # We need at least days_to_predict look-ahead for labeling, plus some rows of features
    MIN_ROWS = 15 + days_to_predict
    
    df_feat = df.copy()
    df_feat[FEATURE_COLS] = df_feat[FEATURE_COLS].ffill().bfill()  # ffill first (no future leak), bfill only for initial NaNs
    df_feat = df_feat.dropna(subset=FEATURE_COLS)
    
    if len(df_feat) < MIN_ROWS:
        # Fallback to Linear Regression
        preds, slope, r2 = predict_future_prices(df, days_to_predict)
        return preds, slope, r2, "linear_regression_fallback"
    
    current_price = df_feat["close"].iloc[-1]
    
    # --- Build training set ---
    # For each row i, target is the return from close[i] to close[i + day]
    X_list = []
    y_list = []  # multi-output: [ret_d1, ret_d2, ...]
    
    # Only use rows where we have future data for all days_to_predict
    n = len(df_feat)
    for i in range(n - days_to_predict):
        row_features = df_feat.iloc[i][FEATURE_COLS].values
        targets = []
        valid = True
        for d in range(1, days_to_predict + 1):
            if i + d >= n:
                valid = False
                break
            future_close = df_feat["close"].iloc[i + d]
            current_close = df_feat["close"].iloc[i]
            ret = (future_close - current_close) / (current_close + 1e-10)
            targets.append(ret)
        
        if not valid:
            continue
        # Skip zero-change days (weekends/holidays with repeated prices)
        # These poison the model by creating misleading zero-return labels
        if abs(targets[0]) < 1e-5:
            continue
        if not any(np.isnan(v) for v in row_features) and not any(np.isnan(t) for t in targets):
            X_list.append(row_features)
            y_list.append(targets)
    
    if len(X_list) < 10:
        # Fallback to Linear Regression
        preds, slope, r2 = predict_future_prices(df, days_to_predict)
        return preds, slope, r2, "linear_regression_fallback"
    
    X_train = np.array(X_list, dtype=np.float64)
    y_train = np.array(y_list, dtype=np.float64)
    # Sanitize: replace NaN/Inf with column median to avoid model crashes
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Train separate RF for each day ahead to avoid multi-output instability on small data
    predicted_prices = []
    r2_scores = []
    
    for d_idx in range(days_to_predict):
        max_depth = params.get("max_depth", 5) if params else 5
        min_samples_leaf = params.get("min_samples_leaf", 4) if params else 4
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            random_state=42,
            n_jobs=1
        )
        rf.fit(X_train, y_train[:, d_idx])
        
        # Score on training set (proxy for model quality; will be optimistic)
        r2 = rf.score(X_train, y_train[:, d_idx])
        r2_scores.append(r2)
        
        # Predict from the latest feature row
        latest_features = np.nan_to_num(df_feat.iloc[-1][FEATURE_COLS].values.reshape(1, -1).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        ret_pred = rf.predict(latest_features)[0]
        
        # Reconstruct absolute price from current price
        predicted_price = current_price * (1 + ret_pred)
        predicted_prices.append(predicted_price)
    
    avg_r2 = float(np.mean(r2_scores))
    
    # Slope: direction of predicted movement
    if len(predicted_prices) > 1:
        slope = (predicted_prices[-1] - current_price) / max(len(predicted_prices), 1)
    else:
        slope = predicted_prices[0] - current_price
    
    return np.array(predicted_prices), slope, avg_r2, "random_forest"


def predict_future_prices_mlp(df, days_to_predict=5, params=None):
    """
    Uses a Multi-Layer Perceptron (MLP) Neural Network (Deep Learning) Regressor
    to predict the next N days' prices via return forecasting.
    Scales features to stabilize neural network training.
    """
    FEATURE_COLS = [
        "rsi", "rsi_slope",
        "macd", "macd_signal", "macd_hist", "macd_hist_slope",
        "return_1d", "return_2d", "return_3d", "return_5d",
        "momentum_10", "price_accel", "atr", "ema_ratio",
        "dist_sma5", "dist_sma20", "dist_sma50",
        "bb_position", "volume_ratio"
    ]
    
    # Add macro indicators if they exist in df
    macro_cols = ["dxy", "us10y", "vix", "brent", "dji", "eurusd", "xagusd", "real_yield", "days_to_fed", "days_to_cpi", "days_to_nfp"]
    for col in macro_cols:
        if col in df.columns:
            FEATURE_COLS.append(col)
    
    MIN_ROWS = 15 + days_to_predict
    df_feat = df.copy()
    df_feat[FEATURE_COLS] = df_feat[FEATURE_COLS].ffill().bfill()  # ffill first (no future leak), bfill only for initial NaNs
    df_feat = df_feat.dropna(subset=FEATURE_COLS)
    
    if len(df_feat) < MIN_ROWS:
        preds, slope, r2 = predict_future_prices(df, days_to_predict)
        return preds, slope, r2, "linear_regression_fallback"
        
    current_price = df_feat["close"].iloc[-1]
    
    X_list = []
    y_list = []
    
    n = len(df_feat)
    for i in range(n - days_to_predict):
        row_features = df_feat.iloc[i][FEATURE_COLS].values
        targets = []
        valid = True
        for d in range(1, days_to_predict + 1):
            if i + d >= n:
                valid = False
                break
            future_close = df_feat["close"].iloc[i + d]
            current_close = df_feat["close"].iloc[i]
            ret = (future_close - current_close) / (current_close + 1e-10)
            targets.append(ret)
            
        if not valid:
            continue
        # Skip zero-change days (weekends/holidays)
        if abs(targets[0]) < 1e-5:
            continue
        if not any(np.isnan(v) for v in row_features) and not any(np.isnan(t) for t in targets):
            X_list.append(row_features)
            y_list.append(targets)
            
    if len(X_list) < 10:
        preds, slope, r2 = predict_future_prices(df, days_to_predict)
        return preds, slope, r2, "linear_regression_fallback"
        
    X_train = np.array(X_list, dtype=np.float64)
    y_train = np.array(y_list, dtype=np.float64)
    # Sanitize: replace NaN/Inf to prevent MLP crash on missing features
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    predicted_prices = []
    r2_scores = []
    
    for d_idx in range(days_to_predict):
        hidden_layer_sizes = params.get("hidden_layer_sizes", (32, 16)) if params else (32, 16)
        alpha = params.get("alpha", 1e-4) if params else 1e-4
        if isinstance(hidden_layer_sizes, list):
            hidden_layer_sizes = tuple(hidden_layer_sizes)
            
        mlp = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            activation='tanh',
            solver='lbfgs',
            max_iter=500,
            random_state=42
        )
        mlp.fit(X_train_scaled, y_train[:, d_idx])
        
        r2 = mlp.score(X_train_scaled, y_train[:, d_idx])
        r2_scores.append(r2)
        
        latest_features = np.nan_to_num(df_feat.iloc[-1][FEATURE_COLS].values.reshape(1, -1).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        latest_features_scaled = scaler.transform(latest_features)
        
        ret_pred = mlp.predict(latest_features_scaled)[0]
        predicted_price = current_price * (1 + ret_pred)
        predicted_prices.append(predicted_price)
        
    avg_r2 = float(np.mean(r2_scores))
    
    if len(predicted_prices) > 1:
        slope = (predicted_prices[-1] - current_price) / max(len(predicted_prices), 1)
    else:
        slope = predicted_prices[0] - current_price
        
    return np.array(predicted_prices), slope, avg_r2, "mlp_neural_network"


def predict_future_prices_xgb(df, days_to_predict=5, params=None):
    """
    Uses XGBoost Regressor trained on technical features
    to predict the next N days' closing prices via return forecasting.
    Falls back to Linear Regression if data is insufficient.
    """
    FEATURE_COLS = [
        "rsi", "rsi_slope",
        "macd", "macd_signal", "macd_hist", "macd_hist_slope",
        "return_1d", "return_2d", "return_3d", "return_5d",
        "momentum_10", "price_accel", "atr", "ema_ratio",
        "dist_sma5", "dist_sma20", "dist_sma50",
        "bb_position", "volume_ratio"
    ]
    
    # Add macro indicators if they exist in df
    macro_cols = ["dxy", "us10y", "vix", "brent", "dji", "eurusd", "xagusd", "real_yield", "days_to_fed", "days_to_cpi", "days_to_nfp"]
    for col in macro_cols:
        if col in df.columns:
            FEATURE_COLS.append(col)
    
    MIN_ROWS = 15 + days_to_predict
    df_feat = df.copy()
    df_feat[FEATURE_COLS] = df_feat[FEATURE_COLS].ffill().bfill()  # ffill first (no future leak), bfill only for initial NaNs
    df_feat = df_feat.dropna(subset=FEATURE_COLS)
    
    if len(df_feat) < MIN_ROWS:
        preds, slope, r2 = predict_future_prices(df, days_to_predict)
        return preds, slope, r2, "linear_regression_fallback"
        
    current_price = df_feat["close"].iloc[-1]
    
    X_list = []
    y_list = []
    
    n = len(df_feat)
    for i in range(n - days_to_predict):
        row_features = df_feat.iloc[i][FEATURE_COLS].values
        targets = []
        valid = True
        for d in range(1, days_to_predict + 1):
            if i + d >= n:
                valid = False
                break
            future_close = df_feat["close"].iloc[i + d]
            current_close = df_feat["close"].iloc[i]
            ret = (future_close - current_close) / (current_close + 1e-10)
            targets.append(ret)
            
        if not valid:
            continue
        # Skip zero-change days (weekends/holidays)
        if abs(targets[0]) < 1e-5:
            continue
        if not any(np.isnan(v) for v in row_features) and not any(np.isnan(t) for t in targets):
            X_list.append(row_features)
            y_list.append(targets)
            
    if len(X_list) < 10:
        preds, slope, r2 = predict_future_prices(df, days_to_predict)
        return preds, slope, r2, "linear_regression_fallback"
        
    X_train = np.array(X_list, dtype=np.float64)
    y_train = np.array(y_list, dtype=np.float64)
    # Sanitize: replace NaN/Inf to prevent XGBoost crash on missing features
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    try:
        import xgboost as xgb
        # Custom objective: Directional Huber Loss
        # Penalizes wrong-direction predictions 2× harder than wrong magnitude.
        # Gradient/hessian required by XGBoost custom objective API.
        def _directional_huber_obj(y_true, y_pred):
            delta = 1.0
            resid = y_pred - y_true
            # Huber base gradient
            abs_r = np.abs(resid)
            grad = np.where(abs_r <= delta, resid, delta * np.sign(resid))
            hess = np.where(abs_r <= delta, 1.0, delta / (abs_r + 1e-8))
            # Direction penalty: 2× for wrong direction sign
            direction_wrong = (np.sign(y_pred) != np.sign(y_true)).astype(float)
            penalty = 1.0 + direction_wrong  # 1.0 correct, 2.0 wrong direction
            return grad * penalty, hess * penalty


        predicted_prices = []
        r2_scores = []
        
        for d_idx in range(days_to_predict):
            max_depth = params.get("max_depth", 3) if params else 3
            learning_rate = params.get("learning_rate", 0.05) if params else 0.05
            model = xgb.XGBRegressor(
                n_estimators=150,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.7,
                colsample_bytree=0.7,
                min_child_weight=5,
                gamma=0.2,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=1,
                verbosity=0
            )
            model.set_params(objective=_directional_huber_obj)
            model.fit(X_train, y_train[:, d_idx])
            
            r2 = model.score(X_train, y_train[:, d_idx])
            r2_scores.append(r2)
            
            latest_features = np.nan_to_num(df_feat.iloc[-1][FEATURE_COLS].values.reshape(1, -1).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            ret_pred = model.predict(latest_features)[0]
            
            predicted_price = current_price * (1 + ret_pred)
            predicted_prices.append(predicted_price)
            
        avg_r2 = float(np.mean(r2_scores))
        
        if len(predicted_prices) > 1:
            slope = (predicted_prices[-1] - current_price) / max(len(predicted_prices), 1)
        else:
            slope = predicted_prices[0] - current_price
            
        return np.array(predicted_prices), slope, avg_r2, "xgboost"
    except Exception as e:
        print(f"XGBoost training/prediction failed: {e}. Falling back to Random Forest...")
        rf_result = predict_future_prices_rf(df, days_to_predict)
        preds, slope, r2 = rf_result[0], rf_result[1], rf_result[2]
        rf_model_used = rf_result[3]
        return preds, slope, r2, f"rf_fallback_for_xgb_{rf_model_used}"


def predict_future_prices_lstm(df, days_to_predict=5):
    """
    Uses a lightweight LSTM (sequence model) to predict future gold prices.
    Processes 20-day sliding windows to capture temporal dependencies — the key
    advantage over MLP which treats each day independently.

    Uses PyTorch if available, otherwise falls back to XGBoost.
    Falls back to XGBoost if data < 30 rows.
    """
    FEATURE_COLS = [
        "rsi", "rsi_slope",
        "macd", "macd_hist", "macd_hist_slope",
        "return_1d", "return_2d", "return_3d", "return_5d",
        "momentum_10", "price_accel", "atr", "ema_ratio",
        "dist_sma5", "dist_sma20",
        "bb_position", "volume_ratio"
    ]
    macro_cols = ["dxy", "us10y", "vix", "brent", "dji", "eurusd", "xagusd", "real_yield", "days_to_fed", "days_to_cpi", "days_to_nfp"]
    for col in macro_cols:
        if col in df.columns:
            FEATURE_COLS.append(col)

    SEQ_LEN = 20       # Look-back window in trading days
    MIN_ROWS = SEQ_LEN + days_to_predict + 5
    EPOCHS   = 80
    LR       = 3e-3

    df_feat = df.copy()
    df_feat[FEATURE_COLS] = df_feat[FEATURE_COLS].ffill().bfill()  # ffill first (no future leak), bfill only for initial NaNs
    df_feat = df_feat.dropna(subset=FEATURE_COLS)

    if len(df_feat) < MIN_ROWS:
        preds, slope, r2 = predict_future_prices(df, days_to_predict)
        return preds, slope, r2, "linear_regression_fallback"

    current_price = df_feat["close"].iloc[-1]

    try:
        import torch
        torch.set_num_threads(1)  # Prevent OpenMP deadlock in thread pools/executors
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler as _SS


        # Build sequences (X) and targets (y = cumulative returns relative to day i)
        scaler = _SS()
        X_all = scaler.fit_transform(df_feat[FEATURE_COLS].values.astype(float))

        X_seqs, y_seqs = [], []
        n = len(X_all)
        for i in range(SEQ_LEN, n - days_to_predict):
            current_close = df_feat["close"].iloc[i]
            # Calculate cumulative returns for days 1 to days_to_predict ahead
            targets = []
            for d in range(1, days_to_predict + 1):
                future_close = df_feat["close"].iloc[i + d]
                ret = (future_close - current_close) / (current_close + 1e-10)
                targets.append(ret)
                
            # Skip windows whose target is zero-change (weekend)
            if abs(targets[0]) < 1e-5:
                continue
            X_seqs.append(X_all[i - SEQ_LEN:i])
            y_seqs.append(targets)

        if len(X_seqs) < 10:
            raise ValueError("Not enough non-zero sequences for LSTM training")

        X_t = torch.tensor(np.array(X_seqs), dtype=torch.float32)   # (N, SEQ_LEN, F)
        y_t = torch.tensor(np.array(y_seqs), dtype=torch.float32)   # (N, days_to_predict)

        n_feat = X_t.shape[2]

        # Compact LSTM: 1 layer, hidden=32
        class GoldLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(n_feat, 32, num_layers=1,
                                    batch_first=True, dropout=0.0)
                self.fc   = nn.Sequential(
                    nn.Linear(32, 16),
                    nn.Tanh(),
                    nn.Linear(16, days_to_predict)
                )
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])  # last timestep

        model_lstm = GoldLSTM()
        optimizer  = torch.optim.Adam(model_lstm.parameters(), lr=LR, weight_decay=1e-4)
        loss_fn    = nn.MSELoss()

        model_lstm.train()
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            pred = model_lstm(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_lstm.parameters(), 1.0)
            optimizer.step()

        # Inference: use the last SEQ_LEN rows
        model_lstm.eval()
        with torch.no_grad():
            last_seq = torch.tensor(
                X_all[-SEQ_LEN:][np.newaxis, :, :], dtype=torch.float32
            )
            ret_preds = model_lstm(last_seq).numpy()[0]  # shape: (days_to_predict,)

        predicted_prices = [current_price * (1 + r) for r in ret_preds]

        # R² proxy: correlation on training set
        model_lstm.eval()
        with torch.no_grad():
            train_preds = model_lstm(X_t).numpy()
        ss_res = float(np.sum((y_t.numpy() - train_preds) ** 2))
        ss_tot = float(np.sum((y_t.numpy() - y_t.numpy().mean()) ** 2))
        avg_r2 = max(0.0, 1.0 - ss_res / (ss_tot + 1e-10))

        if len(predicted_prices) > 1:
            slope = (predicted_prices[-1] - current_price) / max(len(predicted_prices), 1)
        else:
            slope = predicted_prices[0] - current_price

        return np.array(predicted_prices), slope, avg_r2, "lstm"

    except Exception as e:
        print(f"LSTM failed ({e}), falling back to XGBoost...")
        return predict_future_prices_xgb(df, days_to_predict)


def calculate_actionable_advice(pred_prices, current_price, final_score, rsi_val, bb_pos, slope, conversion_factor=None, is_usd=False):
    t1_price = float(pred_prices[0])
    max_pred_price = float(max(pred_prices))
    
    # 1. Expected profit
    if max_pred_price > current_price:
        expected_profit_amount = max_pred_price - current_price
        expected_profit_pct = (expected_profit_amount / current_price) * 100
    else:
        expected_profit_amount = 0.0
        expected_profit_pct = 0.0

    # 2. Next price
    next_price = t1_price
    next_price_change_pct = ((t1_price - current_price) / current_price) * 100
    
    # 3. Stop loss
    if is_usd:
        stop_loss_price = current_price * 0.98  # 2% stop loss for global gold
    else:
        stop_loss_price = current_price * 0.93  # Default 7% stop loss in VN stock market

    # 4. Target buy range
    if t1_price < current_price:
        # Tomorrow is a dip, buy close to the dip
        buy_lower = t1_price * 0.99
        buy_upper = t1_price * 1.01
    else:
        # Tomorrow is up, buy close to current price
        buy_lower = current_price * 0.995
        buy_upper = current_price * 1.015

    if is_usd:
        target_buy_range_str = f"${buy_lower:,.2f} - ${buy_upper:,.2f}"
    else:
        target_buy_range_str = f"{int(round(buy_lower)):,}đ - {int(round(buy_upper)):,}đ"

    # 5. Buy now conclusion
    if is_usd:
        # USD gold trend threshold
        slope_threshold = 0.1
        if final_score >= 1.5:
            if rsi_val <= 35 or bb_pos <= 0.15:
                buy_now_conclusion = "Nên Mua Ngay"
                buy_now_subtext = "Giá vàng thế giới đang ở vùng chiết khấu sâu cực kỳ hấp dẫn, các chỉ báo kỹ thuật quá bán cho thấy khả năng đảo chiều phục hồi rất cao."
            else:
                buy_now_conclusion = "Có Thể Giải Ngân"
                buy_now_subtext = "Xu hướng tăng giá vàng đang hình thành, có thể chia nhỏ vốn mua gom dần hoặc giải ngân vị thế ban đầu."
        elif final_score <= -1.5:
            buy_now_conclusion = "Tuyệt Đối Không Mua"
            buy_now_subtext = "Áp lực bán mạnh và xu hướng giảm giá vàng đang chiếm ưu thế. Mua thời điểm này rủi ro chịu lỗ ngắn hạn lớn."
        else: # Theo dõi
            if slope > slope_threshold:
                buy_now_conclusion = "Chờ Nhịp Rung Lắc"
                buy_now_subtext = "Giá vàng có xu hướng phục hồi nhưng nên đợi nhịp điều chỉnh nhẹ trong phiên để có điểm mua an toàn."
            else:
                buy_now_conclusion = "Không Nên Mua Ngay"
                buy_now_subtext = "Động lực tăng giá vàng chưa rõ ràng. Nên kiên nhẫn quan sát thêm để xuất hiện điểm bứt phá."
    else:
        # VN Stock
        if final_score >= 1.5:
            if rsi_val <= 35 or bb_pos <= 0.15:
                buy_now_conclusion = "Nên Mua Ngay"
                buy_now_subtext = "Vùng giá chiết khấu sâu cực kỳ hấp dẫn, các chỉ báo kỹ thuật quá bán cho thấy khả năng đảo chiều phục hồi rất cao."
            else:
                buy_now_conclusion = "Có Thể Giải Ngân"
                buy_now_subtext = "Xu hướng tăng đang hình thành, có thể chia nhỏ vốn mua gom dần hoặc giải ngân vị thế ban đầu."
        elif final_score <= -1.5:
            buy_now_conclusion = "Tuyệt Đối Không Mua"
            buy_now_subtext = "Áp lực bán mạnh và xu hướng giảm đang chiếm ưu thế. Mua thời điểm này rủi ro chịu lỗ ngắn hạn lớn."
        else: # Theo dõi
            if slope > 0.01:
                buy_now_conclusion = "Chờ Nhịp Rung Lắc"
                buy_now_subtext = "Cổ phiếu có xu hướng phục hồi nhưng nên đợi giá lùi nhẹ trong phiên để có điểm mua an toàn."
            else:
                buy_now_conclusion = "Không Nên Mua Ngay"
                buy_now_subtext = "Dòng tiền và động thái tích lũy chưa rõ ràng. Nên kiên nhẫn quan sát thêm để xuất hiện điểm bứt phá."

    # 6. Best timing to buy
    if is_usd:
        # Gold specific timing
        slope_threshold = 0.1
        if final_score <= -1.5:
            best_buy_time = "Chờ tạo đáy"
            best_buy_time_reason = "Mô hình dự báo xu hướng giảm tiếp diễn. Chưa có tín hiệu dừng rơi rõ ràng, nên đứng ngoài quan sát."
        elif pred_prices[0] > current_price and slope > slope_threshold:
            best_buy_time = "Phiên Á (9:00 - 11:30)"
            best_buy_time_reason = "Dự kiến xu hướng tăng giá vàng tiếp diễn. Nên giải ngân vào phiên Á khi thị trường biến động chậm để có mức giá tốt nhất trước khi phiên Âu-Mỹ mở cửa gây biến động mạnh."
        elif pred_prices[0] < current_price:
            best_buy_time = "Phiên Mỹ (20:00 - 22:00)"
            best_buy_time_reason = "Dự báo giá vàng thế giới có nhịp điều chỉnh kỹ thuật ngắn hạn. Nên canh giải ngân vào đầu phiên Mỹ sau các tin tức kinh tế quan trọng để đón nhịp hồi phục sau khi giá test xong các mốc hỗ trợ."
        else:
            best_buy_time = "Khung giờ trưa (11:30 - 13:30)"
            best_buy_time_reason = "Xu hướng chủ đạo là đi ngang tích lũy. Nên giải ngân vào khung giờ trưa (giao thoa giữa phiên Á và Âu) khi thanh khoản thấp và biến động giá hẹp để giảm thiểu rủi ro biến động đột ngột."
    else:
        # VN Stock timing
        if final_score <= -1.5:
            best_buy_time = "Chờ tạo đáy"
            best_buy_time_reason = "Mô hình dự báo xu hướng giảm tiếp diễn. Chưa có tín hiệu dừng rơi rõ ràng, nên đứng ngoài quan sát."
        elif pred_prices[0] < current_price and max(pred_prices[1:]) > current_price:
            best_buy_time = "Chiều ngày mai (13:30 - 14:15)"
            best_buy_time_reason = "Dự báo phiên mai có nhịp giảm nhẹ. Khung giờ 13:30 - 14:15 là lúc hàng T+2.5 về tài khoản tạo áp lực bán lớn nhất, thích hợp để canh giá đỏ tối ưu trước khi bật tăng."
        elif pred_prices[1] < current_price and max(pred_prices[2:]) > current_price:
            best_buy_time = "Chiều ngày mốt (Sau 13:30)"
            best_buy_time_reason = "Nhịp điều chỉnh dự kiến kéo dài sang phiên ngày mốt trước khi tạo đáy đi lên. Mua vào phiên chiều ngày mốt sẽ có giá vốn an toàn."
        elif pred_prices[0] > current_price and slope > 0.02:
            best_buy_time = "Sáng ngày mai (9:15 - 10:30)"
            best_buy_time_reason = "Dự kiến giá sẽ bứt phá ngay đầu phiên mai. Canh mua sớm trong phiên sáng để có vị thế tốt trước khi dòng tiền kéo tăng mạnh."
        else:
            best_buy_time = "Chiều ngày mai (Sau 14:00)"
            best_buy_time_reason = "Xu hướng chủ đạo là đi ngang tích lũy. Nên giải ngân vào cuối phiên chiều sau 14:00 để kiểm chứng giá đóng cửa của ngày nhằm giảm thiểu rủi ro biến động đột ngột."

    result = {
        "buy_now_conclusion": buy_now_conclusion,
        "buy_now_subtext": buy_now_subtext,
        "best_buy_time": best_buy_time,
        "best_buy_time_reason": best_buy_time_reason,
        "target_buy_range": target_buy_range_str,
        "expected_profit_pct": float(round(expected_profit_pct, 2)),
        "expected_profit_amount": float(round(expected_profit_amount, 2)),
        "next_price": float(round(next_price, 2)),
        "next_price_change_pct": float(round(next_price_change_pct, 2)),
        "stop_loss_price": float(round(stop_loss_price, 2))
    }

    if conversion_factor and conversion_factor > 0:
        result["next_price_usd"] = float(round(next_price * conversion_factor, 2))
        result["target_buy_range_usd"] = f"{buy_lower * conversion_factor:,.2f}$ - {buy_upper * conversion_factor:,.2f}$"
        result["expected_profit_amount_usd"] = float(round(expected_profit_amount * conversion_factor, 2))
        result["stop_loss_price_usd"] = float(round(stop_loss_price * conversion_factor, 2))

    return result


def _build_model_metrics(pred_prices, slope, r2, current_price, base_score, reasons, model_name, rsi_val, bb_pos, conversion_factor=None, is_usd=False):
    """
    Builds the model-specific result dictionary.
    Adjusts score contribution from ML trend using the given slope and r2.
    """
    # Remove the old LR-based ML score contribution and replace with model-specific
    predicted_change = pred_prices[-1] - current_price
    predicted_pct_change = (predicted_change / current_price) * 100
    
    # ML score contribution (same logic as original but re-applied for this model)
    ml_weight = min(1.5, abs(slope) / (current_price + 1e-10) * 100 * r2 * 2.0)
    ml_score_contribution = max(0.25, min(1.5, ml_weight)) * (1 if slope > 0 else -1)
    
    # Final score = base (tech indicators) + ml contribution
    final_score = max(-5.0, min(5.0, base_score + ml_score_contribution))
    
    # Determine recommendation
    if final_score >= 1.5:
        recommendation = "Nên Mua"
        action_class = "buy"
    elif final_score <= -1.5:
        recommendation = "Nên Bán / Tránh Mua"
        action_class = "sell"
    else:
        recommendation = "Theo Dõi"
        action_class = "hold"
    
    if slope > 0.05:
        direction = "Tăng"
        direction_class = "up"
    elif slope < -0.05:
        direction = "Giảm"
        direction_class = "down"
    else:
        direction = "Đi ngang"
        direction_class = "sideways"
    
    model_label = {
        "random_forest": "Random Forest (Học máy nâng cao)",
        "linear_regression": "Hồi quy tuyến tính (Linear Regression)",
        "ensemble": "Ensemble Hybrid (Tổ hợp 4 mô hình)",
        "mlp_neural_network": "Mạng Nơ-ron MLP (Deep Learning)",
        "xgboost": "XGBoost Regressor (Học máy nâng cao)",
        "linear_regression_fallback": "Hồi quy tuyến tính (Fallback)",
        "rf_fallback_for_xgb_random_forest": "XGBoost (Lỗi hệ thống - Tự động chuyển RF)",
        "rf_fallback_for_xgb_linear_regression_fallback": "XGBoost (Lỗi hệ thống - Tự động chuyển LR)"
    }.get(model_name, model_name)
    
    advice = calculate_actionable_advice(pred_prices, current_price, final_score, rsi_val, bb_pos, slope, conversion_factor=conversion_factor, is_usd=is_usd)
    
    ml_prediction = {
        "predicted_prices": [float(p) for p in pred_prices],
        "slope": float(slope),
        "r_squared": float(r2),
        "predicted_change": float(predicted_change),
        "predicted_pct_change": float(predicted_pct_change),
        "model_name": model_label,
    }
    if conversion_factor and conversion_factor > 0:
        ml_prediction["predicted_prices_usd"] = [float(round(p * conversion_factor, 2)) for p in pred_prices]

    return {
        "score": round(final_score, 2),
        "recommendation": recommendation,
        "action_class": action_class,
        "direction": direction,
        "direction_class": direction_class,
        "actionable_advice": advice,
        "ml_prediction": ml_prediction
    }


def analyze_and_recommend(df, conversion_factor=None, is_usd=False, model_weights=None, params=None, biases=None, conflict_events=None):
    """
    Analyzes technical indicators and machine learning trend to output
    a buying recommendation, price direction prediction, and score.
    Now returns predictions from 3 models: Linear Regression, Random Forest, Ensemble.
    """
    if model_weights is None:
        model_weights = {
            "random_forest": 0.20,
            "linear_regression": 0.20,
            "mlp": 0.20,
            "xgboost": 0.20,
            "lstm": 0.20
        }
    
    # Extract weights
    w_rf   = model_weights.get("random_forest", 0.20)
    w_lr   = model_weights.get("linear_regression", 0.20)
    w_mlp  = model_weights.get("mlp", 0.20)
    w_xgb  = model_weights.get("xgboost", 0.20)
    w_lstm = model_weights.get("lstm", 0.20)
    
    # Normalize weights to sum to 1.0
    total_w = w_rf + w_lr + w_mlp + w_xgb + w_lstm
    if total_w > 0:
        w_rf   /= total_w
        w_lr   /= total_w
        w_mlp  /= total_w
        w_xgb  /= total_w
        w_lstm /= total_w
    else:
        w_rf = w_lr = w_mlp = w_xgb = w_lstm = 0.20

    if len(df) < 20:
        return {
            "status": "error",
            "message": "Không đủ dữ liệu lịch sử để thực hiện dự đoán (Yêu cầu tối thiểu 20 phiên)."
        }
        
    # Calculate indicators
    df_ind = calculate_technical_indicators(df)
    
    # Get latest values (today)
    latest = df_ind.iloc[-1]
    prev = df_ind.iloc[-2] if len(df_ind) > 1 else latest
    
    current_price = latest["close"]

    # ─────────────────────────────────────────────────────────────────
    # STEP 1: Calculate technical-indicator base score (shared across models)
    # Score: 0 to ±5.0
    # ─────────────────────────────────────────────────────────────────
    tech_score = 0.0
    reasons = []
    
    # 1. SMA Trend Analysis (Max weight: 1.5)
    if pd.notna(latest["sma_5"]) and pd.notna(latest["sma_20"]):
        if latest["sma_5"] > latest["sma_20"]:
            if prev["sma_5"] <= prev["sma_20"]:
                tech_score += 1.5
                reasons.append("SMA5 vừa cắt lên trên SMA20 (Tín hiệu Giao cắt Vàng - Golden Cross mạnh mẽ)")
            else:
                tech_score += 0.5
                reasons.append("Đường SMA5 nằm trên SMA20 (Xu hướng tăng giá ngắn hạn)")
        else:
            if prev["sma_5"] >= prev["sma_20"]:
                tech_score -= 1.5
                reasons.append("SMA5 vừa cắt xuống dưới SMA20 (Tín hiệu Giao cắt Tử thần - Death Cross xấu)")
            else:
                tech_score -= 0.5
                reasons.append("Đường SMA5 nằm dưới SMA20 (Xu hướng giảm giá ngắn hạn)")
                
    # 2. RSI Analysis (Max weight: 1.5)
    rsi_val = latest["rsi"]
    if pd.notna(rsi_val):
        if rsi_val <= 30:
            tech_score += 1.5
            reasons.append(f"RSI đạt {rsi_val:.1f} (Vùng Quá Bán - Oversold, cơ hội hồi phục kỹ thuật cực tốt)")
        elif rsi_val >= 70:
            tech_score -= 1.5
            reasons.append(f"RSI đạt {rsi_val:.1f} (Vùng Quá Mua - Overbought, rủi ro điều chỉnh cao)")
        else:
            rsi_diff = rsi_val - prev["rsi"]
            if rsi_diff > 3:
                tech_score += 0.25
                reasons.append(f"RSI ({rsi_val:.1f}) đang hướng lên, cho thấy động lượng hồi phục")
            elif rsi_diff < -3:
                tech_score -= 0.25
                reasons.append(f"RSI ({rsi_val:.1f}) đang hướng xuống, áp lực bán gia tăng")
                
    # 3. MACD Analysis (Max weight: 1.5)
    macd_val = latest["macd"]
    sig_val = latest["macd_signal"]
    if pd.notna(macd_val) and pd.notna(sig_val):
        if macd_val > sig_val:
            if prev["macd"] <= prev["macd_signal"]:
                tech_score += 1.5
                reasons.append("MACD cắt lên đường Tín hiệu (Tín hiệu MUA mạnh)")
            else:
                tech_score += 0.5
                reasons.append("Đường MACD nằm trên đường Tín hiệu (Động lượng tăng tích cực)")
        else:
            if prev["macd"] >= prev["macd_signal"]:
                tech_score -= 1.5
                reasons.append("MACD cắt xuống dưới đường Tín hiệu (Tín hiệu BÁN mạnh)")
            else:
                tech_score -= 0.5
                reasons.append("Đường MACD nằm dưới đường Tín hiệu (Áp lực điều chỉnh còn duy trì)")
                
    # 4. Bollinger Bands Analysis (Max weight: 1.0)
    if pd.notna(latest["bb_lower"]) and pd.notna(latest["bb_upper"]):
        if current_price <= latest["bb_lower"]:
            tech_score += 1.0
            reasons.append("Giá chạm dải dưới Bollinger Band (Có xu hướng đảo chiều bật tăng)")
        elif current_price >= latest["bb_upper"]:
            tech_score -= 1.0
            reasons.append("Giá chạm dải trên Bollinger Band (Áp lực chốt lời gia tăng)")

    # 5. Geopolitical Conflict Analysis (Max weight: 1.2)
    conflict_score = 0.0
    conflict_reasons = []
    if conflict_events:
        for event in conflict_events:
            title = event.get("title", "").lower()
            desc = event.get("description", "").lower()
            status = event.get("status", "")
            parties = event.get("parties", "")
            
            has_us = any(w in title or w in desc for w in ["mỹ", "washington", "nhà trắng", "pentagon", "usa", "us"])
            has_iran = any(w in title or w in desc for w in ["iran", "tehran"])
            has_israel = any(w in title or w in desc for w in ["israel", "gaza", "hamas"])
            has_russia_ukraine = any(w in title or w in desc for w in ["nga", "ukraine", "kiev", "moscow", "russia"])
            
            if status == "Giao tranh quân sự" or "tấn công" in title or "tên lửa" in title or "không kích" in title:
                if has_us and has_iran:
                    weight = -0.4
                    reason = f"Giao tranh quân sự Mỹ - Iran: Kéo theo giá dầu tăng vọt, gián tiếp gây sức ép làm giảm giá vàng (-0.40 điểm)."
                elif has_israel and has_iran:
                    weight = 0.4
                    reason = f"Giao tranh quân sự trực tiếp/căng thẳng cao giữa {parties}: Kích hoạt dòng tiền trú ẩn an toàn cực mạnh vào vàng (+{weight} điểm)."
                elif has_israel or has_us:
                    weight = 0.3
                    reason = f"Giao tranh quân sự/không kích tại {parties}: Rủi ro địa chính trị leo thang hỗ trợ đà tăng giá vàng (+{weight} điểm)."
                elif has_russia_ukraine:
                    weight = 0.2
                    reason = f"Giao tranh tại Ukraine/Nga: Căng thẳng khu vực duy trì vị thế trú ẩn của vàng (+{weight} điểm)."
                else:
                    weight = 0.15
                    reason = f"Xung đột quân sự quốc tế ({parties}): Tăng nhẹ nhu cầu phòng thủ tài sản (+{weight} điểm)."
                conflict_score += weight
                conflict_reasons.append(reason)
            elif status == "Leo thang căng thẳng" or "cảnh báo" in title or "leo thang" in title or "đe dọa" in title:
                if has_us and has_iran:
                    weight = -0.25
                    reason = f"Căng thẳng ngoại giao/quân sự Mỹ - Iran leo thang: Làm tăng giá dầu, gây áp lực giảm lên giá vàng (-0.25 điểm)."
                elif has_israel and has_iran:
                    weight = 0.25
                    reason = f"Leo thang căng thẳng ngoại giao/quân sự {parties}: Rủi ro leo thang xung đột thúc đẩy nhu cầu bảo toàn vốn (+{weight} điểm)."
                else:
                    weight = 0.15
                    reason = f"Căng thẳng leo thang tại {parties}: Tạo tâm lý thận trọng trên thị trường vàng (+{weight} điểm)."
                conflict_score += weight
                conflict_reasons.append(reason)
            elif status == "Đàm phán hòa bình" or "ngừng bắn" in title or "đàm phán" in title or "thỏa thuận" in title:
                weight = 0.25
                reason = f"Tiến trình đàm phán hòa bình/thỏa thuận tại {parties}: Giảm bớt nhu cầu trú ẩn an toàn vào vàng (-{weight} điểm)."
                conflict_score -= weight
                conflict_reasons.append(reason)
            elif status == "Trừng phạt ngoại giao" or "cấm vận" in title or "trừng phạt" in title:
                weight = 0.1
                reason = f"Lệnh trừng phạt/cấm vận liên quan {parties}: Tăng nhẹ rủi ro đứt gãy chuỗi cung ứng/kinh tế vĩ mô (+{weight} điểm)."
                conflict_score += weight
                conflict_reasons.append(reason)

        conflict_score = max(-1.2, min(1.2, conflict_score))
        if conflict_score != 0:
            tech_score += conflict_score
            reasons.extend(conflict_reasons)

    # ─────────────────────────────────────────────────────────────────
    # STEP 2: Get predictions from all 3 models
    # ─────────────────────────────────────────────────────────────────
    days_to_predict = 5
    
    # Model A: Linear Regression
    lr_prices, lr_slope, lr_r2 = predict_future_prices(df_ind, days_to_predict=days_to_predict)
    
    # Model B: Random Forest
    rf_result = predict_future_prices_rf(df_ind, days_to_predict=days_to_predict, params=params.get("random_forest") if params else None)
    rf_prices, rf_slope, rf_r2 = rf_result[0], rf_result[1], rf_result[2]
    rf_model_used = rf_result[3]
    
    # Model C: MLP Neural Network
    mlp_result = predict_future_prices_mlp(df_ind, days_to_predict=days_to_predict, params=params.get("mlp") if params else None)
    mlp_prices, mlp_slope, mlp_r2 = mlp_result[0], mlp_result[1], mlp_result[2]
    mlp_model_used = mlp_result[3]
    
    # Model D: XGBoost Regressor
    xgb_result = predict_future_prices_xgb(df_ind, days_to_predict=days_to_predict, params=params.get("xgboost") if params else None)
    xgb_prices, xgb_slope, xgb_r2 = xgb_result[0], xgb_result[1], xgb_result[2]
    xgb_model_used = xgb_result[3]

    # Model E: LSTM Sequence Model
    lstm_result = predict_future_prices_lstm(df_ind, days_to_predict=days_to_predict)
    lstm_prices, lstm_slope, lstm_r2 = lstm_result[0], lstm_result[1], lstm_result[2]
    lstm_model_used = lstm_result[3]
    
    # Apply rolling bias correction if available
    if biases:
        rf_prices = rf_prices + biases.get("random_forest", 0.0)
        lr_prices = lr_prices + biases.get("linear_regression", 0.0)
        mlp_prices = mlp_prices + biases.get("mlp", 0.0)
        xgb_prices = xgb_prices + biases.get("xgboost", 0.0)
        lstm_prices = lstm_prices + biases.get("lstm", 0.0)
        
        # Re-calculate slopes relative to current price
        rf_slope = (rf_prices[-1] - current_price) / max(len(rf_prices), 1)
        lr_slope = (lr_prices[-1] - current_price) / max(len(lr_prices), 1)
        mlp_slope = (mlp_prices[-1] - current_price) / max(len(mlp_prices), 1)
        xgb_slope = (xgb_prices[-1] - current_price) / max(len(xgb_prices), 1)
        lstm_slope = (lstm_prices[-1] - current_price) / max(len(lstm_prices), 1)
    
    # Model F: Ensemble Hybrid (Weighted combination of 5 models)
    ens_prices = (rf_prices * w_rf + lr_prices * w_lr + mlp_prices * w_mlp
                  + xgb_prices * w_xgb + lstm_prices * w_lstm)
    ens_slope  = (rf_slope * w_rf + lr_slope * w_lr + mlp_slope * w_mlp
                  + xgb_slope * w_xgb + lstm_slope * w_lstm)
    ens_r2     = (rf_r2 * w_rf + lr_r2 * w_lr + mlp_r2 * w_mlp
                  + xgb_r2 * w_xgb + lstm_r2 * w_lstm)

    # ─────────────────────────────────────────────────────────────────
    # STEP 3: Build per-model result dictionaries
    # ─────────────────────────────────────────────────────────────────
    bb_pos_val = float(latest["bb_position"]) if pd.notna(latest.get("bb_position")) else 0.5
    rsi_val_float = float(rsi_val) if pd.notna(rsi_val) else 50.0
    
    lr_metrics   = _build_model_metrics(lr_prices,   lr_slope,   lr_r2,   current_price, tech_score, reasons, "linear_regression", rsi_val_float, bb_pos_val, conversion_factor=conversion_factor, is_usd=is_usd)
    rf_metrics   = _build_model_metrics(rf_prices,   rf_slope,   rf_r2,   current_price, tech_score, reasons, rf_model_used,        rsi_val_float, bb_pos_val, conversion_factor=conversion_factor, is_usd=is_usd)
    mlp_metrics  = _build_model_metrics(mlp_prices,  mlp_slope,  mlp_r2,  current_price, tech_score, reasons, mlp_model_used,       rsi_val_float, bb_pos_val, conversion_factor=conversion_factor, is_usd=is_usd)
    xgb_metrics  = _build_model_metrics(xgb_prices,  xgb_slope,  xgb_r2,  current_price, tech_score, reasons, xgb_model_used,       rsi_val_float, bb_pos_val, conversion_factor=conversion_factor, is_usd=is_usd)
    lstm_metrics = _build_model_metrics(lstm_prices, lstm_slope, lstm_r2, current_price, tech_score, reasons, lstm_model_used,      rsi_val_float, bb_pos_val, conversion_factor=conversion_factor, is_usd=is_usd)
    ens_metrics  = _build_model_metrics(ens_prices,  ens_slope,  ens_r2,  current_price, tech_score, reasons, "ensemble",           rsi_val_float, bb_pos_val, conversion_factor=conversion_factor, is_usd=is_usd)
    
    # Append model-specific ML reason to each model's reasons
    lr_reason   = f"Mô hình Hồi Quy Tuyến Tính dự báo xu hướng {'Tăng' if lr_slope > 0 else 'Giảm'} trong 5 phiên tới (R²: {lr_r2:.2f})"
    rf_reason   = f"Mô hình {'Random Forest' if rf_model_used == 'random_forest' else 'Linear Regression (Fallback)'} dự báo xu hướng {'Tăng' if rf_slope > 0 else 'Giảm'} trong 5 phiên tới (R²: {rf_r2:.2f})"
    mlp_reason  = f"Mô hình {'Mạng Nơ-ron MLP' if mlp_model_used == 'mlp_neural_network' else 'Linear Regression (Fallback)'} dự báo xu hướng {'Tăng' if mlp_slope > 0 else 'Giảm'} trong 5 phiên tới (R²: {mlp_r2:.2f})"
    xgb_reason  = f"Mô hình {'XGBoost Regressor' if xgb_model_used == 'xgboost' else 'Linear Regression (Fallback)'} dự báo xu hướng {'Tăng' if xgb_slope > 0 else 'Giảm'} trong 5 phiên tới (R²: {xgb_r2:.2f})"
    lstm_reason = f"Mô hình LSTM (Chuỗi Thời Gian) dự báo xu hướng {'Tăng' if lstm_slope > 0 else 'Giảm'} trong 5 phiên tới (R²: {lstm_r2:.2f})"
    ens_reason  = f"Mô hình Ensemble Hybrid (Tổ hợp 5 mô hình) dự báo xu hướng {'Tăng' if ens_slope > 0 else 'Giảm'} trong 5 phiên tới (R²: {ens_r2:.2f})"

    # ─────────────────────────────────────────────────────────────────
    # STEP 4: Use Random Forest as the default top-level response
    # ─────────────────────────────────────────────────────────────────
    default = rf_metrics
    default_reasons = reasons + [rf_reason]
    
    analysis = {
        "status": "success",
        "ticker": latest.get("code", df.iloc[0].get("code", "UNKNOWN")),
        "current_price": float(current_price),
        "yesterday_close": float(prev["close"]),
        "change": float(latest.get("change", current_price - prev["close"])),
        "pct_change": float(latest.get("pct_change", ((current_price - prev["close"]) / prev["close"]) * 100)),
        "volume": float(latest["volume"]),
        "indicators": {
            "sma_5": float(latest["sma_5"]) if pd.notna(latest["sma_5"]) else None,
            "sma_20": float(latest["sma_20"]) if pd.notna(latest["sma_20"]) else None,
            "sma_50": float(latest["sma_50"]) if pd.notna(latest["sma_50"]) else None,
            "rsi": float(rsi_val) if pd.notna(rsi_val) else None,
            "macd": float(macd_val) if pd.notna(macd_val) else None,
            "macd_signal": float(sig_val) if pd.notna(sig_val) else None,
            "bb_upper": float(latest["bb_upper"]) if pd.notna(latest["bb_upper"]) else None,
            "bb_lower": float(latest["bb_lower"]) if pd.notna(latest["bb_lower"]) else None,
        },
        # Top-level defaults to Random Forest
        "ml_prediction": default["ml_prediction"],
        "score": default["score"],
        "recommendation": default["recommendation"],
        "action_class": default["action_class"],
        "direction": default["direction"],
        "direction_class": default["direction_class"],
        "actionable_advice": default["actionable_advice"],
        "reasons": default_reasons,
        # All model results for frontend switching
        "models": {
            "random_forest": {
                **rf_metrics,
                "weight": w_rf,
                "reasons": reasons + [rf_reason]
            },
            "linear_regression": {
                **lr_metrics,
                "weight": w_lr,
                "reasons": reasons + [lr_reason]
            },
            "mlp": {
                **mlp_metrics,
                "weight": w_mlp,
                "reasons": reasons + [mlp_reason]
            },
            "xgboost": {
                **xgb_metrics,
                "weight": w_xgb,
                "reasons": reasons + [xgb_reason]
            },
            "lstm": {
                **lstm_metrics,
                "weight": w_lstm,
                "reasons": reasons + [lstm_reason]
            },
            "ensemble": {
                **ens_metrics,
                "weight": 1.0,
                "reasons": reasons + [ens_reason]
            }
        },
        # Send historical data for plotting in frontend (limit to last 60 days to keep JSON small)
        "history": df_ind.tail(60)[["date", "open", "high", "low", "close", "volume", "rsi", "macd", "macd_signal"]].to_dict(orient="records"),
        "backtest": None
    }
    
    # Calculate backtest (previous prediction vs actual last row)
    if len(df) >= 21:
        try:
            df_prev = df.iloc[:-1].copy()
            df_prev_ind = calculate_technical_indicators(df_prev)
            
            # Predict T+1 from df_prev_ind
            lr_p_back, _, _ = predict_future_prices(df_prev_ind, days_to_predict=1)
            
            rf_res_back = predict_future_prices_rf(df_prev_ind, days_to_predict=1)
            rf_p_back = rf_res_back[0]
            
            mlp_res_back = predict_future_prices_mlp(df_prev_ind, days_to_predict=1)
            mlp_p_back = mlp_res_back[0]
            
            xgb_res_back = predict_future_prices_xgb(df_prev_ind, days_to_predict=1)
            xgb_p_back = xgb_res_back[0]
            
            ens_p_back = rf_p_back[0] * w_rf + lr_p_back[0] * w_lr + mlp_p_back[0] * w_mlp + xgb_p_back[0] * w_xgb
            
            actual_last = df.iloc[-1]
            actual_close = float(actual_last["close"])
            actual_date = actual_last["date"]
            
            rf_pred = float(rf_p_back[0])
            lr_pred = float(lr_p_back[0])
            mlp_pred = float(mlp_p_back[0])
            xgb_pred = float(xgb_p_back[0])
            ens_pred = float(ens_p_back)
            
            analysis["backtest"] = {
                "date": actual_date,
                "actual_price": actual_close,
                "models": {
                    "random_forest": {
                        "predicted_price": rf_pred,
                        "error_pct": ((actual_close - rf_pred) / (rf_pred + 1e-10)) * 100
                    },
                    "linear_regression": {
                        "predicted_price": lr_pred,
                        "error_pct": ((actual_close - lr_pred) / (lr_pred + 1e-10)) * 100
                    },
                    "mlp": {
                        "predicted_price": mlp_pred,
                        "error_pct": ((actual_close - mlp_pred) / (mlp_pred + 1e-10)) * 100
                    },
                    "xgboost": {
                        "predicted_price": xgb_pred,
                        "error_pct": ((actual_close - xgb_pred) / (xgb_pred + 1e-10)) * 100
                    },
                    "ensemble": {
                        "predicted_price": ens_pred,
                        "error_pct": ((actual_close - ens_pred) / (ens_pred + 1e-10)) * 100
                    }
                }
            }
            if conversion_factor and conversion_factor > 0:
                analysis["backtest"]["actual_price_usd"] = float(round(actual_close * conversion_factor, 2))
                for mKey in analysis["backtest"]["models"]:
                    p_val = analysis["backtest"]["models"][mKey]["predicted_price"]
                    analysis["backtest"]["models"][mKey]["predicted_price_usd"] = float(round(p_val * conversion_factor, 2))
        except Exception as e:
            print(f"Error calculating backtest: {e}")
            
    import math
    def sanitize_floats(obj):
        if isinstance(obj, dict):
            return {k: sanitize_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_floats(x) for x in obj]
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        return obj
    
    return sanitize_floats(analysis)


def optimize_hyperparameters(df, days_to_predict=1):
    """
    Performs grid search hyperparameter tuning for Random Forest, XGBoost, and MLP
    using TimeSeriesSplit cross-validation on the last 180 days of gold data.
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    
    FEATURE_COLS = [
        "rsi", "rsi_slope",
        "macd", "macd_signal", "macd_hist", "macd_hist_slope",
        "return_1d", "return_2d", "return_3d", "return_5d",
        "momentum_10", "price_accel", "atr", "ema_ratio",
        "dist_sma5", "dist_sma20", "dist_sma50",
        "bb_position", "volume_ratio"
    ]
    
    # Add macro indicators if they exist in df
    macro_cols = ["dxy", "us10y", "vix", "brent", "dji", "spx", "eurusd", "xagusd", "real_yield", "days_to_fed", "days_to_cpi", "days_to_nfp"]
    for col in macro_cols:
        if col in df.columns:
            FEATURE_COLS.append(col)
            
    df_feat = df.copy()
    df_feat[FEATURE_COLS] = df_feat[FEATURE_COLS].ffill().bfill()
    df_feat = df_feat.dropna(subset=FEATURE_COLS)
    
    # Train on the last 180 days to stay fresh
    df_feat = df_feat.tail(180)
    if len(df_feat) < 30:
        return {} # insufficient data for tuning
        
    X_list = []
    y_list = []
    n = len(df_feat)
    for i in range(n - days_to_predict):
        row_features = df_feat.iloc[i][FEATURE_COLS].values
        targets = []
        valid = True
        for d in range(1, days_to_predict + 1):
            if i + d >= n:
                valid = False
                break
            future_close = df_feat["close"].iloc[i + d]
            current_close = df_feat["close"].iloc[i]
            ret = (future_close - current_close) / (current_close + 1e-10)
            targets.append(ret)
        if not valid:
            continue
        if abs(targets[0]) < 1e-5:
            continue
        if not any(np.isnan(v) for v in row_features) and not any(np.isnan(t) for t in targets):
            X_list.append(row_features)
            y_list.append(targets)
            
    if len(X_list) < 20:
        return {}
        
    X_train = np.array(X_list, dtype=np.float64)
    y_train = np.array(y_list, dtype=np.float64)[:, 0] # tune for 1-day ahead
    
    # Sanitize
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    tscv = TimeSeriesSplit(n_splits=3)
    best_params = {}
    
    # 1. Tune Random Forest
    rf_grid = {
        "max_depth": [3, 5, 7],
        "min_samples_leaf": [2, 4, 6]
    }
    best_rf_score = -float('inf')
    best_rf_params = {"max_depth": 5, "min_samples_leaf": 4}
    
    for depth in rf_grid["max_depth"]:
        for leaf in rf_grid["min_samples_leaf"]:
            scores = []
            for train_idx, test_idx in tscv.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[test_idx]
                y_tr, y_val = y_train[train_idx], y_train[test_idx]
                model = RandomForestRegressor(n_estimators=100, max_depth=depth, min_samples_leaf=leaf, max_features="sqrt", random_state=42, n_jobs=1)
                model.fit(X_tr, y_tr)
                scores.append(model.score(X_val, y_val))
            mean_score = np.mean(scores)
            if mean_score > best_rf_score:
                best_rf_score = mean_score
                best_rf_params = {"max_depth": int(depth), "min_samples_leaf": int(leaf)}
                
    best_params["random_forest"] = best_rf_params
    
    # 2. Tune XGBoost
    xgb_grid = {
        "max_depth": [2, 3, 5],
        "learning_rate": [0.03, 0.05, 0.1]
    }
    best_xgb_score = -float('inf')
    best_xgb_params = {"max_depth": 3, "learning_rate": 0.05}
    
    for depth in xgb_grid["max_depth"]:
        for lr in xgb_grid["learning_rate"]:
            scores = []
            for train_idx, test_idx in tscv.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[test_idx]
                y_tr, y_val = y_train[train_idx], y_train[test_idx]
                model = xgb.XGBRegressor(n_estimators=100, max_depth=depth, learning_rate=lr, subsample=0.7, colsample_bytree=0.7, random_state=42, n_jobs=1, verbosity=0)
                model.fit(X_tr, y_tr)
                scores.append(model.score(X_val, y_val))
            mean_score = np.mean(scores)
            if mean_score > best_xgb_score:
                best_xgb_score = mean_score
                best_xgb_params = {"max_depth": int(depth), "learning_rate": float(lr)}
                
    best_params["xgboost"] = best_xgb_params
    
    # 3. Tune MLP
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    mlp_grid = {
        "hidden_layer_sizes": [(16, 8), (32, 16)],
        "alpha": [1e-4, 1e-3, 1e-2]
    }
    best_mlp_score = -float('inf')
    best_mlp_params = {"hidden_layer_sizes": (32, 16), "alpha": 1e-4}
    
    for size in mlp_grid["hidden_layer_sizes"]:
        for alpha_val in mlp_grid["alpha"]:
            scores = []
            for train_idx, test_idx in tscv.split(X_train_scaled):
                X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[test_idx]
                y_tr, y_val = y_train[train_idx], y_train[test_idx]
                model = MLPRegressor(hidden_layer_sizes=size, alpha=alpha_val, activation='tanh', solver='lbfgs', max_iter=400, random_state=42)
                model.fit(X_tr, y_tr)
                scores.append(model.score(X_val, y_val))
            mean_score = np.mean(scores)
            if mean_score > best_mlp_score:
                best_mlp_score = mean_score
                best_mlp_params = {"hidden_layer_sizes": [int(x) for x in size], "alpha": float(alpha_val)}
                
    best_params["mlp"] = best_mlp_params
    
    return best_params
