import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


GOLD_MODEL_VERSION = "gold-oos-calibrated-v8-20260906"


GOLD_MACRO_LEVEL_FEATURES = [
    "dxy", "us10y", "vix", "brent", "dji", "spx", "eurusd", "xagusd",
    "real_yield", "gld", "gld_trust", "days_to_fed", "days_to_cpi", "days_to_nfp",
    "days_to_pce", "pce_headline_yoy", "pce_core_yoy", "pce_headline_mom",
    "pce_core_mom",
]

GOLD_MACRO_DERIVED_FEATURES = [
    "dxy_return_1d", "dxy_return_5d",
    "vix_return_1d", "vix_return_5d",
    "brent_return_1d", "brent_return_5d",
    "dji_return_1d", "dji_return_5d",
    "spx_return_1d", "spx_return_5d",
    "eurusd_return_1d", "eurusd_return_5d",
    "xagusd_return_1d", "xagusd_return_5d",
    "gld_return_1d", "gld_return_5d",
    "gld_trust_return_1d", "gld_trust_return_5d",
    "us10y_change_1d", "us10y_change_5d",
    "real_yield_change_1d", "real_yield_change_5d",
    "pce_headline_yoy_change", "pce_core_yoy_change",
    "pce_headline_mom_change", "pce_core_mom_change",
    "gold_silver_ratio", "event_risk_3d",
]

GOLD_CORE_MODEL_FEATURES = [
    "return_1d", "return_2d", "return_3d", "return_5d", "momentum_10",
    "price_accel", "rsi", "rsi_slope", "macd_hist", "macd_hist_slope",
    "ema_ratio", "dist_sma20", "bb_position", "realized_vol_5",
    "realized_vol_20", "drawdown_20", "trend_strength",
]

GOLD_STABLE_MACRO_FEATURES = [
    "dxy_return_1d", "us10y_change_1d", "real_yield_change_1d",
    "days_to_fed", "days_to_cpi", "days_to_nfp", "days_to_pce",
    "pce_headline_yoy", "pce_core_yoy", "pce_headline_mom", "pce_core_mom",
    "pce_headline_yoy_change", "pce_core_yoy_change",
    "pce_headline_mom_change", "pce_core_mom_change", "event_risk_3d",
]


def _available_macro_features(df):
    """Use macro features with broad chronological coverage.

    A late-starting series used to pass the old five-row threshold and then
    `dropna` discarded most training history.  That left roughly 40 samples
    for dozens of features and made every holdout R² unstable/negative.
    """
    available = []
    min_observations = max(30, int(np.ceil(len(df) * 0.65)))
    for col in GOLD_MACRO_LEVEL_FEATURES + GOLD_MACRO_DERIVED_FEATURES:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.notna().sum() >= min_observations and pd.notna(values.iloc[-1]):
            available.append(col)
    return available


def _gold_model_feature_columns(df):
    """Return all available technical, macro and event channels for CNN."""
    ordered = [col for col in GOLD_CORE_MODEL_FEATURES if col in df.columns]
    ordered.extend(
        col for col in GOLD_MACRO_LEVEL_FEATURES + GOLD_MACRO_DERIVED_FEATURES
        if col in df.columns
    )
    ordered.extend(
        col for col in ("days_to_fed", "days_to_cpi", "days_to_nfp", "days_to_pce", "event_risk_3d")
        if col in df.columns
    )
    return list(dict.fromkeys(ordered))


GOLD_CNN_FEATURE_DEFAULTS = {
    "return_1d": 0.0, "return_2d": 0.0, "return_3d": 0.0, "return_5d": 0.0,
    "momentum_10": 0.0, "price_accel": 0.0, "rsi": 50.0, "rsi_slope": 0.0,
    "macd_hist": 0.0, "macd_hist_slope": 0.0, "ema_ratio": 0.0,
    "dist_sma20": 0.0, "bb_position": 0.5, "realized_vol_5": 0.0,
    "realized_vol_20": 0.0, "drawdown_20": 0.0, "trend_strength": 0.0,
    "dxy": 100.0, "us10y": 4.0, "vix": 15.0, "brent": 75.0,
    "dji": 35000.0, "spx": 5000.0, "eurusd": 1.08, "xagusd": 25.0,
    "real_yield": 1.5, "gld": 200.0, "gld_trust": 200.0,
    "days_to_fed": 15.0, "days_to_cpi": 15.0, "days_to_nfp": 15.0,
    "days_to_pce": 15.0, "pce_headline_yoy": 2.5, "pce_core_yoy": 2.7,
    "pce_headline_mom": 0.2, "pce_core_mom": 0.2, "event_risk_3d": 0.0,
    "gold_silver_ratio": 80.0,
}


def _prepare_gold_model_features(df, feature_cols):
    """Prepare shared technical, macro and event channels without leakage."""
    frame = df.copy()
    for column in feature_cols:
        values = pd.to_numeric(frame[column], errors="coerce")
        default = GOLD_CNN_FEATURE_DEFAULTS.get(column, 0.0)
        # Only carry information forward. Missing values before the first
        # release receive a fixed domain baseline, never a future value.
        frame[column] = values.ffill().fillna(default)
    return frame


def _normalize_weights(weights, keys):
    clean = {}
    for key in keys:
        try:
            value = float((weights or {}).get(key, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        clean[key] = value if np.isfinite(value) and value > 0 else 0.0
    total = sum(clean.values())
    if total <= 0:
        return {key: 1.0 / len(keys) for key in keys}
    return {key: value / total for key, value in clean.items()}


def _robust_clip_targets(values, sigma=3.0):
    """Winsorize return targets with train-only median/MAD limits."""
    array = np.asarray(values, dtype=float)
    original_shape = array.shape
    matrix = array.reshape(-1, 1) if array.ndim == 1 else array.copy()
    clipped = matrix.copy()
    for column in range(matrix.shape[1]):
        series = matrix[:, column]
        median = float(np.median(series))
        mad_sigma = 1.4826 * float(np.median(np.abs(series - median)))
        if not np.isfinite(mad_sigma) or mad_sigma < 1e-6:
            mad_sigma = float(np.std(series))
        if np.isfinite(mad_sigma) and mad_sigma > 0:
            clipped[:, column] = np.clip(
                series,
                median - sigma * mad_sigma,
                median + sigma * mad_sigma,
            )
    return clipped.reshape(original_shape)


def _predict_direction_and_regime(df, profit_taking_risk=None):
    """Train a separate direction classifier and describe the current regime.

    Regression estimates magnitude; this classifier estimates sign.  Its gate
    is enabled only when chronological holdout accuracy beats the majority
    baseline, preventing an unproven classifier from influencing Ensemble.
    """
    # Use the same stationary technical + macro/event channels as the
    # regressors. Current news is handled by the auditable context overlay
    # because a historical daily news-score series is not available.
    feature_cols = _gold_model_feature_columns(df)
    work = _prepare_gold_model_features(df, feature_cols)[feature_cols + ["close"]].copy()
    target_return = work["close"].shift(-1) / work["close"] - 1.0
    valid = target_return.notna() & work[feature_cols].notna().all(axis=1)
    X = work.loc[valid, feature_cols].to_numpy(dtype=float)
    returns = target_return.loc[valid].to_numpy(dtype=float)
    y = (returns > 0.0).astype(int)

    result = {
        "model": "extra_trees_direction_classifier",
        "eligible": False,
        "probability_up": 0.5,
        "probability_down": 0.5,
        "confidence": 0.0,
        "holdout_accuracy": None,
        "balanced_accuracy": None,
        "majority_baseline": None,
        "skill_vs_majority": 0.0,
        "holdout_samples": 0,
        "expected_move": 0.0,
        "regime": "unknown",
        "regime_inputs": {},
        "top_features": [],
    }
    if len(X) < 50 or len(np.unique(y)) < 2:
        return result

    def make_classifier():
        return ExtraTreesClassifier(
            n_estimators=400,
            max_depth=3,
            min_samples_leaf=8,
            max_features=0.60,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )

    # Expanding-window validation spans several regimes instead of depending
    # on one final split. Each baseline is learned from its training fold;
    # looking at the validation class distribution would leak hindsight.
    from sklearn.model_selection import TimeSeriesSplit

    validation_actual = []
    validation_predicted = []
    validation_baseline = []
    recent_actual = np.asarray([], dtype=int)
    recent_predicted = np.asarray([], dtype=int)
    recent_baseline = np.asarray([], dtype=int)
    fold_count = 0
    for train_index, valid_index in TimeSeriesSplit(n_splits=4).split(X):
        if len(train_index) < 30 or len(valid_index) < 10:
            continue
        validation_model = make_classifier()
        validation_model.fit(X[train_index], y[train_index])
        fold_predicted = validation_model.predict(X[valid_index])
        majority_label = int(np.mean(y[train_index]) >= 0.5)
        fold_baseline = np.full(len(valid_index), majority_label, dtype=int)
        fold_actual = y[valid_index]
        validation_actual.extend(fold_actual.tolist())
        validation_predicted.extend(fold_predicted.tolist())
        validation_baseline.extend(fold_baseline.tolist())
        recent_actual = fold_actual
        recent_predicted = fold_predicted
        recent_baseline = fold_baseline
        fold_count += 1
    if not validation_actual:
        return result

    valid_y = np.asarray(validation_actual, dtype=int)
    validation_pred = np.asarray(validation_predicted, dtype=int)
    baseline_pred = np.asarray(validation_baseline, dtype=int)
    accuracy = float(np.mean(validation_pred == valid_y))
    majority = float(np.mean(baseline_pred == valid_y))
    recent_accuracy = float(np.mean(recent_predicted == recent_actual))
    recent_majority = float(np.mean(recent_baseline == recent_actual))
    recalls = []
    for label in (0, 1):
        mask = valid_y == label
        if np.any(mask):
            recalls.append(float(np.mean(validation_pred[mask] == label)))
    balanced_accuracy = float(np.mean(recalls)) if recalls else 0.5
    skill = float(np.clip(
        (accuracy - majority) / max(1.0 - majority, 1e-10),
        -1.0,
        1.0,
    ))

    final_model = make_classifier()
    final_model.fit(X, y)
    latest = work.iloc[-1][feature_cols].to_numpy(dtype=float).reshape(1, -1)
    class_probabilities = dict(zip(
        final_model.classes_.tolist(),
        final_model.predict_proba(latest)[0].tolist(),
    ))
    probability_up = float(class_probabilities.get(1, 0.5))
    robust_move = float(np.median(np.abs(returns[-60:]))) if len(returns) else 0.0
    eligible = bool(
        len(valid_y) >= 15
        and fold_count >= 3
        and accuracy >= majority + 0.03
        and balanced_accuracy >= 0.54
        and recent_accuracy >= recent_majority
        and skill > 0.0
    )
    result.update({
        "eligible": eligible,
        "probability_up": probability_up,
        "probability_down": float(1.0 - probability_up),
        "confidence": float(abs(probability_up - 0.5) * 2.0),
        "holdout_accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "majority_baseline": majority,
        "skill_vs_majority": skill,
        "holdout_samples": int(len(valid_y)),
        "walk_forward_splits": int(fold_count),
        "recent_holdout_accuracy": recent_accuracy,
        "recent_majority_baseline": recent_majority,
        "expected_move": robust_move,
        "top_features": [
            {"feature": feature_cols[index], "importance": float(importance)}
            for index, importance in sorted(
                enumerate(final_model.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )[:8]
        ],
    })

    latest_row = df.iloc[-1]

    def finite_number(value, default=0.0):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return number if np.isfinite(number) else default

    vol_5 = finite_number(latest_row.get("realized_vol_5"))
    vol_20 = finite_number(latest_row.get("realized_vol_20"))
    trend = finite_number(latest_row.get("trend_strength"))
    event_risk = finite_number(latest_row.get("event_risk_3d"))
    profit_score = finite_number((profit_taking_risk or {}).get("score"))
    if (profit_taking_risk or {}).get("level") in {"medium", "high"}:
        regime = "profit_taking_risk"
    elif event_risk >= 1.0:
        regime = "event_risk"
    elif vol_20 > 0 and vol_5 / vol_20 >= 1.5:
        regime = "high_volatility"
    elif trend >= 0.004:
        regime = "uptrend"
    elif trend <= -0.004:
        regime = "downtrend"
    else:
        regime = "sideways"
    result["regime"] = regime
    result["regime_inputs"] = {
        "volatility_ratio_5_20": float(vol_5 / vol_20) if vol_20 > 0 else None,
        "trend_strength": trend,
        "event_risk_3d": event_risk,
        "profit_taking_score": profit_score,
    }
    return result


def _adaptive_ensemble_weights(prior_weights, validation_r2, model_implementations):
    """Blend historical weights with current chronological holdout quality.

    A fallback model is heavily discounted so the ensemble does not count the
    same underlying forecast two or three times when optional dependencies fail.
    """
    keys = ["random_forest", "linear_regression", "mlp", "xgboost", "lstm", "cnn"]
    priors = _normalize_weights(prior_weights, keys)
    quality = {}
    for key in keys:
        try:
            r2 = float(validation_r2.get(key, -1.0))
        except (TypeError, ValueError):
            r2 = -1.0
        if not np.isfinite(r2):
            r2 = -1.0
        # Keep negative OOS scores meaningful without allowing one noisy split
        # to reduce a model to exactly zero weight.
        # Preserve separation between mildly negative and catastrophically
        # negative scores; clipping all values at -1 would give a broken model
        # the same weight as a merely weak one.
        quality[key] = float(np.exp(0.65 * np.clip(r2, -4.0, 0.85)))
        implementation = str((model_implementations or {}).get(key, key))
        if "fallback" in implementation:
            quality[key] *= 0.15

    quality = _normalize_weights(quality, keys)
    combined = {
        key: 0.35 * priors[key] + 0.65 * quality[key]
        for key in keys
    }
    # A small floor prevents abrupt all-or-nothing changes between refreshes.
    combined = {key: max(0.03, value) for key, value in combined.items()}
    return _normalize_weights(combined, keys)


def _robust_daily_volatility(df):
    """Estimate daily volatility while limiting the influence of bad price ticks."""
    returns = pd.to_numeric(df.get("return_1d"), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().tail(60)
    if len(returns) < 5:
        return 0.01
    median = float(returns.median())
    mad_sigma = 1.4826 * float(np.median(np.abs(returns - median)))
    if not np.isfinite(mad_sigma) or mad_sigma <= 0:
        mad_sigma = float(returns.clip(returns.quantile(0.05), returns.quantile(0.95)).std())
    return float(np.clip(mad_sigma, 0.004, 0.03))


def _estimate_profit_taking_risk(df, market_signals=None):
    """Estimate an auditable profit-taking regime from present/past data.

    The output is a 0-100 risk score, not a calibrated probability.  It looks
    for the common setup preceding profit-taking: a fast run-up, price stretched
    above SMA20, overbought RSI/Bollinger position, and momentum rolling over.
    """
    latest = df.iloc[-1]

    def numeric(name, default=0.0):
        try:
            value = float(latest.get(name, default))
            return value if np.isfinite(value) else float(default)
        except (TypeError, ValueError):
            return float(default)

    return_5d = numeric("return_5d")
    return_1d = numeric("return_1d")
    dist_sma20 = numeric("dist_sma20")
    rsi = numeric("rsi", 50.0)
    bb_position = numeric("bb_position", 0.5)
    macd_rollover = max(0.0, -numeric("macd_hist_slope"))
    rsi_rollover = max(0.0, -numeric("rsi_slope"))

    components = {
        "Tăng nhanh 5 phiên": float(np.clip(return_5d / 0.05, 0.0, 1.0)),
        "Giá cách xa SMA20": float(np.clip(dist_sma20 / 0.04, 0.0, 1.0)),
        "RSI cao": float(np.clip((rsi - 60.0) / 20.0, 0.0, 1.0)),
        "Sát/vượt dải trên Bollinger": float(np.clip((bb_position - 0.70) / 0.30, 0.0, 1.0)),
        "Động lượng bắt đầu suy yếu": float(np.clip(
            max(macd_rollover / max(abs(numeric("macd")), 1e-6), rsi_rollover / 8.0),
            0.0,
            1.0,
        )),
    }
    weights = {
        "Tăng nhanh 5 phiên": 0.30,
        "Giá cách xa SMA20": 0.25,
        "RSI cao": 0.20,
        "Sát/vượt dải trên Bollinger": 0.15,
        "Động lượng bắt đầu suy yếu": 0.10,
    }
    technical_risk = sum(weights[name] * value for name, value in components.items())

    news_confirmed = any(
        "chốt lời" in " ".join([
            str(signal.get("title", "")),
            " ".join(str(driver) for driver in (signal.get("drivers") or [])),
        ]).lower()
        for signal in (market_signals or [])
    )
    # News confirms an observed regime but cannot manufacture a high score on
    # its own.  This keeps a stale headline from dominating a fresh price path.
    score = float(np.clip(0.85 * technical_risk + (0.15 if news_confirmed else 0.0), 0.0, 1.0))
    if score >= 0.65:
        level = "high"
        label = "Cao"
    elif score >= 0.40:
        level = "medium"
        label = "Trung bình"
    else:
        level = "low"
        label = "Thấp"

    drivers = [
        name for name, value in sorted(components.items(), key=lambda item: item[1], reverse=True)
        if value >= 0.35
    ][:4]
    if news_confirmed:
        drivers.append("Tin thị trường xác nhận lực bán chốt lời")
    if return_5d > 0 and return_1d < 0:
        drivers.append("Giá đảo chiều giảm sau nhịp tăng 5 phiên")

    return {
        "score": score,
        "score_percent": round(score * 100.0, 1),
        "level": level,
        "label": label,
        "is_news_confirmed": news_confirmed,
        "drivers": drivers[:5],
        "components": components,
        "interpretation": "Điểm rủi ro theo chế độ thị trường, không phải xác suất chắc chắn.",
    }


def _summarize_market_context(
    market_signals,
    technical_score,
    profit_taking_risk=None,
    fed_policy_outlook=None,
):
    """Convert released macro/policy context into an auditable direction score.

    This layer is intentionally capped and nudges every model through one
    shared, auditable overlay. It does not fabricate historical values or let
    a news headline override price volatility. Positive means supportive for
    Gold, negative means bearish.
    """
    signals = list(market_signals or [])
    directional_scores = []
    drivers = []
    macro_values = {}

    for signal in signals:
        bias = str(signal.get("short_term_bias", "neutral")).lower()
        if bias in {"bearish", "down", "negative"}:
            directional_scores.append(-1.0)
        elif bias in {"bullish", "supportive", "up", "positive"}:
            directional_scores.append(1.0)

        for driver in signal.get("drivers") or []:
            if driver and driver not in drivers:
                drivers.append(str(driver))

        values = signal.get("macro_values") or {}
        if isinstance(values, dict):
            macro_values.update(values)

    components = []
    if directional_scores:
        components.append((0.30, float(np.mean(directional_scores))))

    # The policy layer already combines inflation, labour, rates and FED
    # speech signals. Positive means supportive for Gold; negative means
    # restrictive/hawkish pressure on Gold.
    try:
        fed_gold_score = float(
            (fed_policy_outlook or {}).get("gold_implication", {}).get("score", 0.0)
        )
    except (TypeError, ValueError):
        fed_gold_score = 0.0
    if fed_policy_outlook:
        components.append((0.30, float(np.clip(fed_gold_score, -1.0, 1.0))))
        for driver in (fed_policy_outlook.get("drivers") or [])[:3]:
            if driver and driver not in drivers:
                drivers.append(str(driver))

    # A hotter-than-consensus PCE print is normally bearish for Gold in the
    # short run because it supports higher-for-longer rates and real yields.
    try:
        pce_surprise = float(macro_values.get("pce_surprise_yoy"))
    except (TypeError, ValueError):
        pce_surprise = np.nan
    if np.isfinite(pce_surprise):
        surprise_score = -float(np.clip(pce_surprise / 0.20, -1.0, 1.0))
        components.append((0.15, surprise_score))

    try:
        profit_score = float((profit_taking_risk or {}).get("score", 0.0))
    except (TypeError, ValueError):
        profit_score = 0.0
    # Low background risk is neutral. Only a meaningful overextension setup
    # contributes a capped bearish adjustment.
    profit_signal = float(np.clip((profit_score - 0.25) / 0.75, 0.0, 1.0))
    if profit_signal > 0:
        components.append((0.15, -profit_signal))

    tech_direction = float(np.clip(float(technical_score) / 3.0, -1.0, 1.0))
    components.append((0.10, tech_direction))

    # Weights are absolute confidence contributions. Missing news/PCE remains
    # neutral instead of magnifying the remaining weak component to 100%.
    score = sum(weight * value for weight, value in components)
    score = float(np.clip(score, -1.0, 1.0))
    if score <= -0.15:
        label = "bearish"
    elif score >= 0.15:
        label = "bullish"
    else:
        label = "neutral"

    return {
        "score": score,
        "label": label,
        "drivers": drivers[:4],
        "macro_values": macro_values,
        "components": [
            {"weight": float(weight), "score": float(value)}
            for weight, value in components
        ],
        "fed_policy_outlook": fed_policy_outlook or {},
    }


def _stabilize_price_path(pred_prices, current_price, daily_volatility):
    """Clip implausible cumulative moves using robust recent volatility."""
    stabilized = []
    for horizon, price in enumerate(np.asarray(pred_prices, dtype=float), start=1):
        cumulative_limit = float(np.clip(3.0 * daily_volatility * np.sqrt(horizon), 0.01, 0.12))
        predicted_return = float(price / (current_price + 1e-10) - 1.0)
        stabilized.append(current_price * (1.0 + np.clip(predicted_return, -cumulative_limit, cumulative_limit)))
    return np.asarray(stabilized, dtype=float)


def _time_series_split_index(n_samples, min_train=12, min_valid=5):
    """Return a chronological holdout split, or None when the sample is too small."""
    if n_samples < min_train + min_valid:
        return None
    split = max(min_train, int(n_samples * 0.8))
    if n_samples - split < min_valid:
        split = n_samples - min_valid
    return split if split >= min_train else None


def _out_of_sample_r2(y_true, y_pred):
    """Calculate R² safely for a time-ordered validation segment."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.isclose(np.var(y_true), 0.0):
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def _holdout_r2(model_factory, X, y, scaler_factory=None):
    """Fit only on the past portion and score on the unseen future portion."""
    split = _time_series_split_index(len(X))
    if split is None:
        return 0.0

    train_X, valid_X = X[:split], X[split:]
    if scaler_factory is not None:
        scaler = scaler_factory()
        train_X = scaler.fit_transform(train_X)
        valid_X = scaler.transform(valid_X)

    model = model_factory()
    model.fit(train_X, _robust_clip_targets(y[:split]))
    return _out_of_sample_r2(y[split:], model.predict(valid_X))

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

    # 12. Regime features.  Models need to know whether the same absolute
    # indicator value is happening in a calm or volatile market.
    df["realized_vol_5"] = df["return_1d"].rolling(window=5).std()
    df["realized_vol_20"] = df["return_1d"].rolling(window=20).std()
    rolling_high_20 = df["close"].rolling(window=20).max()
    df["drawdown_20"] = df["close"] / (rolling_high_20 + 1e-10) - 1.0
    df["trend_strength"] = (df["sma_5"] - df["sma_20"]) / (df["close"] + 1e-10)

    # 13. Macro changes are more useful for a short-horizon return forecast
    # than raw levels alone.  Every value is calculated from present/past rows.
    price_like_macro = ["dxy", "vix", "brent", "dji", "spx", "eurusd", "xagusd", "gld", "gld_trust"]
    for col in price_like_macro:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            df[f"{col}_return_1d"] = numeric.pct_change(1, fill_method=None)
            df[f"{col}_return_5d"] = numeric.pct_change(5, fill_method=None)

    for col in ("us10y", "real_yield"):
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            df[f"{col}_change_1d"] = numeric.diff(1)
            df[f"{col}_change_5d"] = numeric.diff(5)

    # PCE is monthly and remains unchanged between releases.  A one-row
    # change therefore marks the first trading session that can use the newly
    # released value without leaking it into earlier rows.
    for col in ("pce_headline_yoy", "pce_core_yoy", "pce_headline_mom", "pce_core_mom"):
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            df[f"{col}_change"] = numeric.diff(1)

    if "xagusd" in df.columns:
        silver = pd.to_numeric(df["xagusd"], errors="coerce")
        df["gold_silver_ratio"] = df["close"] / (silver + 1e-10)

    event_cols = [col for col in ("days_to_fed", "days_to_cpi", "days_to_nfp", "days_to_pce") if col in df.columns]
    if event_cols:
        event_distance = df[event_cols].apply(pd.to_numeric, errors="coerce").min(axis=1)
        df["event_risk_3d"] = (event_distance <= 3).astype(float)
    
    return df


def predict_future_prices(df, days_to_predict=5, params=None):
    """
    Uses strongly regularized linear regression on stationary return features.

    Regressing price level on a time counter failed badly after regime jumps.
    This version predicts cumulative returns and shrinks noisy coefficients
    toward the liquid-market persistence baseline (tomorrow ~= today).
    """
    from sklearn.linear_model import Ridge

    frame = df if "return_1d" in df.columns else calculate_technical_indicators(df)
    feature_cols = _gold_model_feature_columns(frame)
    prepared = _prepare_gold_model_features(frame, feature_cols)
    work = prepared[feature_cols + ["close"]].copy()
    work = work.dropna(subset=["close"])
    current_price = float(work["close"].iloc[-1])
    if len(work) < 30:
        return np.repeat(current_price, days_to_predict), 0.0, 0.0

    predicted_prices = []
    r2_scores = []
    latest_features = work.iloc[-1][feature_cols].to_numpy(dtype=float).reshape(1, -1)
    for horizon in range(1, days_to_predict + 1):
        target = work["close"].shift(-horizon) / work["close"] - 1.0
        valid = target.notna()
        X = work.loc[valid, feature_cols].to_numpy(dtype=float)
        y = target.loc[valid].to_numpy(dtype=float)

        split = _time_series_split_index(len(X))
        if split is not None:
            validation_scaler = StandardScaler()
            validation_X = validation_scaler.fit_transform(X[:split])
            ridge_alpha = float(np.clip(params.get("alpha", 1000.0) if params else 1000.0, 100.0, 10000.0))
            validation_model = Ridge(alpha=ridge_alpha)
            validation_model.fit(validation_X, _robust_clip_targets(y[:split]))
            validation_pred = validation_model.predict(validation_scaler.transform(X[split:]))
            r2_scores.append(_out_of_sample_r2(y[split:], validation_pred))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        ridge_alpha = float(np.clip(params.get("alpha", 1000.0) if params else 1000.0, 100.0, 10000.0))
        model = Ridge(alpha=ridge_alpha)
        model.fit(X_scaled, _robust_clip_targets(y))
        predicted_return = float(model.predict(scaler.transform(latest_features))[0])
        predicted_prices.append(current_price * (1.0 + predicted_return))

    prices = np.asarray(predicted_prices, dtype=float)
    slope = float(prices[-1] - current_price) / max(len(prices), 1)
    return prices, slope, float(np.mean(r2_scores)) if r2_scores else 0.0


def predict_future_prices_rf(df, days_to_predict=5, params=None):
    """
    Uses Random Forest Regressor trained on technical features
    to predict the next N days' closing prices via return forecasting.
    Falls back to Linear Regression if data is insufficient.
    """
    FEATURE_COLS = _gold_model_feature_columns(df)
    
    # We need at least days_to_predict look-ahead for labeling, plus some rows of features
    MIN_ROWS = 15 + days_to_predict
    
    # All base models use the same technical + macro + event channels. This
    # keeps the model comparison fair instead of giving only CNN access to
    # FED, PCE and event-proximity information.
    df_feat = _prepare_gold_model_features(df, FEATURE_COLS)
    df_feat = df_feat.dropna(subset=["close"])
    
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
        # The caller already removed weekend carry-forward rows. Keep genuine
        # near-zero trading sessions: dropping them creates selection bias and
        # teaches every model to predict moves that are too large.
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
        max_depth = int(np.clip(params.get("max_depth", 3) if params else 3, 2, 4))
        min_samples_leaf = max(8, int(params.get("min_samples_leaf", 10) if params else 10))
        def make_rf():
            return RandomForestRegressor(
                n_estimators=int(np.clip(params.get("n_estimators", 300) if params else 300, 120, 500)),
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=(params.get("max_features", "sqrt") if params else "sqrt"),
                random_state=42,
                n_jobs=1,
            )

        r2 = _holdout_r2(make_rf, X_train, y_train[:, d_idx])
        rf = make_rf()
        rf.fit(X_train, _robust_clip_targets(y_train[:, d_idx]))
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
    FEATURE_COLS = _gold_model_feature_columns(df)
    
    MIN_ROWS = 15 + days_to_predict
    df_feat = _prepare_gold_model_features(df, FEATURE_COLS)
    df_feat = df_feat.dropna(subset=["close"])
    
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
        # Keep genuine near-zero trading sessions (weekends were removed).
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
    
    hidden_layer_sizes = params.get("hidden_layer_sizes", (8, 4)) if params else (8, 4)
    alpha = max(5.0, float(params.get("alpha", 5.0) if params else 5.0))
    if isinstance(hidden_layer_sizes, (int, float)):
        hidden_layer_sizes = (int(hidden_layer_sizes),)
    elif isinstance(hidden_layer_sizes, list):
        hidden_layer_sizes = tuple(hidden_layer_sizes)
    # Bound legacy/tuned configurations that are too large for ~150 samples.
    hidden_layer_sizes = tuple(max(2, min(16, int(size))) for size in hidden_layer_sizes[:2])

    def make_mlp():
        return MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            activation="tanh",
            solver="lbfgs",
            max_iter=400,
            random_state=42,
        )

    # MLPRegressor supports multi-output targets.  Training one shared network
    # is both faster and more coherent than fitting ten separate networks
    # (five validation models plus five final models).
    split = _time_series_split_index(len(X_train))
    r2_scores = []
    if split is not None:
        validation_scaler = StandardScaler()
        validation_X = validation_scaler.fit_transform(X_train[:split])
        validation_model = make_mlp()
        validation_targets = y_train[:split, 0] if days_to_predict == 1 else y_train[:split]
        validation_model.fit(validation_X, _robust_clip_targets(validation_targets))
        validation_pred = np.asarray(
            validation_model.predict(validation_scaler.transform(X_train[split:])),
            dtype=float,
        ).reshape(len(X_train) - split, days_to_predict)
        for d_idx in range(days_to_predict):
            r2_scores.append(_out_of_sample_r2(y_train[split:, d_idx], validation_pred[:, d_idx]))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    mlp = make_mlp()
    final_targets = y_train[:, 0] if days_to_predict == 1 else y_train
    mlp.fit(X_train_scaled, _robust_clip_targets(final_targets))
    latest_features = np.nan_to_num(
        df_feat.iloc[-1][FEATURE_COLS].values.reshape(1, -1).astype(np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    ret_preds = np.asarray(mlp.predict(scaler.transform(latest_features)), dtype=float).reshape(-1)
    predicted_prices = current_price * (1.0 + ret_preds)
    avg_r2 = float(np.mean(r2_scores)) if r2_scores else 0.0
    
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
    FEATURE_COLS = _gold_model_feature_columns(df)
    
    MIN_ROWS = 15 + days_to_predict
    df_feat = _prepare_gold_model_features(df, FEATURE_COLS)
    df_feat = df_feat.dropna(subset=["close"])
    
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
        # Keep genuine near-zero trading sessions (weekends were removed).
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
        predicted_prices = []
        r2_scores = []
        
        for d_idx in range(days_to_predict):
            max_depth = int(np.clip(params.get("max_depth", 1) if params else 1, 1, 2))
            learning_rate = float(np.clip(params.get("learning_rate", 0.02) if params else 0.02, 0.01, 0.03))

            def make_xgb():
                model = xgb.XGBRegressor(
                    n_estimators=int(np.clip(params.get("n_estimators", 80) if params else 80, 50, 180)),
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    min_child_weight=float(np.clip(params.get("min_child_weight", 15) if params else 15, 8, 30)),
                    gamma=0.001,
                    reg_alpha=1.0,
                    reg_lambda=float(np.clip(params.get("reg_lambda", 30.0) if params else 30.0, 10.0, 80.0)),
                    random_state=42,
                    n_jobs=1,
                    verbosity=0,
                    objective="reg:absoluteerror",
                )
                return model

            r2 = _holdout_r2(make_xgb, X_train, y_train[:, d_idx])
            model = make_xgb()
            model.fit(X_train, _robust_clip_targets(y_train[:, d_idx]))
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
    FEATURE_COLS = _gold_model_feature_columns(df)

    SEQ_LEN = 20       # Look-back window in trading days
    MIN_ROWS = SEQ_LEN + days_to_predict + 5
    EPOCHS   = 50
    LR       = 3e-3

    df_feat = _prepare_gold_model_features(df, FEATURE_COLS)
    df_feat = df_feat.dropna(subset=["close"])

    if len(df_feat) < MIN_ROWS:
        preds, slope, r2 = predict_future_prices(df, days_to_predict)
        return preds, slope, r2, "linear_regression_fallback"

    current_price = df_feat["close"].iloc[-1]

    try:
        import torch
        torch.set_num_threads(1)  # Prevent OpenMP deadlock in thread pools/executors
        torch.manual_seed(42)
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler as _SS


        # Build sequences (X) and targets (y = cumulative returns relative to day i)
        raw_features = df_feat[FEATURE_COLS].values.astype(float)
        scaler = _SS()
        X_all = scaler.fit_transform(raw_features)

        X_seqs, y_seqs, sequence_indices = [], [], []
        n = len(X_all)
        for i in range(SEQ_LEN, n - days_to_predict):
            current_close = df_feat["close"].iloc[i]
            # Calculate cumulative returns for days 1 to days_to_predict ahead
            targets = []
            for d in range(1, days_to_predict + 1):
                future_close = df_feat["close"].iloc[i + d]
                ret = (future_close - current_close) / (current_close + 1e-10)
                targets.append(ret)
                
            # Keep genuine near-zero trading sessions (weekends were removed).
            X_seqs.append(X_all[i - SEQ_LEN:i])
            y_seqs.append(targets)
            sequence_indices.append(i)

        if len(X_seqs) < 10:
            raise ValueError("Not enough non-zero sequences for LSTM training")

        X_t = torch.tensor(np.array(X_seqs), dtype=torch.float32)   # (N, SEQ_LEN, F)
        y_t = torch.tensor(np.array(y_seqs), dtype=torch.float32)   # (N, days_to_predict)
        y_fit_t = torch.tensor(
            _robust_clip_targets(np.array(y_seqs)), dtype=torch.float32
        )

        n_feat = X_t.shape[2]

        # Compact LSTM: 1 layer, hidden=32
        class GoldLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(n_feat, 16, num_layers=1,
                                    batch_first=True, dropout=0.0)
                self.fc   = nn.Sequential(
                    nn.Linear(16, 8),
                    nn.Tanh(),
                    nn.Linear(8, days_to_predict)
                )
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])  # last timestep

        def train_lstm(train_X, train_y, epochs):
            model = GoldLSTM()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-2)
            loss_fn = nn.MSELoss()
            model.train()
            for _ in range(epochs):
                optimizer.zero_grad()
                pred = model(train_X)
                loss = loss_fn(pred, train_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            return model

        # Hold out the latest chronological sequences for an honest metric.
        # The validation scaler is fitted only on data available before that
        # holdout; the final scaler/model below may use all historical data.
        split = _time_series_split_index(len(X_seqs), min_train=10, min_valid=5)
        use_persistence_fallback = False
        output_scale = 1.0
        if split is not None:
            validation_scaler = _SS()
            validation_scaler.fit(raw_features[:sequence_indices[split]])
            validation_features = validation_scaler.transform(raw_features)
            validation_X = np.array([
                validation_features[i - SEQ_LEN:i] for i in sequence_indices
            ], dtype=np.float32)
            validation_model = train_lstm(
                torch.tensor(validation_X[:split], dtype=torch.float32),
                y_fit_t[:split],
                max(20, EPOCHS // 2),
            )
            validation_model.eval()
            with torch.no_grad():
                validation_pred = validation_model(
                    torch.tensor(validation_X[split:], dtype=torch.float32)
                ).numpy()
            validation_true = y_t[split:].numpy()
            scale_candidates = (0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
            scale_scores = [
                _out_of_sample_r2(validation_true, validation_pred * scale)
                for scale in scale_candidates
            ]
            best_scale_index = int(np.argmax(scale_scores))
            output_scale = float(scale_candidates[best_scale_index])
            avg_r2 = float(scale_scores[best_scale_index])
            persistence_r2 = _out_of_sample_r2(
                validation_true,
                np.zeros_like(validation_true),
            )
            if output_scale < 1.0 or avg_r2 <= persistence_r2:
                use_persistence_fallback = True
        else:
            avg_r2 = 0.0

        model_lstm = train_lstm(X_t, y_fit_t, EPOCHS)

        # Inference: use the last SEQ_LEN rows
        model_lstm.eval()
        with torch.no_grad():
            last_seq = torch.tensor(
                X_all[-SEQ_LEN:][np.newaxis, :, :], dtype=torch.float32
            )
            ret_preds = model_lstm(last_seq).numpy()[0]  # shape: (days_to_predict,)

        if use_persistence_fallback:
            # Do not replace every horizon with zero return.  That made the
            # dashboard display the same price for T+1...T+5 and hid the
            # shape of the LSTM path.  Keep the model's relative path, but
            # damp its magnitude because its chronological OOS score is
            # weaker than the persistence baseline.
            ret_preds = np.asarray(ret_preds, dtype=float) * output_scale

            # If the network itself emitted a perfectly flat path, use a
            # small, data-derived drift curve as a transparent fallback. It
            # can move down, up, or stay flat according to recent returns;
            # it never fabricates an alternating direction just to make the
            # cards look different.
            if len(ret_preds) > 1 and np.ptp(ret_preds) < 1e-8:
                close_values = pd.to_numeric(df_feat["close"], errors="coerce").dropna().to_numpy(dtype=float)
                if len(close_values) >= 3:
                    recent_returns = np.diff(np.log(np.maximum(close_values[-10:], 1e-10)))
                    drift = float(np.median(recent_returns[-5:])) if len(recent_returns) else 0.0
                    if not np.isfinite(drift):
                        drift = 0.0
                    ret_preds = drift * np.arange(1, days_to_predict + 1, dtype=float) * output_scale

        predicted_prices = [current_price * (1 + r) for r in ret_preds]

        if len(predicted_prices) > 1:
            slope = (predicted_prices[-1] - current_price) / max(len(predicted_prices), 1)
        else:
            slope = predicted_prices[0] - current_price

        implementation = "lstm_damped_low_confidence" if use_persistence_fallback else "lstm"
        return np.array(predicted_prices), slope, avg_r2, implementation

    except Exception as e:
        print(f"LSTM failed ({e}), falling back to XGBoost...")
        return predict_future_prices_xgb(df, days_to_predict)


def predict_future_prices_cnn(df, days_to_predict=5):
    """Predict Gold prices with a compact causal 1D CNN/TCN-style model.

    The network sees the same 20-session feature window as LSTM, but applies
    temporal convolutions over that window.  Targets are cumulative returns
    for T+1...T+N, so the output is a distinct multi-horizon path rather than
    one price copied across all forecast cards.
    """
    # Use the full stationary technical + macro/event feature set when a
    # column has enough chronological coverage. Each feature is a channel;
    # Conv1d slides only along the temporal axis.
    FEATURE_COLS = _gold_model_feature_columns(df)
    SEQ_LEN = 20
    MIN_ROWS = SEQ_LEN + days_to_predict + 5
    EPOCHS = 35
    LR = 2e-3

    df_feat = df.copy()
    for column in FEATURE_COLS:
        values = pd.to_numeric(df_feat[column], errors="coerce")
        default = GOLD_CNN_FEATURE_DEFAULTS.get(column, 0.0)
        # Forward-fill only uses information known at that date. Initial
        # rolling/event gaps receive a fixed domain baseline, never a future
        # value, so the CNN can consume the full macro/event matrix safely.
        df_feat[column] = values.ffill().fillna(default)
    df_feat = df_feat.dropna(subset=["close"])
    if len(df_feat) < MIN_ROWS:
        preds, slope, r2 = predict_future_prices(df, days_to_predict)
        return preds, slope, r2, "linear_regression_fallback_for_cnn"

    current_price = float(df_feat["close"].iloc[-1])

    try:
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(42)
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler as _SS

        raw_features = df_feat[FEATURE_COLS].values.astype(float)
        scaler = _SS()
        X_all = scaler.fit_transform(raw_features)

        X_seqs, y_seqs, sequence_indices = [], [], []
        n = len(X_all)
        for i in range(SEQ_LEN, n - days_to_predict):
            current_close = float(df_feat["close"].iloc[i])
            targets = []
            for horizon in range(1, days_to_predict + 1):
                future_close = float(df_feat["close"].iloc[i + horizon])
                targets.append((future_close - current_close) / (current_close + 1e-10))
            X_seqs.append(X_all[i - SEQ_LEN:i])
            y_seqs.append(targets)
            sequence_indices.append(i)

        if len(X_seqs) < 10:
            raise ValueError("Not enough non-zero sequences for CNN training")

        X_t = torch.tensor(np.asarray(X_seqs), dtype=torch.float32)
        y_t = torch.tensor(np.asarray(y_seqs), dtype=torch.float32)
        y_fit_t = torch.tensor(
            _robust_clip_targets(np.asarray(y_seqs)), dtype=torch.float32
        )
        n_feat = X_t.shape[2]

        class GoldCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.temporal = nn.Sequential(
                    nn.Conv1d(n_feat, 32, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv1d(32, 16, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(16, 8),
                    nn.Tanh(),
                    nn.Linear(8, days_to_predict),
                )

            def forward(self, x):
                # Input is (batch, sequence, features); Conv1d expects
                # (batch, channels, sequence).
                return self.head(self.temporal(x.transpose(1, 2)))

        def train_cnn(train_X, train_y, epochs):
            model = GoldCNN()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-2)
            loss_fn = nn.MSELoss()
            model.train()
            for _ in range(epochs):
                optimizer.zero_grad()
                prediction = model(train_X)
                loss = loss_fn(prediction, train_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            return model

        split = _time_series_split_index(len(X_seqs), min_train=10, min_valid=5)
        use_low_confidence_fallback = False
        output_scale = 1.0
        if split is not None:
            validation_scaler = _SS()
            validation_scaler.fit(raw_features[:sequence_indices[split]])
            validation_features = validation_scaler.transform(raw_features)
            validation_X = np.asarray([
                validation_features[i - SEQ_LEN:i] for i in sequence_indices
            ], dtype=np.float32)
            validation_model = train_cnn(
                torch.tensor(validation_X[:split], dtype=torch.float32),
                y_fit_t[:split],
                max(18, EPOCHS // 2),
            )
            validation_model.eval()
            with torch.no_grad():
                validation_pred = validation_model(
                    torch.tensor(validation_X[split:], dtype=torch.float32)
                ).numpy()
            validation_true = y_t[split:].numpy()
            scale_candidates = (0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
            scale_scores = [
                _out_of_sample_r2(validation_true, validation_pred * scale)
                for scale in scale_candidates
            ]
            best_scale_index = int(np.argmax(scale_scores))
            output_scale = float(scale_candidates[best_scale_index])
            avg_r2 = float(scale_scores[best_scale_index])
            persistence_r2 = _out_of_sample_r2(
                validation_true,
                np.zeros_like(validation_true),
            )
            if output_scale < 1.0 or avg_r2 <= persistence_r2:
                use_low_confidence_fallback = True
        else:
            avg_r2 = 0.0

        model_cnn = train_cnn(X_t, y_fit_t, EPOCHS)
        model_cnn.eval()
        with torch.no_grad():
            last_seq = torch.tensor(
                X_all[-SEQ_LEN:][np.newaxis, :, :], dtype=torch.float32
            )
            ret_preds = model_cnn(last_seq).numpy()[0]

        if use_low_confidence_fallback:
            # Retain the CNN path but reduce its magnitude when chronological
            # OOS quality is below the persistence baseline.
            ret_preds = np.asarray(ret_preds, dtype=float) * output_scale
            if len(ret_preds) > 1 and np.ptp(ret_preds) < 1e-8:
                close_values = pd.to_numeric(
                    df_feat["close"], errors="coerce"
                ).dropna().to_numpy(dtype=float)
                recent_returns = np.diff(np.log(np.maximum(close_values[-10:], 1e-10)))
                drift = float(np.median(recent_returns[-5:])) if len(recent_returns) else 0.0
                if not np.isfinite(drift):
                    drift = 0.0
                ret_preds = drift * np.arange(1, days_to_predict + 1, dtype=float) * output_scale

        predicted_prices = np.asarray(
            [current_price * (1.0 + float(ret)) for ret in ret_preds],
            dtype=float,
        )
        slope = (
            (predicted_prices[-1] - current_price) / max(len(predicted_prices), 1)
            if len(predicted_prices) > 1
            else predicted_prices[0] - current_price
        )
        implementation = "cnn_damped_low_confidence" if use_low_confidence_fallback else "cnn_1d"
        return predicted_prices, slope, float(avg_r2), implementation
    except Exception as error:
        print(f"CNN training/prediction failed ({error}), falling back to linear regression...")
        preds, slope, r2 = predict_future_prices(df, days_to_predict)
        return preds, slope, r2, "linear_regression_fallback_for_cnn"


def calculate_actionable_advice(pred_prices, current_price, final_score, rsi_val, bb_pos, slope):
    """Build gold-specific guidance from the forecast and technical score."""
    t1_price = float(pred_prices[0])
    max_pred_price = float(max(pred_prices))
    expected_profit_amount = max(0.0, max_pred_price - current_price)
    expected_profit_pct = (expected_profit_amount / current_price) * 100
    next_price_change_pct = ((t1_price - current_price) / current_price) * 100
    stop_loss_price = current_price * 0.98

    if t1_price < current_price:
        buy_lower, buy_upper = t1_price * 0.99, t1_price * 1.01
    else:
        buy_lower, buy_upper = current_price * 0.995, current_price * 1.015

    if final_score >= 1.5:
        if rsi_val <= 35 or bb_pos <= 0.15:
            buy_now_conclusion = "Nên Mua Ngay"
            buy_now_subtext = "Giá vàng đang ở vùng chiết khấu sâu; các chỉ báo kỹ thuật cho thấy khả năng hồi phục."
        else:
            buy_now_conclusion = "Có Thể Giải Ngân"
            buy_now_subtext = "Xu hướng tăng giá vàng đang hình thành; nên chia nhỏ vốn giải ngân."
    elif final_score <= -1.5:
        buy_now_conclusion = "Tuyệt Đối Không Mua"
        buy_now_subtext = "Áp lực bán và xu hướng giảm đang chiếm ưu thế; nên đứng ngoài quan sát."
    elif next_price_change_pct > 0.05:
        buy_now_conclusion = "Chờ Nhịp Rung Lắc"
        buy_now_subtext = "Giá vàng có xu hướng phục hồi nhưng nên chờ một nhịp điều chỉnh để có điểm mua an toàn."
    else:
        buy_now_conclusion = "Không Nên Mua Ngay"
        buy_now_subtext = "Động lực tăng giá vàng chưa rõ ràng; nên quan sát thêm."

    if final_score <= -1.5:
        best_buy_time = "Chờ tạo đáy"
        best_buy_time_reason = "Mô hình dự báo xu hướng giảm tiếp diễn và chưa có tín hiệu dừng rơi rõ ràng."
    elif t1_price > current_price and next_price_change_pct > 0.05:
        best_buy_time = "Phiên Á (9:00 - 11:30)"
        best_buy_time_reason = "Biến động thường chậm hơn, phù hợp để giải ngân từng phần trước phiên Âu-Mỹ."
    elif t1_price < current_price:
        best_buy_time = "Phiên Mỹ (20:00 - 22:00)"
        best_buy_time_reason = "Có thể chờ nhịp điều chỉnh sau tin tức quan trọng rồi đánh giá lại vùng hỗ trợ."
    else:
        best_buy_time = "Khung giờ trưa (11:30 - 13:30)"
        best_buy_time_reason = "Xu hướng đi ngang; nên giải ngân thận trọng khi biên độ biến động hẹp."

    return {
        "buy_now_conclusion": buy_now_conclusion,
        "buy_now_subtext": buy_now_subtext,
        "best_buy_time": best_buy_time,
        "best_buy_time_reason": best_buy_time_reason,
        "target_buy_range": f"${buy_lower:,.2f} - ${buy_upper:,.2f}",
        "expected_profit_pct": float(round(expected_profit_pct, 2)),
        "expected_profit_amount": float(round(expected_profit_amount, 2)),
        "next_price": float(round(t1_price, 2)),
        "next_price_change_pct": float(round(next_price_change_pct, 2)),
        "stop_loss_price": float(round(stop_loss_price, 2)),
    }
def _build_model_metrics(pred_prices, slope, r2, current_price, base_score, reasons, model_name, rsi_val, bb_pos):
    """
    Builds the model-specific result dictionary.
    Adjusts score contribution from ML trend using the given slope and r2.
    """
    # The dashboard's direction and recommendation are explicitly T+1.  Keep
    # the five-session move as separate metadata so a T+5 rebound cannot make
    # a falling T+1 card appear as an "up" recommendation.
    t1_change = float(pred_prices[0] - current_price)
    t1_pct_change = float((t1_change / (current_price + 1e-10)) * 100)
    horizon_change = float(pred_prices[-1] - current_price)
    horizon_pct_change = float((horizon_change / (current_price + 1e-10)) * 100)

    # Keep the score model-specific even when every raw R² is below zero.
    # The old max(0, R²) collapsed all weak models to exactly the same score
    # and made switching cards appear broken.  Map the useful range
    # [-0.25, 1.0] continuously, then retain only a small 10% forecast voice
    # at the weakest end.  This exposes genuine differences without turning
    # an unproven model into a strong buy/sell recommendation.
    raw_r2 = float(r2) if np.isfinite(r2) else -1.0
    validated_r2 = max(0.0, min(1.0, raw_r2))
    oos_score_factor = max(0.0, min(1.0, (raw_r2 + 0.25) / 1.25))
    forecast_evidence = 0.10 + 0.90 * oos_score_factor
    forecast_direction_strength = float(np.tanh(t1_pct_change / 0.25))
    ml_score_contribution = 1.25 * forecast_direction_strength * forecast_evidence

    # Low OOS quality must also temper the technical recommendation. This
    # prevents a weak model plus one threshold-based indicator from producing
    # an unjustified strong buy/sell label.
    confidence_scale = 0.55 + 0.45 * oos_score_factor
    technical_score_contribution = float(base_score * confidence_scale)
    final_score = max(-5.0, min(5.0, technical_score_contribution + ml_score_contribution))
    
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
    
    direction_threshold = current_price * 0.0002
    if t1_change > direction_threshold:
        direction = "Tăng"
        direction_class = "up"
    elif t1_change < -direction_threshold:
        direction = "Giảm"
        direction_class = "down"
    else:
        direction = "Đi ngang"
        direction_class = "sideways"
    
    model_label = {
        "random_forest": "Random Forest (Học máy nâng cao)",
        "linear_regression": "Hồi quy tuyến tính (Linear Regression)",
        "ensemble": "Ensemble Hybrid (Tổ hợp 6 mô hình)",
        "mlp_neural_network": "Mạng Nơ-ron MLP (Deep Learning)",
        "xgboost": "XGBoost Regressor (Học máy nâng cao)",
        "lstm": "LSTM (Chuỗi thời gian)",
        "linear_regression_fallback": "Hồi quy tuyến tính (Fallback)",
        "rf_fallback_for_xgb_random_forest": "XGBoost (Lỗi hệ thống - Tự động chuyển RF)",
        "rf_fallback_for_xgb_linear_regression_fallback": "XGBoost (Lỗi hệ thống - Tự động chuyển LR)",
        "lstm_damped_low_confidence": "LSTM (Giảm biên độ do OOS thấp)",
        "cnn_1d": "CNN 1D (Chuỗi thời gian)",
        "cnn_damped_low_confidence": "CNN 1D (Giảm biên độ do OOS thấp)",
        "linear_regression_fallback_for_cnn": "CNN 1D (Fallback hồi quy tuyến tính)"
    }.get(model_name, model_name)
    
    advice = calculate_actionable_advice(pred_prices, current_price, final_score, rsi_val, bb_pos, slope)

    # Keep the path explanation explicit for the UI.  A T+2 increase can be
    # explained relative to both today's price and the T+1 forecast, rather
    # than repeating the same T+1 explanation on every card.
    horizon_explanations = []
    for index, forecast_price in enumerate(pred_prices):
        horizon = index + 1
        forecast_price = float(forecast_price)
        reference_price = float(current_price if index == 0 else pred_prices[index - 1])
        step_change = forecast_price - reference_price
        step_pct_change = (step_change / (reference_price + 1e-10)) * 100
        cumulative_change = forecast_price - float(current_price)
        cumulative_pct_change = (cumulative_change / (float(current_price) + 1e-10)) * 100
        step_threshold = float(current_price) * 0.0002
        if step_change > step_threshold:
            step_direction = "Tăng tiếp"
        elif step_change < -step_threshold:
            step_direction = "Giảm tiếp"
        else:
            step_direction = "Đi ngang"
        if cumulative_change > step_threshold:
            direction_from_current = "Tăng"
        elif cumulative_change < -step_threshold:
            direction_from_current = "Giảm"
        else:
            direction_from_current = "Đi ngang"
        horizon_explanations.append({
            "horizon": f"T+{horizon}",
            "price": forecast_price,
            "reference_price": reference_price,
            "reference_label": "giá hiện tại" if index == 0 else f"giá dự báo T+{horizon - 1}",
            "step_change": float(step_change),
            "step_pct_change": float(step_pct_change),
            "step_direction": step_direction,
            "change_from_current": float(cumulative_change),
            "pct_change_from_current": float(cumulative_pct_change),
            "direction_from_current": direction_from_current,
        })
    
    ml_prediction = {
        "predicted_prices": [float(p) for p in pred_prices],
        "slope": float(slope),
        "r_squared": float(r2),
        "metric_type": "chronological_holdout_r2",
        "predicted_change": t1_change,
        "predicted_pct_change": t1_pct_change,
        "t1_predicted_change": t1_change,
        "t1_predicted_pct_change": t1_pct_change,
        "horizon_predicted_change": horizon_change,
        "horizon_predicted_pct_change": horizon_pct_change,
        "model_name": model_label,
        "horizon_explanations": horizon_explanations,
    }
    return {
        "score": round(final_score, 2),
        "score_breakdown": {
            "technical": round(technical_score_contribution, 4),
            "model_forecast": round(float(ml_score_contribution), 4),
            "raw_r_squared_oos": raw_r2,
            "oos_score_factor": round(float(oos_score_factor), 4),
        },
        "recommendation": recommendation,
        "action_class": action_class,
        "direction": direction,
        "direction_class": direction_class,
        "actionable_advice": advice,
        "ml_prediction": ml_prediction
    }


def analyze_and_recommend(
    df,
    model_weights=None,
    params=None,
    biases=None,
    production_metrics=None,
    conflict_events=None,
    market_signals=None,
    fed_policy_outlook=None,
    progress_callback=None,
):
    """
    Analyzes technical indicators and machine learning trend to output
    a buying recommendation, price direction prediction, and score.
    Returns predictions from six base models plus their weighted ensemble.
    """
    def report_progress(stage, progress):
        if callable(progress_callback):
            try:
                progress_callback(stage, progress)
            except Exception as progress_err:
                print(f"Gold model progress callback failed: {progress_err}")

    if model_weights is None:
        model_weights = {
            "random_forest": 1 / 6,
            "linear_regression": 1 / 6,
            "mlp": 1 / 6,
            "xgboost": 1 / 6,
            "lstm": 1 / 6,
            "cnn": 1 / 6,
        }
    
    # Extract weights
    w_rf   = model_weights.get("random_forest", 1 / 6)
    w_lr   = model_weights.get("linear_regression", 1 / 6)
    w_mlp  = model_weights.get("mlp", 1 / 6)
    w_xgb  = model_weights.get("xgboost", 1 / 6)
    w_lstm = model_weights.get("lstm", 1 / 6)
    w_cnn  = model_weights.get("cnn", 1 / 6)
    
    # Normalize weights to sum to 1.0
    total_w = w_rf + w_lr + w_mlp + w_xgb + w_lstm + w_cnn
    if total_w > 0:
        w_rf   /= total_w
        w_lr   /= total_w
        w_mlp  /= total_w
        w_xgb  /= total_w
        w_lstm /= total_w
        w_cnn  /= total_w
    else:
        w_rf = w_lr = w_mlp = w_xgb = w_lstm = w_cnn = 1 / 6

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

    profit_taking_risk = _estimate_profit_taking_risk(df_ind, market_signals)
    direction_model = _predict_direction_and_regime(
        df_ind,
        profit_taking_risk=profit_taking_risk,
    )
    market_context = _summarize_market_context(
        market_signals,
        tech_score,
        profit_taking_risk=profit_taking_risk,
        fed_policy_outlook=fed_policy_outlook,
    )
    if profit_taking_risk["level"] in {"medium", "high"}:
        reasons.append(
            f"Rủi ro chốt lời {profit_taking_risk['label'].lower()} "
            f"({profit_taking_risk['score_percent']:.0f}/100): "
            + ", ".join(profit_taking_risk["drivers"][:3])
        )
    if direction_model.get("eligible"):
        direction_label = "tăng" if direction_model["probability_up"] >= 0.5 else "giảm"
        reasons.append(
            f"Bộ phân loại hướng xác nhận thiên về {direction_label} "
            f"(P tăng {direction_model['probability_up'] * 100:.0f}%, "
            f"holdout {direction_model['holdout_accuracy'] * 100:.0f}% so với "
            f"baseline {direction_model['majority_baseline'] * 100:.0f}%)"
        )
    if market_context["label"] == "bearish":
        reasons.append(
            "Tín hiệu FED/lạm phát/vĩ mô đã công bố tạo áp lực giảm ngắn hạn "
            f"(điểm ngữ cảnh {market_context['score']:.2f})"
        )
    elif market_context["label"] == "bullish":
        reasons.append(
            "Tin vĩ mô đã công bố hỗ trợ giá vàng ngắn hạn "
            f"(điểm ngữ cảnh +{market_context['score']:.2f})"
        )

    # ─────────────────────────────────────────────────────────────────
    # STEP 2: Get predictions from all six base models
    # ─────────────────────────────────────────────────────────────────
    days_to_predict = 5
    report_progress("Tính tín hiệu kỹ thuật và vĩ mô", 47)
    
    # Model A: Linear Regression
    report_progress("Đang huấn luyện mô hình Hồi quy tuyến tính", 49)
    lr_prices, lr_slope, lr_r2 = predict_future_prices(
        df_ind,
        days_to_predict=days_to_predict,
        params=params.get("linear_regression") if params else None,
    )
    report_progress("Đã hoàn tất mô hình Hồi quy tuyến tính", 52)

    # Model B: Random Forest
    report_progress("Đang huấn luyện mô hình Random Forest", 54)
    rf_result = predict_future_prices_rf(df_ind, days_to_predict=days_to_predict, params=params.get("random_forest") if params else None)
    rf_prices, rf_slope, rf_r2 = rf_result[0], rf_result[1], rf_result[2]
    rf_model_used = rf_result[3]
    report_progress("Đã hoàn tất mô hình Random Forest", 63)

    # Model C: MLP Neural Network
    report_progress("Đang huấn luyện mô hình MLP", 65)
    mlp_result = predict_future_prices_mlp(df_ind, days_to_predict=days_to_predict, params=params.get("mlp") if params else None)
    mlp_prices, mlp_slope, mlp_r2 = mlp_result[0], mlp_result[1], mlp_result[2]
    mlp_model_used = mlp_result[3]
    report_progress("Đã hoàn tất mô hình MLP", 74)

    # Model D: XGBoost Regressor
    report_progress("Đang huấn luyện mô hình XGBoost", 76)
    xgb_result = predict_future_prices_xgb(df_ind, days_to_predict=days_to_predict, params=params.get("xgboost") if params else None)
    xgb_prices, xgb_slope, xgb_r2 = xgb_result[0], xgb_result[1], xgb_result[2]
    xgb_model_used = xgb_result[3]
    report_progress("Đã hoàn tất mô hình XGBoost", 84)

    # Model E: LSTM Sequence Model
    report_progress("Đang huấn luyện mô hình LSTM", 86)
    lstm_result = predict_future_prices_lstm(df_ind, days_to_predict=days_to_predict)
    lstm_prices, lstm_slope, lstm_r2 = lstm_result[0], lstm_result[1], lstm_result[2]
    lstm_model_used = lstm_result[3]
    report_progress("Đã hoàn tất mô hình LSTM", 89)

    # Model F: 1D CNN over the recent temporal window
    report_progress("Đang huấn luyện mô hình CNN 1D", 90)
    cnn_result = predict_future_prices_cnn(
        df_ind,
        days_to_predict=days_to_predict,
    )
    cnn_prices, cnn_slope, cnn_r2 = cnn_result[0], cnn_result[1], cnn_result[2]
    cnn_model_used = cnn_result[3]
    report_progress("Đã hoàn tất mô hình CNN 1D", 92)

    daily_volatility = _robust_daily_volatility(df_ind)

    # Apply rolling bias correction if available
    if biases:
        # Price-level feedback can become enormous after a regime jump.  Cap
        # it by robust volatility before applying it to the new forecast.
        max_bias = current_price * float(np.clip(2.0 * daily_volatility, 0.005, 0.025))
        rf_prices = rf_prices + np.clip(float(biases.get("random_forest", 0.0)), -max_bias, max_bias)
        lr_prices = lr_prices + np.clip(float(biases.get("linear_regression", 0.0)), -max_bias, max_bias)
        mlp_prices = mlp_prices + np.clip(float(biases.get("mlp", 0.0)), -max_bias, max_bias)
        xgb_prices = xgb_prices + np.clip(float(biases.get("xgboost", 0.0)), -max_bias, max_bias)
        lstm_prices = lstm_prices + np.clip(float(biases.get("lstm", 0.0)), -max_bias, max_bias)
        cnn_prices = cnn_prices + np.clip(float(biases.get("cnn", 0.0)), -max_bias, max_bias)

    # Keep every model inside a volatility-aware range before combining them.
    rf_prices = _stabilize_price_path(rf_prices, current_price, daily_volatility)
    lr_prices = _stabilize_price_path(lr_prices, current_price, daily_volatility)
    mlp_prices = _stabilize_price_path(mlp_prices, current_price, daily_volatility)
    xgb_prices = _stabilize_price_path(xgb_prices, current_price, daily_volatility)
    lstm_prices = _stabilize_price_path(lstm_prices, current_price, daily_volatility)
    cnn_prices = _stabilize_price_path(cnn_prices, current_price, daily_volatility)

    def path_slope(prices):
        return float(prices[-1] - current_price) / max(len(prices), 1)

    rf_slope = path_slope(rf_prices)
    lr_slope = path_slope(lr_prices)
    mlp_slope = path_slope(mlp_prices)
    xgb_slope = path_slope(xgb_prices)
    lstm_slope = path_slope(lstm_prices)
    cnn_slope = path_slope(cnn_prices)

    # Reweight on every run using current chronological holdout quality, while
    # retaining a smaller prior from resolved production predictions.
    validation_r2 = {
        "random_forest": rf_r2,
        "linear_regression": lr_r2,
        "mlp": mlp_r2,
        "xgboost": xgb_r2,
        "lstm": lstm_r2,
        "cnn": cnn_r2,
    }
    model_implementations = {
        "random_forest": rf_model_used,
        "linear_regression": "linear_regression",
        "mlp": mlp_model_used,
        "xgboost": xgb_model_used,
        "lstm": lstm_model_used,
        "cnn": cnn_model_used,
    }
    adaptive_weights = _adaptive_ensemble_weights(model_weights, validation_r2, model_implementations)
    w_rf = adaptive_weights["random_forest"]
    w_lr = adaptive_weights["linear_regression"]
    w_mlp = adaptive_weights["mlp"]
    w_xgb = adaptive_weights["xgboost"]
    w_lstm = adaptive_weights["lstm"]
    w_cnn = adaptive_weights["cnn"]

    # Model F: adaptive Ensemble.  When models disagree, shrink the magnitude
    # toward the current price instead of presenting a high-conviction move.
    weighted_paths = [
        (w_rf, rf_prices),
        (w_lr, lr_prices),
        (w_mlp, mlp_prices),
        (w_xgb, xgb_prices),
        (w_lstm, lstm_prices),
        (w_cnn, cnn_prices),
    ]
    holdout_oos_quality = sum(
        adaptive_weights[key] * max(0.0, min(1.0, float(validation_r2[key])))
        for key in adaptive_weights
    )
    production_oos_quality = sum(
        adaptive_weights[key] * max(
            0.0,
            min(1.0, float(((production_metrics or {}).get(key) or {}).get("reliability", 0.0))),
        )
        for key in adaptive_weights
    )
    has_versioned_production_evidence = any(
        int(((production_metrics or {}).get(key) or {}).get("samples", 0)) >= 3
        for key in adaptive_weights
    )
    # Current holdout remains the primary evidence. Resolved production
    # forecasts add a smaller independent signal (direction + MAE versus the
    # persistence baseline) so reliability is not reduced to R² alone.
    direction_oos_quality = (
        max(0.0, float(direction_model.get("skill_vs_majority", 0.0)))
        * float(direction_model.get("confidence", 0.0))
        if direction_model.get("eligible") else 0.0
    )
    positive_oos_quality = float(
        0.65 * holdout_oos_quality
        + 0.25 * production_oos_quality
        + 0.10 * direction_oos_quality
    )
    confidence_label = (
        "low" if positive_oos_quality < 0.15
        else "medium" if positive_oos_quality < 0.45
        else "high"
    )
    context_score = float(market_context.get("score", 0.0))
    # Cap the deterministic PCE/news adjustment below one normal daily move.
    # It should correct a blind spot, not replace the statistical models.
    context_return_cap = float(min(0.75 * daily_volatility, 0.008))
    context_strength = float(0.35 + 0.65 * (1.0 - np.clip(positive_oos_quality, 0.0, 1.0)))
    direction_gate_weight = (
        min(0.20, 0.05 + 0.40 * max(0.0, direction_model["skill_vs_majority"]))
        if direction_model.get("eligible") else 0.0
    )
    direction_gate_return = (
        (2.0 * float(direction_model.get("probability_up", 0.5)) - 1.0)
        * float(direction_model.get("expected_move", 0.0))
        * direction_gate_weight
    )

    # Apply the released FED/macro/news context once to every base-model
    # path. The previous implementation applied it only inside Ensemble,
    # which made the UI compare context-free model cards against a
    # context-aware Ensemble. The overlay is deliberately separate from the
    # learned return so its contribution remains visible and auditable.
    context_overlay_returns = []
    classifier_overlay_returns = []

    def apply_context_overlay(prices):
        adjusted = []
        for horizon, price in enumerate(prices):
            context_component = (
                context_score
                * context_return_cap
                * context_strength
                / np.sqrt(horizon + 1.0)
            )
            classifier_component = direction_gate_return / np.sqrt(horizon + 1.0)
            if len(context_overlay_returns) <= horizon:
                context_overlay_returns.append(float(context_component))
                classifier_overlay_returns.append(float(classifier_component))
            base_return = float(price / (current_price + 1e-10) - 1.0)
            adjusted.append(current_price * (1.0 + base_return + context_component + classifier_component))
        return _stabilize_price_path(np.asarray(adjusted, dtype=float), current_price, daily_volatility)

    rf_prices = apply_context_overlay(rf_prices)
    lr_prices = apply_context_overlay(lr_prices)
    mlp_prices = apply_context_overlay(mlp_prices)
    xgb_prices = apply_context_overlay(xgb_prices)
    lstm_prices = apply_context_overlay(lstm_prices)
    cnn_prices = apply_context_overlay(cnn_prices)

    # The arrays above are newly allocated by the overlay/stabilizer. Refresh
    # the voting paths so Ensemble aggregates exactly the same context-aware
    # forecasts shown for each member in the UI.
    weighted_paths = [
        (w_rf, rf_prices),
        (w_lr, lr_prices),
        (w_mlp, mlp_prices),
        (w_xgb, xgb_prices),
        (w_lstm, lstm_prices),
        (w_cnn, cnn_prices),
    ]

    ens_prices = []
    direction_consensus = []
    signed_direction_consensus = []
    direction_guard_applied = []
    prediction_intervals = []
    for horizon in range(days_to_predict):
        returns = np.asarray([
            float(prices[horizon] / (current_price + 1e-10) - 1.0)
            for _, prices in weighted_paths
        ])
        weights_array = np.asarray([weight for weight, _ in weighted_paths])
        signed_vote = float(np.sum(weights_array * np.sign(returns)))
        raw_return = float(np.sum(weights_array * returns))
        consensus = float(abs(signed_vote))
        # Persistence (tomorrow ~= today) is a hard baseline for liquid prices.
        # If OOS quality is weak, only accept a fraction of the model move. As
        # quality and direction consensus improve, the forecast can express a
        # larger move instead of being permanently over-conservative.
        # When every holdout R² is negative, the learned move should be close
        # to persistence instead of retaining half of an unreliable signal.
        quality_multiplier = 0.05 + 0.95 * np.sqrt(positive_oos_quality)
        consensus_multiplier = 0.30 + 0.70 * consensus
        model_return = raw_return * quality_multiplier * consensus_multiplier
        # Context has already been applied consistently to each component
        # path above; do not add it a second time to Ensemble.
        shrunk_return = model_return

        # A context/news adjustment may refine a mixed forecast, but it must
        # not silently reverse a unanimous model direction. Previously a
        # small positive weighted model return could be overwhelmed by a
        # bearish context score, making Ensemble show "Giảm" while every
        # component model showed "Tăng". Keep the forecast conservative by
        # never crossing through zero when at least 80% of the adaptive vote
        # agrees on one direction.
        guard_applied = False
        if signed_vote >= 0.80 and shrunk_return < 0.0:
            shrunk_return = max(0.0, model_return)
            guard_applied = True
        elif signed_vote <= -0.80 and shrunk_return > 0.0:
            shrunk_return = min(0.0, model_return)
            guard_applied = True
        ens_price = float(current_price * (1.0 + shrunk_return))
        ens_prices.append(ens_price)
        direction_consensus.append(consensus)
        signed_direction_consensus.append(signed_vote)
        direction_guard_applied.append(guard_applied)

        component_prices = np.asarray([prices[horizon] for _, prices in weighted_paths], dtype=float)
        model_dispersion = float(np.sqrt(np.sum(weights_array * np.square(component_prices - ens_price))))
        volatility_error = float(current_price * daily_volatility * np.sqrt(horizon + 1))
        interval_half_width = 1.282 * np.sqrt(model_dispersion ** 2 + volatility_error ** 2)
        prediction_intervals.append({
            "lower": float(max(0.0, ens_price - interval_half_width)),
            "upper": float(ens_price + interval_half_width),
            "confidence_level": 0.80,
        })

    ens_prices = np.asarray(ens_prices, dtype=float)
    ens_slope = path_slope(ens_prices)
    ens_r2 = float(np.clip(positive_oos_quality * np.mean(direction_consensus), 0.0, 1.0))

    # ─────────────────────────────────────────────────────────────────
    # STEP 3: Build per-model result dictionaries
    # ─────────────────────────────────────────────────────────────────
    bb_pos_val = float(latest["bb_position"]) if pd.notna(latest.get("bb_position")) else 0.5
    rsi_val_float = float(rsi_val) if pd.notna(rsi_val) else 50.0
    
    lr_metrics   = _build_model_metrics(lr_prices,   lr_slope,   lr_r2,   current_price, tech_score, reasons, "linear_regression", rsi_val_float, bb_pos_val)
    rf_metrics   = _build_model_metrics(rf_prices,   rf_slope,   rf_r2,   current_price, tech_score, reasons, rf_model_used,        rsi_val_float, bb_pos_val)
    mlp_metrics  = _build_model_metrics(mlp_prices,  mlp_slope,  mlp_r2,  current_price, tech_score, reasons, mlp_model_used,       rsi_val_float, bb_pos_val)
    xgb_metrics  = _build_model_metrics(xgb_prices,  xgb_slope,  xgb_r2,  current_price, tech_score, reasons, xgb_model_used,       rsi_val_float, bb_pos_val)
    lstm_metrics = _build_model_metrics(lstm_prices, lstm_slope, lstm_r2,  current_price, tech_score, reasons, lstm_model_used,      rsi_val_float, bb_pos_val)
    cnn_metrics  = _build_model_metrics(cnn_prices,  cnn_slope,  cnn_r2,   current_price, tech_score, reasons, cnn_model_used,       rsi_val_float, bb_pos_val)
    ens_metrics  = _build_model_metrics(ens_prices,  ens_slope,  ens_r2,  current_price, tech_score, reasons, "ensemble",           rsi_val_float, bb_pos_val)

    for metrics in (lr_metrics, rf_metrics, mlp_metrics, xgb_metrics, lstm_metrics, cnn_metrics):
        metrics["ml_prediction"]["context_overlay_returns"] = context_overlay_returns
        metrics["ml_prediction"]["classifier_overlay_returns"] = classifier_overlay_returns
        metrics["ml_prediction"]["context_overlay_scope"] = "toàn bộ 6 mô hình + Ensemble"
        metrics["ml_prediction"]["market_context"] = market_context
        metrics["ml_prediction"]["profit_taking_risk"] = profit_taking_risk

    ens_metrics["ml_prediction"]["metric_type"] = "conservative_oos_ensemble_score"
    ens_metrics["ml_prediction"]["prediction_intervals"] = prediction_intervals
    ens_metrics["ml_prediction"]["direction_consensus"] = direction_consensus
    ens_metrics["ml_prediction"]["signed_direction_consensus"] = signed_direction_consensus
    ens_metrics["ml_prediction"]["direction_guard_applied"] = direction_guard_applied
    ens_metrics["ml_prediction"]["context_overlay_returns"] = context_overlay_returns
    ens_metrics["ml_prediction"]["classifier_overlay_returns"] = classifier_overlay_returns
    ens_metrics["ml_prediction"]["context_overlay_scope"] = "toàn bộ 6 mô hình + Ensemble"
    ens_metrics["ml_prediction"]["daily_volatility"] = daily_volatility
    ens_metrics["ml_prediction"]["oos_reliability"] = float(positive_oos_quality)
    ens_metrics["ml_prediction"]["confidence_label"] = confidence_label
    ens_metrics["ml_prediction"]["market_context"] = market_context
    ens_metrics["ml_prediction"]["profit_taking_risk"] = profit_taking_risk
    ens_metrics["ml_prediction"]["direction_model"] = direction_model
    
    # Append model-specific ML reason to each model's reasons
    def t1_direction(prices):
        change = float(prices[0] - current_price)
        if abs(change) <= current_price * 0.0002:
            return "Đi ngang"
        return "Tăng" if change > 0 else "Giảm"

    lr_reason   = f"Hồi quy tuyến tính dự báo T+1 {t1_direction(lr_prices)} (R² ngoài mẫu: {lr_r2:.2f})"
    rf_reason   = f"{'Random Forest' if rf_model_used == 'random_forest' else 'Mô hình fallback'} dự báo T+1 {t1_direction(rf_prices)} (R² ngoài mẫu: {rf_r2:.2f})"
    mlp_reason  = f"{'Mạng Nơ-ron MLP' if mlp_model_used == 'mlp_neural_network' else 'Mô hình fallback'} dự báo T+1 {t1_direction(mlp_prices)} (R² ngoài mẫu: {mlp_r2:.2f})"
    xgb_reason  = f"{'XGBoost Regressor' if xgb_model_used == 'xgboost' else 'Mô hình fallback'} dự báo T+1 {t1_direction(xgb_prices)} (R² ngoài mẫu: {xgb_r2:.2f})"
    lstm_reason_prefix = "LSTM giảm biên độ do OOS thấp" if lstm_model_used == "lstm_damped_low_confidence" else "LSTM"
    lstm_reason = f"{lstm_reason_prefix} dự báo T+1 {t1_direction(lstm_prices)} (R² ngoài mẫu: {lstm_r2:.2f})"
    cnn_reason_prefix = "CNN 1D giảm biên độ do OOS thấp" if cnn_model_used == "cnn_damped_low_confidence" else ("CNN 1D fallback" if "fallback" in cnn_model_used else "CNN 1D")
    cnn_reason = f"{cnn_reason_prefix} dự báo T+1 {t1_direction(cnn_prices)} (R² ngoài mẫu: {cnn_r2:.2f})"
    ens_reason  = (
        f"Ensemble thích nghi dự báo T+1 {t1_direction(ens_prices)}; "
        f"đồng thuận hướng {direction_consensus[0] * 100:.0f}%, "
        f"điểm OOS bảo thủ {ens_r2:.2f}, ngữ cảnh {market_context['label']}"
    )
    if direction_guard_applied and direction_guard_applied[0]:
        ens_reason += "; đã khóa không đảo chiều vì đồng thuận các mô hình T+1"

    # Each price model also casts a policy vote. This is not a separate FED
    # neural network: it is an auditable inference from the shared policy
    # nowcast (CPI/PCE/labour/rates/FOMC) adjusted by that model's own gold
    # path. Keeping this distinction explicit prevents the UI from claiming
    # that a price model directly observes the FED's private decision process.
    fed_action_labels = {"hike": "Tăng lãi suất", "hold": "Giữ nguyên", "cut": "Giảm lãi suất"}
    fed_nowcast = fed_policy_outlook or {}
    fed_action_nowcast = fed_nowcast.get("fed_action") or {}
    source_fed_probabilities = fed_action_nowcast.get("probabilities") or {}
    base_fed_probabilities = {}
    for action_key in ("hike", "hold", "cut"):
        try:
            base_fed_probabilities[action_key] = max(0.0, float(source_fed_probabilities.get(action_key, 0.0) or 0.0))
        except (TypeError, ValueError):
            base_fed_probabilities[action_key] = 0.0
    if sum(base_fed_probabilities.values()) <= 0:
        base_fed_probabilities = {"hike": 0.10, "hold": 0.80, "cut": 0.10}
    base_fed_total = sum(base_fed_probabilities.values())
    base_fed_probabilities = {
        key: float(value / base_fed_total) for key, value in base_fed_probabilities.items()
    }

    def build_model_fed_vote(model_label, prices):
        first_return = float((float(prices[0]) - float(current_price)) / (float(current_price) + 1e-10))
        gold_signal = float(np.clip(first_return / 0.003, -1.0, 1.0))
        probabilities = dict(base_fed_probabilities)
        # A model that sees gold falling gives a small hawkish tilt; a model
        # that sees gold rising gives a small dovish tilt. The macro nowcast
        # remains dominant, so one price model cannot overpower the data.
        tilt = float(np.clip(-gold_signal * 0.045, -0.045, 0.045))
        if tilt > 0:
            probabilities["hike"] += tilt
            probabilities["hold"] -= tilt * 0.65
            probabilities["cut"] -= tilt * 0.35
        elif tilt < 0:
            probabilities["cut"] += abs(tilt)
            probabilities["hold"] -= abs(tilt) * 0.65
            probabilities["hike"] -= abs(tilt) * 0.35
        probabilities = {key: max(0.0, value) for key, value in probabilities.items()}
        total = sum(probabilities.values()) or 1.0
        probabilities = {key: float(value / total) for key, value in probabilities.items()}
        action_key = max(probabilities, key=probabilities.get)
        policy_score = float(fed_action_nowcast.get("score", 0.0) or 0.0)
        if policy_score >= 0.12:
            bias = "hawkish"
        elif policy_score <= -0.12:
            bias = "dovish"
        else:
            bias = "neutral"
        source_gold_score = float((fed_nowcast.get("gold_implication") or {}).get("score", 0.0) or 0.0)
        gold_score = source_gold_score
        if action_key == "hike":
            gold_score -= 0.08
        elif action_key == "cut":
            gold_score += 0.08
        gold_implication = "up" if gold_score >= 0.12 else "down" if gold_score <= -0.12 else "sideways"
        confidence = "high" if max(probabilities.values()) >= 0.70 else "medium" if max(probabilities.values()) >= 0.52 else "low"
        return {
            "action": action_key,
            "action_key": action_key,
            "action_label": fed_action_labels[action_key],
            "probabilities": probabilities,
            "bias": bias,
            "confidence": confidence,
            "gold_implication": gold_implication,
            "gold_implication_label": {"up": "Tăng", "down": "Giảm", "sideways": "Đi ngang"}[gold_implication],
            "gold_signal_pct": first_return * 100.0,
            "reason": (
                f"{model_label} suy ra FED {fed_action_labels[action_key].lower()} với xác suất "
                f"{probabilities[action_key] * 100:.0f}% từ policy nowcast và đường giá vàng riêng; "
                f"model dự báo vàng T+1 {first_return * 100:+.2f}%. Đây là phiếu suy luận, không phải output trực tiếp của FED."
            ),
            "method": "policy nowcast + đường giá riêng của model",
        }

    roundtable_sources = [
        ("random_forest", rf_metrics, rf_prices, w_rf, rf_reason),
        ("linear_regression", lr_metrics, lr_prices, w_lr, lr_reason),
        ("mlp", mlp_metrics, mlp_prices, w_mlp, mlp_reason),
        ("xgboost", xgb_metrics, xgb_prices, w_xgb, xgb_reason),
        ("lstm", lstm_metrics, lstm_prices, w_lstm, lstm_reason),
        ("cnn", cnn_metrics, cnn_prices, w_cnn, cnn_reason),
    ]
    roundtable_members = []
    direction_threshold = float(current_price) * 0.0002
    for model_key, metrics, prices, weight, model_reason in roundtable_sources:
        model_label = metrics["ml_prediction"].get("model_name", model_key)
        model_fed_vote = build_model_fed_vote(model_label, prices)
        horizon_details = metrics["ml_prediction"].get("horizon_explanations") or []
        member_horizons = []
        for horizon_index, forecast_price in enumerate(prices):
            forecast_price = float(forecast_price)
            change = forecast_price - float(current_price)
            if change > direction_threshold:
                vote = "up"
                vote_label = "Tăng"
            elif change < -direction_threshold:
                vote = "down"
                vote_label = "Giảm"
            else:
                vote = "sideways"
                vote_label = "Đi ngang"
            detail = horizon_details[horizon_index] if horizon_index < len(horizon_details) else {}
            step_direction = detail.get("step_direction", vote_label)
            step_pct_change = float(detail.get("step_pct_change", 0.0) or 0.0)
            horizon_reason = (
                f"{metrics['ml_prediction'].get('model_name', model_key)} dự báo T+{horizon_index + 1} "
                f"{vote_label.lower()} {change / (float(current_price) + 1e-10) * 100.0:+.2f}% so với hiện tại; "
                f"so với phiên dự báo liền trước: {step_direction.lower()} {step_pct_change:+.2f}%."
            )
            member_horizons.append({
                "horizon": horizon_index + 1,
                "vote": vote,
                "vote_label": vote_label,
                "price": forecast_price,
                "change": float(change),
                "pct_change": float(change / (float(current_price) + 1e-10) * 100.0),
                "step_direction": step_direction,
                "step_pct_change": step_pct_change,
                "reason": horizon_reason,
            })
        roundtable_members.append({
            "model": model_key,
            "label": model_label,
            "weight": float(weight),
            "r_squared": float(metrics["ml_prediction"].get("r_squared", 0.0)),
            "reason": model_reason,
            "evidence": [
                model_reason,
                *list(reasons[:3]),
                *list(market_context.get("drivers") or [])[:2],
            ],
            "counter_argument": (
                f"R² OOS {float(metrics['ml_prediction'].get('r_squared', 0.0)):.3f}; "
                "nếu âm hoặc gần 0 thì hướng vẫn chỉ là tín hiệu yếu. Các model dùng chung dữ liệu "
                "kỹ thuật/vĩ mô nên đồng thuận không tương đương bằng chứng độc lập."
            ),
            "fed_vote": model_fed_vote,
            "horizons": member_horizons,
        })

    fed_weighted_probabilities = {
        action_key: float(sum(
            float(member.get("weight", 0.0) or 0.0)
            * float((member.get("fed_vote") or {}).get("probabilities", {}).get(action_key, 0.0) or 0.0)
            for member in roundtable_members
        ))
        for action_key in ("hike", "hold", "cut")
    }
    fed_primary_votes = {action_key: 0 for action_key in ("hike", "hold", "cut")}
    for member in roundtable_members:
        action_key = (member.get("fed_vote") or {}).get("action")
        if action_key in fed_primary_votes:
            fed_primary_votes[action_key] += 1
    fed_action_key = max(fed_weighted_probabilities, key=fed_weighted_probabilities.get)
    fed_policy_score = float(fed_action_nowcast.get("score", 0.0) or 0.0)
    fed_bias = "hawkish" if fed_policy_score >= 0.12 else "dovish" if fed_policy_score <= -0.12 else "neutral"
    fallback_gold_score = float((fed_nowcast.get("gold_implication") or {}).get("score", 0.0) or 0.0)
    if fed_action_key == "hike":
        fallback_gold_score -= 0.08
    elif fed_action_key == "cut":
        fallback_gold_score += 0.08
    fallback_gold_implication = "up" if fallback_gold_score >= 0.12 else "down" if fallback_gold_score <= -0.12 else "sideways"
    fed_forecast = {
        "action": fed_action_key,
        "action_key": fed_action_key,
        "action_label": fed_action_labels[fed_action_key],
        "probabilities": fed_weighted_probabilities,
        "bias": fed_bias,
        "confidence": (
            "high" if fed_weighted_probabilities[fed_action_key] >= 0.70
            else "medium" if fed_weighted_probabilities[fed_action_key] >= 0.52
            else "low"
        ),
        "gold_implication": fallback_gold_implication,
        "gold_implication_label": {"up": "Tăng", "down": "Giảm", "sideways": "Đi ngang"}[fallback_gold_implication],
        "gold_implication_score": fallback_gold_score,
        "next_meeting": fed_nowcast.get("next_fed_event"),
        "conclusion": (
            f"Hội nghị nghiêng về FED {fed_action_labels[fed_action_key].lower()} ở kỳ họp kế tiếp; "
            f"hệ quả ngắn hạn lên vàng: {({'up': 'tăng', 'down': 'giảm', 'sideways': 'đi ngang'}[fallback_gold_implication])}."
        ),
        "key_arguments": [
            f"Phiếu hành động FED: {fed_primary_votes['hike']} Tăng, {fed_primary_votes['hold']} Giữ, {fed_primary_votes['cut']} Giảm.",
            f"Xác suất gộp theo trọng số model: Tăng {fed_weighted_probabilities['hike'] * 100:.1f}%, Giữ {fed_weighted_probabilities['hold'] * 100:.1f}%, Giảm {fed_weighted_probabilities['cut'] * 100:.1f}%.",
            *list(fed_nowcast.get("drivers") or [])[:3],
        ],
        "risk_warning": "Đây là nowcast từ dữ liệu công khai và phiếu mô hình, không phải cam kết hay thông báo chính thức của FED.",
        "source": "Ensemble policy vote trước khi Codex phản biện",
    }
    fed_vote_summary = {
        "weighted_probabilities": fed_weighted_probabilities,
        "primary_action": fed_action_key,
        "primary_action_label": fed_action_labels[fed_action_key],
        "vote_counts": fed_primary_votes,
        "bias": fed_bias,
        "next_meeting": fed_nowcast.get("next_fed_event"),
    }

    horizon_votes = []
    decisions = []
    direction_confidence_scores = []
    direction_confidence_labels = []
    ensemble_horizon_details = ens_metrics["ml_prediction"].get("horizon_explanations") or []
    for horizon_index in range(days_to_predict):
        vote_tally = {"up": 0.0, "down": 0.0, "sideways": 0.0}
        vote_counts = {"up": 0, "down": 0, "sideways": 0}
        for member in roundtable_members:
            vote = member["horizons"][horizon_index]["vote"]
            vote_tally[vote] += float(member["weight"])
            vote_counts[vote] += 1

        ensemble_price = float(ens_prices[horizon_index])
        ensemble_change = ensemble_price - float(current_price)
        if ensemble_change > direction_threshold:
            final_vote, final_label = "up", "Tăng"
        elif ensemble_change < -direction_threshold:
            final_vote, final_label = "down", "Giảm"
        else:
            final_vote, final_label = "sideways", "Đi ngang"
        context_pct = float(
            ((context_overlay_returns[horizon_index] if horizon_index < len(context_overlay_returns) else 0.0)
             + (classifier_overlay_returns[horizon_index] if horizon_index < len(classifier_overlay_returns) else 0.0))
            * 100.0
        )
        consensus_pct = float(direction_consensus[horizon_index] * 100.0)
        # Direction and price-level accuracy answer different questions. A
        # unanimous vote that agrees with independently derived macro context
        # can support a medium-confidence direction even while price R² stays
        # weak. Correlated model votes are capped at 35%, and a high label is
        # impossible without meaningful OOS/classifier evidence.
        context_direction = "up" if context_pct > 0.02 else ("down" if context_pct < -0.02 else "sideways")
        context_alignment = 1.0 if context_direction == final_vote and final_vote != "sideways" else 0.0
        context_direction_quality = context_alignment * min(1.0, abs(context_score))
        classifier_direction_quality = (
            max(0.0, float(direction_model.get("skill_vs_majority", 0.0)))
            * float(direction_model.get("confidence", 0.0))
            if direction_model.get("eligible") else 0.0
        )
        direction_confidence_score = float(np.clip(
            0.35 * float(direction_consensus[horizon_index])
            + 0.25 * context_direction_quality
            + 0.25 * positive_oos_quality
            + 0.15 * classifier_direction_quality,
            0.0,
            1.0,
        ))
        direction_confidence_label = (
            "low" if direction_confidence_score < 0.40
            else "medium" if direction_confidence_score < 0.70
            else "high"
        )
        if direction_confidence_label == "high" and positive_oos_quality < 0.15:
            direction_confidence_label = "medium"
        direction_confidence_scores.append(direction_confidence_score)
        direction_confidence_labels.append(direction_confidence_label)
        detail = ensemble_horizon_details[horizon_index] if horizon_index < len(ensemble_horizon_details) else {}
        final_arguments = [
            f"Phiếu thành viên: {vote_counts['up']} Tăng, {vote_counts['down']} Giảm, {vote_counts['sideways']} Đi ngang.",
            f"Trọng số OOS: Tăng {vote_tally['up'] * 100:.1f}%, Giảm {vote_tally['down'] * 100:.1f}%, Đi ngang {vote_tally['sideways'] * 100:.1f}%.",
            f"Mức đồng thuận có trọng số {consensus_pct:.1f}%.",
            f"Context FED, dữ liệu kinh tế, tin tức và địa chính trị tác động {context_pct:+.2f}% ở T+{horizon_index + 1}.",
            f"Tin cậy hướng {direction_confidence_score * 100:.1f}% ({direction_confidence_label}); "
            f"tin cậy mức giá OOS {ens_r2 * 100:.1f}% ({confidence_label}).",
        ]
        if detail:
            final_arguments.append(
                f"So với {detail.get('reference_label', 'phiên trước')}: "
                f"{str(detail.get('step_direction', final_label)).lower()} "
                f"{float(detail.get('step_pct_change', 0.0) or 0.0):+.2f}%."
            )
        final_arguments.extend(list(market_context.get("drivers") or [])[:3])
        final_arguments.extend(reasons[:3])
        horizon_votes.append({
            "horizon": horizon_index + 1,
            "weighted_tally": {key: float(value) for key, value in vote_tally.items()},
            "vote_counts": vote_counts,
            "consensus": float(direction_consensus[horizon_index]),
            "signed_consensus": float(signed_direction_consensus[horizon_index]),
        })
        decisions.append({
            "horizon": horizon_index + 1,
            "vote": final_vote,
            "label": final_label,
            "price": ensemble_price,
            "change": float(ensemble_change),
            "pct_change": float(ensemble_change / (current_price + 1e-10) * 100.0),
            "confidence": direction_confidence_label,
            "direction_confidence_score": direction_confidence_score,
            "price_confidence": confidence_label,
            "price_oos_score": float(ens_r2),
            "oos_score": float(ens_r2),
            "context_overlay_pct": context_pct,
            "direction_guard_applied": bool(direction_guard_applied[horizon_index]),
            "reason": (
                f"Ensemble quyết định T+{horizon_index + 1} {final_label}; "
                f"đồng thuận {consensus_pct:.0f}%, context {context_pct:+.2f}%."
            ),
            "arguments": final_arguments,
        })

    roundtable = {
        "title": "Hội nghị bàn tròn dự báo giá vàng",
        "chair": "ensemble",
        "quorum": len(roundtable_members),
        "members": roundtable_members,
        "fed_policy_outlook": fed_nowcast,
        "fed_vote_summary": fed_vote_summary,
        "fed_forecast": fed_forecast,
        "horizon_votes": horizon_votes,
        "weighted_tally": horizon_votes[0]["weighted_tally"],
        "vote_counts": horizon_votes[0]["vote_counts"],
        "consensus": horizon_votes[0]["consensus"],
        "direction_confidence_scores": direction_confidence_scores,
        "direction_confidence_labels": direction_confidence_labels,
        "price_confidence": confidence_label,
        "context": {
            "label": market_context.get("label", "neutral"),
            "score": float(market_context.get("score", 0.0)),
            "overlay_pct_by_horizon": [
                float((context_value + classifier_value) * 100.0)
                for context_value, classifier_value in zip(
                    context_overlay_returns,
                    classifier_overlay_returns,
                )
            ],
            "drivers": list(market_context.get("drivers") or [])[:4],
        },
        "decisions": decisions,
        "decision": decisions[0],
        "rule": "Phiếu có trọng số OOS + context FED/tin tức dùng chung; Ensemble không cộng context lần hai.",
    }

    # ─────────────────────────────────────────────────────────────────
    # STEP 4: Use the adaptive Ensemble as the one canonical top-level response
    # ─────────────────────────────────────────────────────────────────
    default = ens_metrics
    default_reasons = reasons + [ens_reason]
    
    analysis = {
        "status": "success",
        "asset": "gold",
        "model_version": GOLD_MODEL_VERSION,
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
        # Top-level and the default UI now use the same Ensemble forecast.
        "ml_prediction": default["ml_prediction"],
        "score": default["score"],
        "recommendation": default["recommendation"],
        "action_class": default["action_class"],
        "direction": default["direction"],
        "direction_class": default["direction_class"],
        "actionable_advice": default["actionable_advice"],
        "reasons": default_reasons,
        "profit_taking_risk": profit_taking_risk,
        "direction_model": direction_model,
        "ensemble_diagnostics": {
            "model_version": GOLD_MODEL_VERSION,
            "weight_source": (
                "35% lịch sử cùng phiên bản + 65% chất lượng holdout hiện tại"
                if has_versioned_production_evidence
                else "Chưa đủ kết quả cùng phiên bản; prior đều + chất lượng holdout hiện tại"
            ),
            "production_evidence_available": has_versioned_production_evidence,
            "weights": adaptive_weights,
            "validation_r2": validation_r2,
            "model_implementations": model_implementations,
            "direction_consensus": direction_consensus,
            "direction_confidence_scores": direction_confidence_scores,
            "direction_confidence_labels": direction_confidence_labels,
            "signed_direction_consensus": signed_direction_consensus,
            "direction_guard_applied": direction_guard_applied,
            "context_overlay_returns": context_overlay_returns,
            "classifier_overlay_returns": classifier_overlay_returns,
            "context_overlay_scope": "toàn bộ 6 mô hình + Ensemble",
            "daily_volatility": daily_volatility,
            "prediction_intervals": prediction_intervals,
            "oos_reliability": float(positive_oos_quality),
            "holdout_oos_quality": float(holdout_oos_quality),
            "production_oos_quality": float(production_oos_quality),
            "direction_oos_quality": float(direction_oos_quality),
            "production_metrics": production_metrics or {},
            "confidence_label": confidence_label,
            "market_context": market_context,
            "profit_taking_risk": profit_taking_risk,
            "direction_model": direction_model,
            "direction_gate_weight": float(direction_gate_weight),
            "roundtable": roundtable,
        },
        "roundtable": roundtable,
        # All model results for frontend switching
        "models": {
            "random_forest": {
                **rf_metrics,
                "weight": w_rf,
                "reasons": [rf_reason] + reasons
            },
            "linear_regression": {
                **lr_metrics,
                "weight": w_lr,
                "reasons": [lr_reason] + reasons
            },
            "mlp": {
                **mlp_metrics,
                "weight": w_mlp,
                "reasons": [mlp_reason] + reasons
            },
            "xgboost": {
                **xgb_metrics,
                "weight": w_xgb,
                "reasons": [xgb_reason] + reasons
            },
            "lstm": {
                **lstm_metrics,
                "weight": w_lstm,
                "reasons": [lstm_reason] + reasons
            },
            "cnn": {
                **cnn_metrics,
                "weight": w_cnn,
                "reasons": [cnn_reason] + reasons
            },
            "ensemble": {
                **ens_metrics,
                "weight": 1.0,
                "reasons": [ens_reason] + reasons
            }
        },
        # Send historical data for plotting in frontend (limit to last 60 days to keep JSON small)
        "history": df_ind.tail(60)[["date", "open", "high", "low", "close", "volume", "rsi", "macd", "macd_signal"]].to_dict(orient="records"),
        "backtest": None
    }
    report_progress("Đang tổng hợp kết quả các mô hình", 93)
    
    # Calculate backtest (previous prediction vs actual last row)
    if len(df) >= 21:
        try:
            df_prev = df.iloc[:-1].copy()
            df_prev_ind = calculate_technical_indicators(df_prev)
            
            # Predict T+1 from df_prev_ind
            lr_p_back, _, _ = predict_future_prices(
                df_prev_ind,
                days_to_predict=1,
                params=params.get("linear_regression") if params else None,
            )
            
            rf_res_back = predict_future_prices_rf(
                df_prev_ind,
                days_to_predict=1,
                params=params.get("random_forest") if params else None,
            )
            rf_p_back = rf_res_back[0]
            
            mlp_res_back = predict_future_prices_mlp(
                df_prev_ind,
                days_to_predict=1,
                params=params.get("mlp") if params else None,
            )
            mlp_p_back = mlp_res_back[0]
            
            xgb_res_back = predict_future_prices_xgb(
                df_prev_ind,
                days_to_predict=1,
                params=params.get("xgboost") if params else None,
            )
            xgb_p_back = xgb_res_back[0]

            lstm_res_back = predict_future_prices_lstm(df_prev_ind, days_to_predict=1)
            lstm_p_back = lstm_res_back[0]

            cnn_res_back = predict_future_prices_cnn(df_prev_ind, days_to_predict=1)
            cnn_p_back = cnn_res_back[0]
            
            ens_p_back = (
                rf_p_back[0] * w_rf
                + lr_p_back[0] * w_lr
                + mlp_p_back[0] * w_mlp
                + xgb_p_back[0] * w_xgb
                + lstm_p_back[0] * w_lstm
                + cnn_p_back[0] * w_cnn
            )
            
            actual_last = df.iloc[-1]
            actual_close = float(actual_last["close"])
            actual_date = actual_last["date"]
            
            rf_pred = float(rf_p_back[0])
            lr_pred = float(lr_p_back[0])
            mlp_pred = float(mlp_p_back[0])
            xgb_pred = float(xgb_p_back[0])
            lstm_pred = float(lstm_p_back[0])
            cnn_pred = float(cnn_p_back[0])
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
                    "lstm": {
                        "predicted_price": lstm_pred,
                        "error_pct": ((actual_close - lstm_pred) / (lstm_pred + 1e-10)) * 100
                    },
                    "cnn": {
                        "predicted_price": cnn_pred,
                        "error_pct": ((actual_close - cnn_pred) / (cnn_pred + 1e-10)) * 100
                    },
                    "ensemble": {
                        "predicted_price": ens_pred,
                        "error_pct": ((actual_close - ens_pred) / (ens_pred + 1e-10)) * 100
                    }
                }
            }
        except Exception as e:
            print(f"Error calculating backtest: {e}")

    report_progress("Đã hoàn tất dự báo và kiểm tra T+1", 97)
            
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


def optimize_hyperparameters(
    df,
    days_to_predict=5,
    incumbent_params=None,
    return_diagnostics=False,
):
    """Tune production-compatible parameters with expanding-window OOS folds.

    Candidates are scored against persistence (future return = 0), weighted
    toward T+1/T+2, and accepted only if they improve the incumbent. This
    avoids promoting a configuration that merely overfits one recent split.
    """
    from itertools import product
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    feature_cols = _gold_model_feature_columns(df)
    work = _prepare_gold_model_features(df, feature_cols).dropna(subset=["close"]).tail(360)
    if len(work) < 80:
        empty = {"status": "insufficient_data", "samples": int(len(work))}
        return ({}, empty) if return_diagnostics else {}

    horizon_count = int(np.clip(days_to_predict, 1, 5))
    sample_count = len(work) - horizon_count
    X = work.iloc[:sample_count][feature_cols].to_numpy(dtype=float)
    close = work["close"].to_numpy(dtype=float)
    y = np.column_stack([
        close[horizon:horizon + sample_count] / np.maximum(close[:sample_count], 1e-10) - 1.0
        for horizon in range(1, horizon_count + 1)
    ])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    splits = list(TimeSeriesSplit(n_splits=4).split(X))
    horizon_weights = np.asarray([0.35, 0.25, 0.18, 0.13, 0.09][:horizon_count], dtype=float)
    horizon_weights /= horizon_weights.sum()

    def fold_skill(actual, predicted):
        skills = []
        direction_accuracy = []
        for index in range(horizon_count):
            truth = actual[:, index]
            estimate = predicted[:, index]
            model_mse = float(np.mean(np.square(truth - estimate)))
            persistence_mse = float(np.mean(np.square(truth)))
            skill = 1.0 - model_mse / max(persistence_mse, 1e-12)
            skills.append(float(np.clip(skill, -3.0, 1.0)))
            direction_accuracy.append(float(np.mean(np.sign(estimate) == np.sign(truth))))
        persistence_skill = float(np.sum(horizon_weights * np.asarray(skills)))
        direction_skill = float(np.sum(horizon_weights * (2.0 * np.asarray(direction_accuracy) - 1.0)))
        return 0.85 * persistence_skill + 0.15 * direction_skill

    def score_candidate(model_name, params):
        fold_scores = []
        for train_idx, valid_idx in splits:
            train_y = _robust_clip_targets(y[train_idx])
            if model_name in {"mlp", "linear_regression"}:
                scaler = StandardScaler()
                train_X = scaler.fit_transform(X[train_idx])
                valid_X = scaler.transform(X[valid_idx])
                if model_name == "mlp":
                    model = MLPRegressor(
                        hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
                        alpha=float(params["alpha"]),
                        activation="tanh",
                        solver="lbfgs",
                        max_iter=400,
                        random_state=42,
                    )
                    model.fit(train_X, train_y[:, 0] if horizon_count == 1 else train_y)
                    predicted = np.asarray(model.predict(valid_X), dtype=float).reshape(len(valid_idx), horizon_count)
                else:
                    predictions = []
                    for horizon_index in range(horizon_count):
                        model = Ridge(alpha=float(params["alpha"]))
                        model.fit(train_X, train_y[:, horizon_index])
                        predictions.append(model.predict(valid_X))
                    predicted = np.column_stack(predictions)
            else:
                predictions = []
                for horizon_index in range(horizon_count):
                    if model_name == "random_forest":
                        model = RandomForestRegressor(
                            n_estimators=int(params["n_estimators"]),
                            max_depth=int(params["max_depth"]),
                            min_samples_leaf=int(params["min_samples_leaf"]),
                            max_features=params["max_features"],
                            random_state=42,
                            n_jobs=1,
                        )
                    else:
                        model = xgb.XGBRegressor(
                            n_estimators=int(params["n_estimators"]),
                            max_depth=int(params["max_depth"]),
                            learning_rate=float(params["learning_rate"]),
                            subsample=0.7,
                            colsample_bytree=0.7,
                            min_child_weight=float(params["min_child_weight"]),
                            gamma=0.001,
                            reg_alpha=1.0,
                            reg_lambda=float(params["reg_lambda"]),
                            objective="reg:absoluteerror",
                            random_state=42,
                            n_jobs=1,
                            verbosity=0,
                        )
                    model.fit(X[train_idx], train_y[:, horizon_index])
                    predictions.append(model.predict(X[valid_idx]))
                predicted = np.column_stack(predictions)
            fold_scores.append(fold_skill(y[valid_idx], predicted))
        return (
            float(np.mean(fold_scores)),
            float(np.median(fold_scores)),
            [float(value) for value in fold_scores],
        )

    defaults = {
        "linear_regression": {"alpha": 1000.0},
        "random_forest": {"n_estimators": 300, "max_depth": 3, "min_samples_leaf": 10, "max_features": "sqrt"},
        "xgboost": {"n_estimators": 80, "max_depth": 1, "learning_rate": 0.02, "min_child_weight": 15, "reg_lambda": 30.0},
        "mlp": {"hidden_layer_sizes": [8, 4], "alpha": 5.0},
    }
    incumbent_params = incumbent_params or {}
    candidate_sets = {
        "linear_regression": [
            {"alpha": alpha} for alpha in (100.0, 300.0, 1000.0, 3000.0, 10000.0)
        ],
        "random_forest": [
            {"n_estimators": 300, "max_depth": depth, "min_samples_leaf": leaf, "max_features": feature_mode}
            for depth, leaf, feature_mode in product((2, 3, 4), (8, 12, 18), ("sqrt", 0.5))
        ],
        "xgboost": [
            {"n_estimators": 80, "max_depth": depth, "learning_rate": rate, "min_child_weight": child, "reg_lambda": 30.0}
            for depth, rate, child in product((1, 2), (0.01, 0.03), (10, 20, 30))
        ],
        "mlp": [
            {"hidden_layer_sizes": list(size), "alpha": alpha}
            for size, alpha in product(((4,), (8, 4), (16, 8)), (5.0, 10.0, 20.0))
        ],
    }

    def normalize_production_params(model_name, raw_params):
        merged = {**defaults[model_name], **(raw_params or {})}
        if model_name == "random_forest":
            feature_mode = merged.get("max_features", "sqrt")
            if feature_mode != "sqrt":
                feature_mode = float(np.clip(float(feature_mode), 0.1, 1.0))
            return {
                "n_estimators": int(np.clip(merged["n_estimators"], 120, 500)),
                "max_depth": int(np.clip(merged["max_depth"], 2, 4)),
                "min_samples_leaf": max(8, int(merged["min_samples_leaf"])),
                "max_features": feature_mode,
            }
        if model_name == "xgboost":
            return {
                "n_estimators": int(np.clip(merged["n_estimators"], 50, 180)),
                "max_depth": int(np.clip(merged["max_depth"], 1, 2)),
                "learning_rate": float(np.clip(merged["learning_rate"], 0.01, 0.03)),
                "min_child_weight": float(np.clip(merged["min_child_weight"], 8, 30)),
                "reg_lambda": float(np.clip(merged["reg_lambda"], 10, 80)),
            }
        if model_name == "linear_regression":
            return {"alpha": float(np.clip(merged["alpha"], 100.0, 10000.0))}
        sizes = merged.get("hidden_layer_sizes", [8, 4])
        if isinstance(sizes, (int, float)):
            sizes = [int(sizes)]
        return {
            "hidden_layer_sizes": [max(2, min(16, int(size))) for size in list(sizes)[:2]],
            "alpha": max(5.0, float(merged.get("alpha", 5.0))),
        }

    selected = {}
    diagnostics = {
        "status": "success",
        "samples": int(len(X)),
        "horizons": horizon_count,
        "folds": len(splits),
        "objective": "85% persistence-skill + 15% direction-skill; weighted T+1..T+5",
        "models": {},
    }
    for model_name, candidates in candidate_sets.items():
        incumbent = normalize_production_params(model_name, incumbent_params.get(model_name))
        if incumbent not in candidates:
            candidates = [incumbent] + candidates
        incumbent_mean, incumbent_median, incumbent_folds = score_candidate(model_name, incumbent)
        best_config = incumbent
        best_mean = incumbent_mean
        best_median = incumbent_median
        best_folds = incumbent_folds
        for candidate in candidates:
            mean_score, median_score, candidate_folds = score_candidate(model_name, candidate)
            if (mean_score, median_score) > (best_mean, best_median):
                best_config = candidate
                best_mean = mean_score
                best_median = median_score
                best_folds = candidate_folds
        accepted = bool(best_mean >= incumbent_mean + 0.005 and best_median >= incumbent_median - 0.01)
        selected[model_name] = best_config if accepted else incumbent
        diagnostics["models"][model_name] = {
            "accepted": accepted,
            "incumbent_score": float(incumbent_mean),
            "best_score": float(best_mean),
            "improvement": float(best_mean - incumbent_mean),
            "best_median_fold_score": float(best_median),
            "fold_scores": best_folds,
            "selected_params": selected[model_name],
        }

    return (selected, diagnostics) if return_diagnostics else selected
