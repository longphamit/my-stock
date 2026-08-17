import requests
import json
import warnings
from datetime import datetime
from pymongo import MongoClient
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import time

# Ignore MLP convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

try:
    import xgboost as xgb
except ImportError:
    xgb = None

import os
# Initialize MongoDB connection
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client["stock_analytics"]

PRODUCTS = {
    "mega645": "power645",
    "power655": "power655",
    "power535": "power535"
}

# Generic draw prediction function
def predict_draw_game(draws, min_num, max_num, num_per_draw, result_extractor, training_draws=200):
    # draws are sorted descending (newest first)
    if len(draws) < 120:
        return None

    # Pre-calculate appearance indices
    all_numbers = list(range(min_num, max_num + 1))
    appearance_indices = {n: [] for n in all_numbers}
    for idx, d in enumerate(draws):
        res = result_extractor(d)
        for num in res:
            if num in appearance_indices:
                appearance_indices[num].append(idx)

    # Pre-calculate sets for fast lookup
    appearance_indices_set = {n: set(appearance_indices[n]) for n in all_numbers}
    
    split_num = (max_num + min_num) // 2
    PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79}
    
    # Compute global co-occurrence matrix for features (using draws from index 1 onwards to avoid leaking the current/next target draw)
    C_feat = {n: {m: 0 for m in all_numbers} for n in all_numbers}
    for d in draws[1:]:
        res = result_extractor(d)
        for a in res:
            if a in C_feat:
                for b in res:
                    if b in C_feat and a != b:
                        C_feat[a][b] += 1

    # 1. Feature extraction helper
    def get_features_labels(start_t, end_t):
        X, y = [], []
        for t in range(start_t, end_t):
            if t >= len(draws):
                break
            target_res = set(result_extractor(draws[t]))
            
            # Pre-compute frequency arrays for ranking at step t
            f30_all = {x: sum(1 for idx in appearance_indices[x] if t < idx <= t + 30) for x in all_numbers}
            f10_all = {x: sum(1 for idx in appearance_indices[x] if t < idx <= t + 10) for x in all_numbers}
            sorted_by_f30 = sorted(all_numbers, key=lambda x: f30_all[x], reverse=True)
            sorted_by_f10 = sorted(all_numbers, key=lambda x: f10_all[x], reverse=True)
            
            prev_draw = result_extractor(draws[t+1]) if t+1 < len(draws) else []
            
            for n in all_numbers:
                indices_past = [idx for idx in appearance_indices[n] if idx > t]
                
                if indices_past:
                    last_seen = indices_past[0] - t - 1
                else:
                    last_seen = len(draws) - t
                    
                f100 = sum(1 for idx in indices_past if idx <= t + 100)
                f50 = sum(1 for idx in indices_past if idx <= t + 50)
                f30 = f30_all[n]
                f10 = f10_all[n]
                f5 = sum(1 for idx in indices_past if idx <= t + 5)
                
                is_even = 1 if n % 2 == 0 else 0
                is_small = 1 if n <= split_num else 0
                tail_digit = n % 10
                head_digit = n // 10
                digit_sum = tail_digit + head_digit
                is_prime = 1 if n in PRIMES else 0
                
                # Cyclic average gap calculation
                gaps = [indices_past[i+1] - indices_past[i] for i in range(len(indices_past) - 1)]
                avg_gap = float(np.mean(gaps)) if len(gaps) > 0 else float(max_num / num_per_draw)
                gap_ratio = last_seen / (avg_gap + 1)
                
                # Streak calculation
                streak = 0
                while (t + 1 + streak) in appearance_indices_set[n]:
                    streak += 1
                    
                # Relations to previous draw (T-1)
                min_dist_to_prev = min(abs(n - p) for p in prev_draw) if prev_draw else int(max_num - min_num)
                prev_co_occur_sum = sum(C_feat[n].get(p, 0) for p in prev_draw) if prev_draw else 0
                prev_co_occur_max = max(C_feat[n].get(p, 0) for p in prev_draw) if prev_draw else 0
                
                # Popularity ranks
                rank_30 = sorted_by_f30.index(n) + 1
                rank_10 = sorted_by_f10.index(n) + 1
                
                is_hot = 1 if f10 >= 2 else 0
                is_cold = 1 if last_seen > 10 else 0
                
                X.append([
                    last_seen, f100, f50, f30, f10, f5, 
                    is_even, is_small, tail_digit, head_digit, digit_sum, is_prime,
                    avg_gap, gap_ratio, streak,
                    min_dist_to_prev, prev_co_occur_sum, prev_co_occur_max,
                    rank_30, rank_10, is_hot, is_cold
                ])
                y.append(1 if n in target_res else 0)
        return np.array(X), np.array(y)

    def get_features_pred(t_target):
        X = []
        # Pre-compute frequency arrays for ranking at step t_target
        f30_all = {x: sum(1 for idx in appearance_indices[x] if t_target < idx <= t_target + 30) for x in all_numbers}
        f10_all = {x: sum(1 for idx in appearance_indices[x] if t_target < idx <= t_target + 10) for x in all_numbers}
        sorted_by_f30 = sorted(all_numbers, key=lambda x: f30_all[x], reverse=True)
        sorted_by_f10 = sorted(all_numbers, key=lambda x: f10_all[x], reverse=True)
        
        prev_draw = result_extractor(draws[t_target+1]) if t_target+1 < len(draws) else []
        
        for n in all_numbers:
            indices_past = [idx for idx in appearance_indices[n] if idx > t_target]
            
            if indices_past:
                last_seen = indices_past[0] - t_target - 1
            else:
                last_seen = len(draws) - t_target
                
            f100 = sum(1 for idx in indices_past if idx <= t_target + 100)
            f50 = sum(1 for idx in indices_past if idx <= t_target + 50)
            f30 = f30_all[n]
            f10 = f10_all[n]
            f5 = sum(1 for idx in indices_past if idx <= t_target + 5)
            
            is_even = 1 if n % 2 == 0 else 0
            is_small = 1 if n <= split_num else 0
            tail_digit = n % 10
            head_digit = n // 10
            digit_sum = tail_digit + head_digit
            is_prime = 1 if n in PRIMES else 0
            
            gaps = [indices_past[i+1] - indices_past[i] for i in range(len(indices_past) - 1)]
            avg_gap = float(np.mean(gaps)) if len(gaps) > 0 else float(max_num / num_per_draw)
            gap_ratio = last_seen / (avg_gap + 1)
            
            streak = 0
            while (t_target + 1 + streak) in appearance_indices_set[n]:
                streak += 1
                
            min_dist_to_prev = min(abs(n - p) for p in prev_draw) if prev_draw else int(max_num - min_num)
            prev_co_occur_sum = sum(C_feat[n].get(p, 0) for p in prev_draw) if prev_draw else 0
            prev_co_occur_max = max(C_feat[n].get(p, 0) for p in prev_draw) if prev_draw else 0
            
            rank_30 = sorted_by_f30.index(n) + 1
            rank_10 = sorted_by_f10.index(n) + 1
            
            is_hot = 1 if f10 >= 2 else 0
            is_cold = 1 if last_seen > 10 else 0
            
            X.append([
                last_seen, f100, f50, f30, f10, f5, 
                is_even, is_small, tail_digit, head_digit, digit_sum, is_prime,
                avg_gap, gap_ratio, streak,
                min_dist_to_prev, prev_co_occur_sum, prev_co_occur_max,
                rank_30, rank_10, is_hot, is_cold
            ])
        return np.array(X)

    # Train on historical draws (use min of training_draws and len(draws) - 25 to protect backtesting bounds)
    effective_train_draws = min(training_draws, len(draws) - 25)
    X_train, y_train = get_features_labels(1, effective_train_draws + 1)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Models list (enhanced parameters with class weight balancing)
    models = {
        "linear_regression": LogisticRegression(C=0.1, class_weight='balanced', random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=50, max_depth=6, min_samples_split=4, class_weight='balanced', random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=50, max_depth=4, learning_rate=0.04, random_state=42),
        "mlp": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=150, early_stopping=True, random_state=42)
    }
    if xgb:
        models["xgboost"] = xgb.XGBClassifier(
            n_estimators=50, 
            max_depth=4, 
            learning_rate=0.04, 
            scale_pos_weight=float((max_num - num_per_draw) / num_per_draw), 
            random_state=42, 
            eval_metric='logloss'
        )

    # Fit models
    if len(np.unique(y_train)) < 2:
        # Fallback to dummy
        class DummyModel:
            def fit(self, X, y): pass
            def predict(self, X): return np.zeros(len(X))
            def predict_proba(self, X): return np.column_stack([np.ones(len(X)), np.zeros(len(X))])
        for name in models:
            models[name] = DummyModel()
    else:
        # Compute class weights for MLP fit sample weights (balanced class representation)
        neg_count = sum(1 for y in y_train if y == 0)
        pos_count = sum(1 for y in y_train if y == 1)
        w_neg = len(y_train) / (2.0 * neg_count) if neg_count > 0 else 1.0
        w_pos = len(y_train) / (2.0 * pos_count) if pos_count > 0 else 1.0
        sample_weight = np.array([w_pos if y == 1 else w_neg for y in y_train])
        
        for name, model in models.items():
            if name == "mlp":
                model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
            else:
                model.fit(X_train_scaled, y_train)

    # Helper to get lucky numbers and probabilities for a target draw
    def get_model_predictions(t_target):
        X_pred = get_features_pred(t_target)
        X_pred_scaled = scaler.transform(X_pred)
        
        preds = {}
        for name, model in models.items():
            probs = model.predict_proba(X_pred_scaled)[:, 1]
            
            # Map number to probability
            num_probs = []
            for idx, n in enumerate(all_numbers):
                num_probs.append({
                    "number": n,
                    "prob": float(np.clip(probs[idx], 0, 1))
                })
            
            preds[name] = num_probs
            
        # Add weighted ensemble Hybrid: RF (25%), XGB (25% if exists, else add to RF/GB/MLP), GB (25%), MLP (15%), LR (10%)
        active_models = [m for m in preds if m != "ensemble"]
        weights = {}
        if "xgboost" in active_models:
            weights = {
                "random_forest": 0.25, 
                "xgboost": 0.25, 
                "gradient_boosting": 0.25, 
                "mlp": 0.15, 
                "linear_regression": 0.10
            }
        else:
            weights = {
                "random_forest": 0.35, 
                "gradient_boosting": 0.35, 
                "mlp": 0.20, 
                "linear_regression": 0.10
            }
            
        ens_probs = np.zeros(len(all_numbers))
        for m in active_models:
            w = weights.get(m, 1.0 / len(active_models))
            model_probs = np.array([x["prob"] for x in preds[m]])
            ens_probs += w * model_probs
            
        preds["ensemble"] = [{"number": n, "prob": float(ens_probs[idx])} for idx, n in enumerate(all_numbers)]
        
        return preds

    # Predict next draw (t_target = -1)
    next_preds = get_model_predictions(-1)
    
    # Predict latest draw for backtest (t_target = 0)
    back_preds = get_model_predictions(0)
    
    return {
        "next": next_preds,
        "back": back_preds
    }

def get_all_3d_strings(draw):
    # Collect all 3-digit strings in this draw
    result = draw.get("result", {})
    if not isinstance(result, dict):
        return []
    draw_3d_strings = []
    for prize, values in result.items():
        if isinstance(values, list):
            for val in values:
                if isinstance(val, str) and len(val) == 3 and val.isdigit():
                    draw_3d_strings.append(val)
    return draw_3d_strings

def predict_draw_profile(draws, num_per_draw, split_num):
    from sklearn.ensemble import RandomForestRegressor
    
    if len(draws) < 100:
        return {
            "pred_odd": num_per_draw // 2,
            "pred_even": num_per_draw - (num_per_draw // 2),
            "pred_small": num_per_draw // 2,
            "pred_large": num_per_draw - (num_per_draw // 2),
            "pred_sum": int(num_per_draw * split_num)
        }
        
    X_odd, y_odd = [], []
    X_small, y_small = [], []
    X_sum, y_sum = [], []
    
    for t in range(len(draws) - 6, -1, -1):
        feats_odd = []
        feats_small = []
        feats_sum = []
        for i in range(1, 6):
            res_i = draws[t + i].get("result", [])[:num_per_draw]
            odds_i = sum(1 for n in res_i if n % 2 == 1)
            smalls_i = sum(1 for n in res_i if n <= split_num)
            sum_i = sum(res_i)
            feats_odd.append(odds_i)
            feats_small.append(smalls_i)
            feats_sum.append(sum_i)
            
        X_odd.append(feats_odd)
        X_small.append(feats_small)
        X_sum.append(feats_sum)
        
        res_t = draws[t].get("result", [])[:num_per_draw]
        y_odd.append(sum(1 for n in res_t if n % 2 == 1))
        y_small.append(sum(1 for n in res_t if n <= split_num))
        y_sum.append(sum(res_t))
        
    clf_odd = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    clf_small = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    reg_sum = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
    
    clf_odd.fit(X_odd, y_odd)
    clf_small.fit(X_small, y_small)
    reg_sum.fit(X_sum, y_sum)
    
    next_feat_odd = []
    next_feat_small = []
    next_feat_sum = []
    for i in range(5):
        res_i = draws[i].get("result", [])[:num_per_draw]
        next_feat_odd.append(sum(1 for n in res_i if n % 2 == 1))
        next_feat_small.append(sum(1 for n in res_i if n <= split_num))
        next_feat_sum.append(sum(res_i))
        
    pred_odd = int(clf_odd.predict([next_feat_odd])[0])
    pred_small = int(clf_small.predict([next_feat_small])[0])
    pred_sum = float(reg_sum.predict([next_feat_sum])[0])
    
    # Calculate confidence probabilities
    try:
        pred_odd_idx = list(clf_odd.classes_).index(pred_odd)
        odd_conf = float(clf_odd.predict_proba([next_feat_odd])[0][pred_odd_idx])
    except Exception:
        odd_conf = 0.5
        
    try:
        pred_small_idx = list(clf_small.classes_).index(pred_small)
        small_conf = float(clf_small.predict_proba([next_feat_small])[0][pred_small_idx])
    except Exception:
        small_conf = 0.5
    
    return {
        "pred_odd": pred_odd,
        "pred_even": num_per_draw - pred_odd,
        "pred_small": pred_small,
        "pred_large": num_per_draw - pred_small,
        "pred_sum": round(pred_sum, 1),
        "odd_confidence": round(odd_conf * 100, 1),
        "small_confidence": round(small_conf * 100, 1)
    }

def run_monte_carlo_filter(product_id, ensemble_probabilities, profile, num_per_draw, max_num, split_num, draws):
    import random
    import math
    from itertools import combinations

    # Calculate co-occurrence matrix C over the last 300 draws
    C = {n: {other: 0 for other in range(1, max_num + 1) if other != n} for n in range(1, max_num + 1)}
    freq = {n: 0 for n in range(1, max_num + 1)}
    
    limit_co = min(300, len(draws))
    for idx in range(limit_co):
        res = draws[idx].get("result", [])[:num_per_draw]
        for n in res:
            if 1 <= n <= max_num:
                freq[n] += 1
        for i in range(len(res)):
            for j in range(i + 1, len(res)):
                a, b = res[i], res[j]
                if 1 <= a <= max_num and 1 <= b <= max_num and a != b:
                    C[a][b] += 1
                    C[b][a] += 1

    # Map number to ensemble probability
    ensemble_map = {x["number"]: x["prob"] for x in ensemble_probabilities}
    
    pred_odd = profile["pred_odd"]
    pred_small = profile["pred_small"]
    pred_sum = profile["pred_sum"]
    
    # Pre-calculate conditional probabilities matrix to speed up Monte Carlo loops 10x
    # P_cond[n_prev][x] = C[n_prev][x] / freq[n_prev]
    P_cond = {n: {other: 0.0 for other in range(1, max_num + 1)} for n in range(1, max_num + 1)}
    for n in range(1, max_num + 1):
        prev_freq = freq[n]
        if prev_freq > 0:
            for other in range(1, max_num + 1):
                if other != n:
                    P_cond[n][other] = C[n].get(other, 0) / float(prev_freq)
                    
    qualified = []
    seen_tickets = set()
    
    numbers_list = list(range(1, max_num + 1))
    # Exaggerate probability differences (temperature = 0.7) to favor high-probability numbers
    base_weights = [ensemble_map.get(n, 0.0) ** 1.5 for n in numbers_list]
    bw_sum = sum(base_weights)
    if bw_sum > 0:
        base_weights = [w / bw_sum for w in base_weights]
    else:
        base_weights = [1.0 / max_num] * max_num
        
    # Helper to run Monte Carlo sampling with soft profile constraints
    def generate_candidates(num_samples, max_consec_allowed):
        local_qualified = []
        for _ in range(num_samples):
            ticket = []
            available_nums = list(numbers_list)
            available_weights = list(base_weights)
            
            # Pick 1st number
            n1 = random.choices(available_nums, weights=available_weights, k=1)[0]
            ticket.append(n1)
            
            # Remove n1
            idx = available_nums.index(n1)
            available_nums.pop(idx)
            available_weights.pop(idx)
            
            # Pick remaining numbers guided by normalized conditional co-occurrence probability
            for _ in range(num_per_draw - 1):
                dyn_weights = []
                for x in available_nums:
                    # Optimized lookup using pre-calculated conditional probabilities matrix
                    co_factor = sum(P_cond[n_prev].get(x, 0.0) for n_prev in ticket) / len(ticket)
                    
                    # Boost factor based on co-occurrence probability (max boost of 4x)
                    boost = 1.0 + 3.0 * co_factor
                    idx_w = numbers_list.index(x)
                    dyn_weights.append(base_weights[idx_w] * boost)
                    
                dw_sum = sum(dyn_weights)
                if dw_sum > 0:
                    dyn_weights = [w / dw_sum for w in dyn_weights]
                else:
                    dyn_weights = [1.0 / len(available_nums)] * len(available_nums)
                    
                chosen = random.choices(available_nums, weights=dyn_weights, k=1)[0]
                ticket.append(chosen)
                
                # Remove chosen
                idx = available_nums.index(chosen)
                available_nums.pop(idx)
                
            ticket.sort()
            ticket_tuple = tuple(ticket)
            if ticket_tuple in seen_tickets:
                continue
                
            consec = 0
            for i in range(len(ticket) - 1):
                if ticket[i+1] - ticket[i] == 1:
                    consec += 1
            if consec > max_consec_allowed:
                continue
                
            # Score ticket: individual log probs + log Jaccard co-occurrence associations - profile_penalty
            score = 0.0
            for n in ticket:
                p_n = ensemble_map.get(n, 0.001)
                score += math.log(max(0.0001, p_n))
                
            # Pairwise Jaccard co-occurrence (normalized co-occurrence metric)
            pairwise = 0.0
            for a, b in combinations(ticket, 2):
                co_ab = C[a].get(b, 0)
                union_ab = freq[a] + freq[b] - co_ab
                jaccard_ab = co_ab / union_ab if union_ab > 0 else 0.0
                pairwise += math.log(1.0 + jaccard_ab)
                
            num_pairs = math.comb(num_per_draw, 2)
            score += pairwise / float(num_pairs) if num_pairs > 0 else pairwise
            
            # Soft constraints profile penalties (reduced penalty coefficients to avoid over-penalizing ML favorites)
            odds_t = sum(1 for n in ticket if n % 2 == 1)
            smalls_t = sum(1 for n in ticket if n <= split_num)
            profile_penalty = abs(odds_t - pred_odd) * 0.5 + abs(smalls_t - pred_small) * 0.5
            
            sum_t = sum(ticket)
            sum_dev = abs(sum_t - pred_sum)
            if sum_dev > 35:
                profile_penalty += (sum_dev - 35) * 0.02
                
            score -= profile_penalty
            
            local_qualified.append({
                "ticket": ticket,
                "score": score
            })
            seen_tickets.add(ticket_tuple)
            if len(local_qualified) >= 1000:
                break
        return local_qualified

    # Generate 40,000 samples and pick top candidates using soft constraints
    qualified = generate_candidates(num_samples=40000, max_consec_allowed=2)
    qualified.sort(key=lambda x: x["score"], reverse=True)
    
    # Deduplicate and apply diversity constraints (max overlap of 2 numbers to cover a wider space)
    unique_qualified = []
    seen_uniq = set()
    max_shared = 2
    
    for item in qualified:
        ticket = item["ticket"]
        t_tup = tuple(ticket)
        if t_tup not in seen_uniq:
            # Check overlap count with already selected tickets
            too_similar = False
            for prev_ticket in unique_qualified:
                shared = len(set(ticket).intersection(set(prev_ticket)))
                if shared > max_shared:
                    too_similar = True
                    break
            
            if not too_similar:
                seen_uniq.add(t_tup)
                unique_qualified.append(ticket)
                if len(unique_qualified) >= 2:
                    break
                    
    # Fallback: if we couldn't find a second ticket with low overlap, relax the constraint
    if len(unique_qualified) < 2 and len(qualified) > 1:
        for item in qualified:
            ticket = item["ticket"]
            t_tup = tuple(ticket)
            if t_tup not in seen_uniq:
                seen_uniq.add(t_tup)
                unique_qualified.append(ticket)
                if len(unique_qualified) >= 2:
                    break
                
    return unique_qualified

def predict_special_number_535(draws):
    """
    Predicts the special number (1-12) for Lotto 5/35 based on historical statistics.
    Returns a single number from 1 to 12.
    """
    try:
        if not draws:
            return 9 # sensible default
            
        special_numbers = []
        for d in draws:
            res = d.get("result", [])
            if len(res) >= 6:
                special_numbers.append(res[5])
                
        if not special_numbers:
            return 9
            
        # Let's count frequencies in the last 100 draws
        recent_specials = special_numbers[:100]
        from collections import Counter
        counts = Counter(recent_specials)
        
        # We can also calculate last seen for each number
        last_seen = {}
        for idx, num in enumerate(special_numbers):
            if num not in last_seen:
                last_seen[num] = idx
                
        best_num = 9
        best_score = -999999
        for n in range(1, 13):
            freq = counts.get(n, 0)
            seen = last_seen.get(n, len(special_numbers))
            
            if seen == 0:
                recency_penalty = 5.0
            elif 3 <= seen <= 12:
                recency_penalty = -2.0
            else:
                recency_penalty = seen * 0.1
                
            score = freq * 1.5 - recency_penalty
            if score > best_score:
                best_score = score
                best_num = n
                
        return best_num
    except Exception as e:
        print(f"Error predicting special number for power535: {e}")
        return 9

def predict_vietlott_game_ml(product_id, draws):
    # Limit draws to at most 1000 to keep ML prediction and backtesting extremely fast
    draws = draws[:1000]

    # Category 1: Standard draw games
    if product_id in ["mega645", "power655", "power535", "keno"]:
        num_per_draw = {
            "mega645": 6,
            "power655": 6,
            "power535": 5,
            "keno": 20
        }[product_id]
        
        max_num = {
            "mega645": 45,
            "power655": 55,
            "power535": 35,
            "keno": 80
        }[product_id]
        
        extractor = lambda d: d.get("result", [])[:num_per_draw]
        res_next = predict_draw_game(draws, 1, max_num, num_per_draw, extractor)
        if not res_next:
            return None
            
        model_names = ["linear_regression", "random_forest", "gradient_boosting", "mlp"]
        if xgb: model_names.append("xgboost")
        model_names.append("ensemble")
        
        # Initialize bias adjustments for online feedback learning
        # Maps model name -> number -> adjustment factor (starts at 0.0)
        bias_adjustments = {
            m: {n: 0.0 for n in range(1, max_num + 1)}
            for m in model_names
        }
        # Increased learning rate to 0.3 for faster adaptability to repeating hot streaks
        learning_rate = 0.3
        
        # Run rolling backtest chronologically (from oldest to newest) to simulate real-time learning
        num_backtests = min(20, len(draws) - 90)
        backtest_entries_chrono = []
        
        for k in range(num_backtests - 1, -1, -1):
            res_k = predict_draw_game(draws[k:], 1, max_num, num_per_draw, extractor)
            if not res_k:
                continue
                
            # Store up to 6 numbers for power535, 7 for power655, to include the special/bonus ball in history
            actual_len = 6 if product_id == "power535" else (7 if product_id == "power655" else num_per_draw)
            actual_numbers = draws[k].get("result", [])[:actual_len]
            actual_set = set(actual_numbers[:num_per_draw])
            backtest_entry = {
                "draw_id": draws[k].get("id"),
                "date": draws[k].get("date"),
                "actual_numbers": actual_numbers,
                "models": {}
            }
            
            step_probs = {}
            for m in model_names:
                back_probs = res_k["back"][m]
                
                # Apply current accumulated bias adjustments
                adjusted_probs = []
                for x in back_probs:
                    n = x["number"]
                    raw_p = x["prob"]
                    adj = bias_adjustments[m].get(n, 0.0)
                    adj_p = float(np.clip(raw_p + adj, 0.0, 1.0))
                    adjusted_probs.append({
                        "number": n,
                        "prob": adj_p,
                        "raw_prob": raw_p,
                        "adjustment": adj
                    })
                
                # Cache for ensemble step probs
                step_probs[m] = {x["number"]: x["prob"] for x in adjusted_probs}
                
                # Predict top numbers based on adjusted probabilities
                sorted_adjusted = sorted(adjusted_probs, key=lambda x: x["prob"], reverse=True)
                num_to_predict = 6 if product_id == "keno" else num_per_draw
                pred_back = [x["number"] for x in sorted_adjusted[:num_to_predict]]
                matched = list(set(pred_back).intersection(actual_set))
                
                backtest_entry["models"][m] = {
                    "predicted_numbers": pred_back,
                    "matched_numbers": matched,
                    "matched_count": len(matched)
                }
                
                # Update bias adjustments: feedback loop based on outcome
                for x in adjusted_probs:
                    n = x["number"]
                    adj_p = x["prob"]
                    y_n = 1.0 if n in actual_set else 0.0
                    error_n = y_n - adj_p
                    bias_adjustments[m][n] += learning_rate * error_n
                    
            # If standard draw game supports Monte Carlo, compute and save it
            if product_id in ["mega645", "power655", "power535"]:
                if product_id == "mega645":
                    split_num = 22
                elif product_id == "power655":
                    split_num = 27
                else:  # power535
                    split_num = 17
                profile_k = predict_draw_profile(draws[k:], num_per_draw, split_num)
                
                # Weighted average for Ensemble probs at step k: RF (25%), XGB (25% if exists, else add to RF/GB/MLP), GB (25%), MLP (15%), LR (10%)
                ensemble_probs_k = []
                base_models = [m for m in model_names if m != "ensemble"]
                weights = {}
                if "xgboost" in base_models:
                    weights = {
                        "random_forest": 0.25, 
                        "xgboost": 0.25, 
                        "gradient_boosting": 0.25, 
                        "mlp": 0.15, 
                        "linear_regression": 0.10
                    }
                else:
                    weights = {
                        "random_forest": 0.35, 
                        "gradient_boosting": 0.35, 
                        "mlp": 0.20, 
                        "linear_regression": 0.10
                    }
                    
                for n in range(1, max_num + 1):
                    w_sum = 0.0
                    for m in base_models:
                        w = weights.get(m, 1.0 / len(base_models))
                        w_sum += w * step_probs[m][n]
                    ensemble_probs_k.append({"number": n, "prob": w_sum})
                    
                monte_carlo_tickets_k = run_monte_carlo_filter(product_id, ensemble_probs_k, profile_k, num_per_draw, max_num, split_num, draws[k:])
                
                if product_id == "power535":
                    spec_num_k = predict_special_number_535(draws[k:])
                    monte_carlo_tickets_k = [t + [spec_num_k] for t in monte_carlo_tickets_k]
                
                # Evaluate matches for each ticket
                matches_k = []
                for t in monte_carlo_tickets_k:
                    # Evaluate intersection using only the main numbers (first num_per_draw elements)
                    m_nums = list(set(t[:num_per_draw]).intersection(actual_set))
                    matches_k.append({
                        "ticket": t,
                        "matched_numbers": m_nums,
                        "matched_count": len(m_nums)
                    })
                    
                matches_k.sort(key=lambda x: x["matched_count"], reverse=True)
                best_match = matches_k[0] if matches_k else {"ticket": [], "matched_numbers": [], "matched_count": 0}
                
                backtest_entry["models"]["monte_carlo"] = {
                    "predicted_numbers": best_match["ticket"],
                    "matched_numbers": best_match["matched_numbers"],
                    "matched_count": best_match["matched_count"],
                    "all_tickets": matches_k,
                    "profile": profile_k
                }
                
            backtest_entries_chrono.append(backtest_entry)
            
        # Predict next upcoming draw using final accumulated bias adjustments
        res_next = predict_draw_game(draws, 1, max_num, num_per_draw, extractor)
        if not res_next:
            return None
            
        forecast = {"models": {}, "backtest_history": list(reversed(backtest_entries_chrono))}
        
        for m in model_names:
            probs = res_next["next"][m]
            adjusted_probs = []
            for x in probs:
                n = x["number"]
                raw_p = x["prob"]
                adj = bias_adjustments[m].get(n, 0.0)
                adj_p = float(np.clip(raw_p + adj, 0.0, 1.0))
                adjusted_probs.append({
                    "number": n,
                    "prob": adj_p,
                    "raw_prob": raw_p,
                    "adjustment": adj
                })
                
            sorted_adjusted = sorted(adjusted_probs, key=lambda x: x["prob"], reverse=True)
            num_to_predict = 6 if product_id == "keno" else num_per_draw
            lucky = [x["number"] for x in sorted_adjusted[:num_to_predict]]
            
            if product_id == "power535":
                spec_num = predict_special_number_535(draws)
                lucky = lucky + [spec_num]

            forecast["models"][m] = {
                "lucky_numbers": lucky,
                "number_probabilities": adjusted_probs
            }

        # AI Profile & Monte Carlo Simulation (only for mega645, power655 & power535)
        if product_id in ["mega645", "power655", "power535"]:
            if product_id == "mega645":
                split_num = 22
            elif product_id == "power655":
                split_num = 27
            else:  # power535
                split_num = 17
            profile = predict_draw_profile(draws, num_per_draw, split_num)
            forecast["profile_prediction"] = profile
            
            # Use Ensemble probabilities as weights for Monte Carlo sampling
            ensemble_probs = forecast["models"]["ensemble"]["number_probabilities"]
            monte_carlo_tickets = run_monte_carlo_filter(product_id, ensemble_probs, profile, num_per_draw, max_num, split_num, draws)
            
            if product_id == "power535":
                spec_num = predict_special_number_535(draws)
                monte_carlo_tickets = [ticket + [spec_num] for ticket in monte_carlo_tickets]

            forecast["models"]["monte_carlo"] = {
                "lucky_numbers": monte_carlo_tickets,
                "number_probabilities": ensemble_probs
            }
            
        return forecast

    # Category 3: Bingo 18
    elif product_id == "bingo18":
        # 1. Predict 6 dice faces (min_num=1, max_num=6, num_per_draw=3)
        faces_res_next = predict_draw_game(draws, 1, 6, 3, lambda d: d.get("result", [])[:3])
        # 2. Predict 16 sums (min_num=3, max_num=18, num_per_draw=1)
        sums_res_next = predict_draw_game(draws, 3, 18, 1, lambda d: [d.get("total", sum(d.get("result", [])))])
        
        if not faces_res_next or not sums_res_next:
            return None
            
        model_names = ["linear_regression", "random_forest", "gradient_boosting", "mlp"]
        if xgb: model_names.append("xgboost")
        model_names.append("ensemble")
        
        # Initialize bias adjustments for faces and sums
        faces_bias = {
            m: {n: 0.0 for n in range(1, 7)}
            for m in model_names
        }
        sums_bias = {
            m: {n: 0.0 for n in range(3, 19)}
            for m in model_names
        }
        learning_rate = 0.3
        
        # Run rolling backtest chronologically
        num_backtests = min(20, len(draws) - 90)
        backtest_entries_chrono = []
        
        for k in range(num_backtests - 1, -1, -1):
            res_faces_k = predict_draw_game(draws[k:], 1, 6, 3, lambda d: d.get("result", [])[:3])
            res_sums_k = predict_draw_game(draws[k:], 3, 18, 1, lambda d: [d.get("total", sum(d.get("result", [])))])
            if not res_faces_k or not res_sums_k:
                continue
                
            actual_numbers = draws[k].get("result", [])[:3]
            actual_set = set(actual_numbers)
            actual_total = draws[k].get("total", sum(draws[k].get("result", [])))
            
            backtest_entry = {
                "draw_id": draws[k].get("id"),
                "date": draws[k].get("date"),
                "actual_numbers": actual_numbers,
                "actual_total": actual_total,
                "models": {}
            }
            
            for m in model_names:
                # Faces prediction with bias adjustment
                back_face_probs = res_faces_k["back"][m]
                adjusted_face_probs = []
                for x in back_face_probs:
                    n = x["number"]
                    raw_p = x["prob"]
                    adj = faces_bias[m].get(n, 0.0)
                    adj_p = float(np.clip(raw_p + adj, 0.0, 1.0))
                    adjusted_face_probs.append({
                        "number": n,
                        "prob": adj_p,
                        "raw_prob": raw_p,
                        "adjustment": adj
                    })
                sorted_faces = sorted(adjusted_face_probs, key=lambda x: x["prob"], reverse=True)
                pred_back_faces = [x["number"] for x in sorted_faces[:3]]
                matched_faces = list(set(pred_back_faces).intersection(actual_set))
                
                # Sums prediction with bias adjustment
                back_sum_probs = res_sums_k["back"][m]
                adjusted_sum_probs = []
                for x in back_sum_probs:
                    n = x["number"]
                    raw_p = x["prob"]
                    adj = sums_bias[m].get(n, 0.0)
                    adj_p = float(np.clip(raw_p + adj, 0.0, 1.0))
                    adjusted_sum_probs.append({
                        "number": n,
                        "prob": adj_p,
                        "raw_prob": raw_p,
                        "adjustment": adj
                    })
                sorted_sums = sorted(adjusted_sum_probs, key=lambda x: x["prob"], reverse=True)
                pred_back_sum = sorted_sums[0]["number"]
                sum_matched = (pred_back_sum == actual_total)
                
                backtest_entry["models"][m] = {
                    "predicted_numbers": pred_back_faces,
                    "matched_numbers": matched_faces,
                    "matched_count": len(matched_faces),
                    "predicted_total": pred_back_sum,
                    "total_matched": sum_matched
                }
                
                # Update faces bias adjustments
                for x in adjusted_face_probs:
                    n = x["number"]
                    adj_p = x["prob"]
                    y_n = 1.0 if n in actual_set else 0.0
                    error_n = y_n - adj_p
                    faces_bias[m][n] += learning_rate * error_n
                    
                # Update sums bias adjustments
                for x in adjusted_sum_probs:
                    n = x["number"]
                    adj_p = x["prob"]
                    y_n = 1.0 if n == actual_total else 0.0
                    error_n = y_n - adj_p
                    sums_bias[m][n] += learning_rate * error_n
                    
            backtest_entries_chrono.append(backtest_entry)
            
        # Build final predictions with the final accumulated adjustments
        forecast = {"models": {}, "backtest_history": list(reversed(backtest_entries_chrono))}
        
        for m in model_names:
            # Faces
            face_probs = faces_res_next["next"][m]
            adjusted_face_probs = []
            for x in face_probs:
                n = x["number"]
                raw_p = x["prob"]
                adj = faces_bias[m].get(n, 0.0)
                adj_p = float(np.clip(raw_p + adj, 0.0, 1.0))
                adjusted_face_probs.append({
                    "number": n,
                    "prob": adj_p,
                    "raw_prob": raw_p,
                    "adjustment": adj
                })
            sorted_faces = sorted(adjusted_face_probs, key=lambda x: x["prob"], reverse=True)
            lucky_faces = [x["number"] for x in sorted_faces[:3]]
            
            # Sums
            sum_probs = sums_res_next["next"][m]
            adjusted_sum_probs = []
            for x in sum_probs:
                n = x["number"]
                raw_p = x["prob"]
                adj = sums_bias[m].get(n, 0.0)
                adj_p = float(np.clip(raw_p + adj, 0.0, 1.0))
                adjusted_sum_probs.append({
                    "number": n,
                    "prob": adj_p,
                    "raw_prob": raw_p,
                    "adjustment": adj
                })
            sorted_sums = sorted(adjusted_sum_probs, key=lambda x: x["prob"], reverse=True)
            lucky_sum = sorted_sums[0]["number"]
            
            forecast["models"][m] = {
                "lucky_numbers": lucky_faces,
                "number_probabilities": adjusted_face_probs,
                "lucky_sum": lucky_sum,
                "sum_probabilities": adjusted_sum_probs
            }
            
        return forecast

    # Category 2: Max 3D / Max 3D Pro
    elif product_id in ["max3d", "max3dpro"]:
        pos_res_next = {}
        for pos_idx, pos_name in enumerate(["hundreds", "tens", "units"]):
            extractor = lambda d, pi=pos_idx: [int(val[pi]) for val in get_all_3d_strings(d)]
            pos_res_next[pos_name] = predict_draw_game(draws, 0, 9, 1, extractor)
            
        if any(v is None for v in pos_res_next.values()):
            return None
            
        model_names = ["linear_regression", "random_forest", "gradient_boosting", "mlp"]
        if xgb: model_names.append("xgboost")
        model_names.append("ensemble")
        
        # Initialize bias adjustments for three positions: hundreds, tens, units
        bias_adjustments = {
            m: {
                pos: {d: 0.0 for d in range(10)}
                for pos in ["hundreds", "tens", "units"]
            }
            for m in model_names
        }
        learning_rate = 0.3
        
        # Run rolling backtest chronologically
        num_backtests = min(20, len(draws) - 90)
        backtest_entries_chrono = []
        
        for k in range(num_backtests - 1, -1, -1):
            pos_res_k = {}
            for pos_idx, pos_name in enumerate(["hundreds", "tens", "units"]):
                extractor = lambda d, pi=pos_idx: [int(val[pi]) for val in get_all_3d_strings(d)]
                pos_res_k[pos_name] = predict_draw_game(draws[k:], 0, 9, 1, extractor)
                
            if any(v is None for v in pos_res_k.values()):
                continue
                
            actual_numbers = get_all_3d_strings(draws[k])
            actual_digits = {
                "hundreds": set(int(val[0]) for val in actual_numbers),
                "tens": set(int(val[1]) for val in actual_numbers),
                "units": set(int(val[2]) for val in actual_numbers)
            }
            
            backtest_entry = {
                "draw_id": draws[k].get("id"),
                "date": draws[k].get("date"),
                "actual_numbers": actual_numbers,
                "models": {}
            }
            
            for m in model_names:
                h_back = pos_res_k["hundreds"]["back"][m]
                t_back = pos_res_k["tens"]["back"][m]
                u_back = pos_res_k["units"]["back"][m]
                
                # Apply current accumulated bias adjustments
                adjusted_positions = {}
                for pos_name, back_probs in [("hundreds", h_back), ("tens", t_back), ("units", u_back)]:
                    adjusted_probs = []
                    for x in back_probs:
                        d = x["number"]
                        raw_p = x["prob"]
                        adj = bias_adjustments[m][pos_name].get(d, 0.0)
                        adj_p = float(np.clip(raw_p + adj, 0.0, 1.0))
                        adjusted_probs.append({
                            "number": d,
                            "prob": adj_p,
                            "raw_prob": raw_p,
                            "adjustment": adj
                        })
                    adjusted_positions[pos_name] = adjusted_probs
                
                h_map = {x["number"]: x["prob"] for x in adjusted_positions["hundreds"]}
                t_map = {x["number"]: x["prob"] for x in adjusted_positions["tens"]}
                u_map = {x["number"]: x["prob"] for x in adjusted_positions["units"]}
                
                combos_back = []
                for val in range(1000):
                    val_str = f"{val:03d}"
                    h = int(val_str[0])
                    t = int(val_str[1])
                    u = int(val_str[2])
                    p = h_map[h] * t_map[t] * u_map[u]
                    combos_back.append({"combo": val_str, "prob": float(p)})
                    
                sorted_combos_back = sorted(combos_back, key=lambda x: x["prob"], reverse=True)
                pred_back_combos = [x["combo"] for x in sorted_combos_back[:3]]
                matched = list(set(pred_back_combos).intersection(set(actual_numbers)))
                
                backtest_entry["models"][m] = {
                    "predicted_numbers": pred_back_combos,
                    "matched_numbers": matched,
                    "matched_count": len(matched)
                }
                
                # Update position bias adjustments
                for pos_name in ["hundreds", "tens", "units"]:
                    for x in adjusted_positions[pos_name]:
                        d = x["number"]
                        adj_p = x["prob"]
                        y_d = 1.0 if d in actual_digits[pos_name] else 0.0
                        error_d = y_d - adj_p
                        bias_adjustments[m][pos_name][d] += learning_rate * error_d
                        
            backtest_entries_chrono.append(backtest_entry)
            
        # Build final predictions using the final accumulated adjustments
        forecast = {"models": {}, "backtest_history": list(reversed(backtest_entries_chrono))}
        
        for m in model_names:
            h_probs = pos_res_next["hundreds"]["next"][m]
            t_probs = pos_res_next["tens"]["next"][m]
            u_probs = pos_res_next["units"]["next"][m]
            
            adjusted_positions = {}
            for pos_name, probs in [("hundreds", h_probs), ("tens", t_probs), ("units", u_probs)]:
                adjusted_probs = []
                for x in probs:
                    d = x["number"]
                    raw_p = x["prob"]
                    adj = bias_adjustments[m][pos_name].get(d, 0.0)
                    adj_p = float(np.clip(raw_p + adj, 0.0, 1.0))
                    adjusted_probs.append({
                        "number": d,
                        "prob": adj_p,
                        "raw_prob": raw_p,
                        "adjustment": adj
                    })
                adjusted_positions[pos_name] = adjusted_probs
                
            h_map = {x["number"]: x["prob"] for x in adjusted_positions["hundreds"]}
            t_map = {x["number"]: x["prob"] for x in adjusted_positions["tens"]}
            u_map = {x["number"]: x["prob"] for x in adjusted_positions["units"]}
            
            combos = []
            for val in range(1000):
                val_str = f"{val:03d}"
                h = int(val_str[0])
                t = int(val_str[1])
                u = int(val_str[2])
                p = h_map[h] * t_map[t] * u_map[u]
                combos.append({"combo": val_str, "prob": float(p)})
                
            sorted_combos = sorted(combos, key=lambda x: x["prob"], reverse=True)
            lucky_combos = [x["combo"] for x in sorted_combos[:3]]
            
            forecast["models"][m] = {
                "lucky_numbers": lucky_combos,
                "position_probabilities": adjusted_positions
            }
            
        return forecast
    return None

def sync_vietlott_data(product_id: str):
    """
    Runs the local crawler to fetch the latest draw results from the official Vietlott website,
    loads the data from the local JSONL file, calculates statistics, and caches results in MongoDB.
    """
    if product_id not in PRODUCTS:
        raise ValueError(f"Sản phẩm {product_id} không được hỗ trợ.")
        
    github_file = PRODUCTS[product_id]
    
    # Run the local crawler
    crawler_product_names = {
        "mega645": "power_645",
        "power655": "power_655",
        "power535": "power_535",
        "keno": "keno",
        "max3d": "3d",
        "max3dpro": "3d_pro",
        "bingo18": "bingo18"
    }
    crawler_prod = crawler_product_names.get(product_id)
    if crawler_prod:
        import subprocess
        import os
        
        import sys
        cmd = [
            sys.executable, 
            "-m", "vietlott.cli.crawl", 
            crawler_prod
        ]
        
        print(f"Vietlott Sync: Running local crawler for {crawler_prod}...")
        env = os.environ.copy()
        env["PYTHONSAFEPATH"] = "1"
        env["PYTHONPATH"] = "vietlott-data-crawler/src"
        
        try:
            result = subprocess.run(
                cmd, 
                env=env, 
                capture_output=True, 
                text=True, 
                timeout=60
            )
            if result.returncode != 0:
                print(f"Local crawler warning (non-zero return code): {result.stderr.strip()}")
            else:
                print(f"Local crawler output: {result.stdout.strip()}")
        except Exception as e:
            print(f"Failed to run local crawler: {e}")

    # Read the data (prefer local file, fallback to GitHub)
    import os
    local_file_path = f"vietlott-data-crawler/data/{github_file}.jsonl"
    lines = []
    
    if os.path.exists(local_file_path):
        print(f"Vietlott Sync: Reading raw data from local file {local_file_path}...")
        try:
            with open(local_file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Failed to read local file {local_file_path}: {e}")
            
    if not lines:
        raise Exception(f"Không tìm thấy dữ liệu cục bộ cho {product_id} tại {local_file_path}. Vui lòng kiểm tra lại trình crawler.")
        
    # Read NDJSON line by line
    draws = []
    for line in lines:
        if not line.strip():
            continue
        try:
            draws.append(json.loads(line))
        except Exception as e:
            print(f"Error parsing line in {product_id}: {e}")
            
    if not draws:
        raise Exception(f"Không có dữ liệu hợp lệ cho {product_id}")
        
    # Sort draws by date and ID descending (newest first)
    # Some Keno drawings happen on the same day, so we sort by date then ID
    draws.sort(key=lambda x: (x.get("date", ""), x.get("id", "")), reverse=True)
    
    # Calculate stats based on game type
    stats = {}
    
    # Category 1: Standard number draw games (Mega 6/45, Power 6/55, Keno, Power 5/35)
    if product_id in ["mega645", "power655", "keno", "power535"]:
        stats = calculate_draw_game_stats(product_id, draws)
    # Category 2: Max 3D / Max 3D Pro (3-digit prize tables)
    elif product_id in ["max3d", "max3dpro"]:
        stats = calculate_3d_game_stats(product_id, draws)
    # Category 3: Bingo 18 (3 numbers 1-6 + sum + size)
    elif product_id == "bingo18":
        stats = calculate_bingo18_stats(draws)
        
    # Calculate AI Forecast
    try:
        print(f"Vietlott Sync: Generating AI forecast for {product_id}...")
        forecast = predict_vietlott_game_ml(product_id, draws)
    except Exception as e:
        print(f"Error generating AI forecast for {product_id}: {e}")
        forecast = None
        
    # Save cache document
    cache_doc = {
        "_id": product_id,
        "product_id": product_id,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_draws": len(draws),
        "latest_draw": draws[0] if draws else None,
        "recent_draws": draws[:100],  # Keep last 100 draws
        "stats": stats,
        "forecast": forecast
    }
    
    db.vietlott_cache.replace_one({"_id": product_id}, cache_doc, upsert=True)
    print(f"Vietlott Sync: Successfully cached {product_id} in MongoDB. Total draws: {len(draws)}.")
    return cache_doc

def calculate_draw_game_stats(product_id: str, draws: list):
    """
    Calculates number frequency, coldness, odd/even, and large/small stats for draw-style lotteries.
    """
    # Define game boundaries
    if product_id == "mega645":
        max_num = 45
        split_num = 22  # 1-22 small, 23-45 large
        num_per_draw = 6
    elif product_id == "power655":
        max_num = 55
        split_num = 27  # 1-27 small, 28-55 large
        num_per_draw = 6 # Differentiate 7th as bonus ball
    elif product_id == "power535":
        max_num = 35
        split_num = 17  # 1-17 small, 18-35 large
        num_per_draw = 5 # 5 regular balls, 6th is bonus
    elif product_id == "keno":
        max_num = 80
        split_num = 40  # 1-40 small, 41-80 large
        num_per_draw = 20

    # Initialize stats dict for all numbers from 1 to max_num
    all_numbers = list(range(1, max_num + 1))
    freq_all = {n: 0 for n in all_numbers}
    freq_100 = {n: 0 for n in all_numbers}
    freq_60 = {n: 0 for n in all_numbers}
    freq_30 = {n: 0 for n in all_numbers}
    
    # Draws since last appearance
    last_seen_index = {n: len(draws) for n in all_numbers} # Default: total draws (never seen)
    
    # Analyze draws
    odd_count = 0
    even_count = 0
    small_count = 0
    large_count = 0
    total_analyzed_balls = 0
    
    # Track splits distribution
    odd_even_splits = {}
    small_large_splits = {}
    
    # We analyze up to 100 draws for odd/even & large/small to represent recent distributions
    recent_limit = min(100, len(draws))
    
    for idx, draw in enumerate(draws):
        result = draw.get("result", [])
        if not isinstance(result, list):
            continue
            
        # Determine regular numbers (in Power 6/55 and Power 5/35, the result list has extra numbers)
        # Power 6/55 has 7 numbers: first 6 are regular, 7th is bonus.
        # Power 5/35 has 6 numbers: first 5 are regular, 6th is bonus.
        reg_result = result[:num_per_draw]
        
        # In frequency and cold calculations, do we include the bonus ball?
        # Yes, let's include the whole result list so users see stats of ALL balls drawn.
        # But for odd/even distribution stats, let's look at regular numbers or recent draws.
        draw_balls = result
        
        # Track frequencies
        for num in draw_balls:
            if num in freq_all:
                freq_all[num] += 1
                if idx < 100:
                    freq_100[num] += 1
                if idx < 60:
                    freq_60[num] += 1
                if idx < 30:
                    freq_30[num] += 1
                    
        # Track last appearance (only first time we see it, which is the newest since draws are sorted desc)
        for num in draw_balls:
            if num in last_seen_index and last_seen_index[num] == len(draws):
                last_seen_index[num] = idx
                
        # Track distributions for recent draws
        if idx < recent_limit:
            for num in reg_result:
                total_analyzed_balls += 1
                if num % 2 == 1:
                    odd_count += 1
                else:
                    even_count += 1
                    
                if num <= split_num:
                    small_count += 1
                else:
                    large_count += 1
            
            # Analyze splits
            odds = sum(1 for n in reg_result if n % 2 == 1)
            evens = num_per_draw - odds
            oe_key = f"{odds}:{evens}"
            odd_even_splits[oe_key] = odd_even_splits.get(oe_key, 0) + 1
            
            smalls = sum(1 for n in reg_result if n <= split_num)
            larges = num_per_draw - smalls
            sl_key = f"{smalls}:{larges}"
            small_large_splits[sl_key] = small_large_splits.get(sl_key, 0) + 1

    # Calculate co-occurrence matrix for the last 300 draws
    co_occur = {n: {other: 0 for other in all_numbers if other != n} for n in all_numbers}
    limit_co = min(300, len(draws))
    for idx in range(limit_co):
        result = draws[idx].get("result", [])
        balls = [b for b in result if b in freq_all]
        for i in range(len(balls)):
            for j in range(i + 1, len(balls)):
                a, b = balls[i], balls[j]
                if a in co_occur and b in co_occur[a]:
                    co_occur[a][b] += 1
                    co_occur[b][a] += 1

    # Form stats list
    number_stats = []
    for n in all_numbers:
        partners_sorted = sorted(co_occur[n].items(), key=lambda x: x[1], reverse=True)[:5]
        top_partners = [
            {"number": p[0], "count": p[1], "prob_pct": round((p[1] / max(1, freq_all[n])) * 100, 1)} 
            for p in partners_sorted
        ]
        number_stats.append({
            "number": n,
            "total_occurrences": freq_all[n],
            "freq_100": freq_100[n],
            "freq_60": freq_60[n],
            "freq_30": freq_30[n],
            "draws_since_last": last_seen_index[n],
            "top_partners": top_partners
        })
        
    # Sort number stats to find Hot and Cold lists
    # Hot numbers: highest frequency in last 100 draws
    hot_numbers = sorted(number_stats, key=lambda x: x["freq_100"], reverse=True)[:10]
    
    # Cold numbers: longest time since last appearance
    cold_numbers = sorted(number_stats, key=lambda x: x["draws_since_last"], reverse=True)[:10]
    
    # Odd/Even and Large/Small percentages
    odd_percent = round((odd_count / total_analyzed_balls) * 100, 1) if total_analyzed_balls > 0 else 50.0
    even_percent = round((even_count / total_analyzed_balls) * 100, 1) if total_analyzed_balls > 0 else 50.0
    small_percent = round((small_count / total_analyzed_balls) * 100, 1) if total_analyzed_balls > 0 else 50.0
    large_percent = round((large_count / total_analyzed_balls) * 100, 1) if total_analyzed_balls > 0 else 50.0
    
    stats_odd_even_splits = [
        { "pattern": k, "count": v, "pct": round((v / recent_limit) * 100, 1) } 
        for k, v in sorted(odd_even_splits.items(), key=lambda x: x[1], reverse=True)
    ]
    stats_small_large_splits = [
        { "pattern": k, "count": v, "pct": round((v / recent_limit) * 100, 1) } 
        for k, v in sorted(small_large_splits.items(), key=lambda x: x[1], reverse=True)
    ]
    
    return {
        "number_stats": number_stats,
        "hot_numbers": hot_numbers,
        "cold_numbers": cold_numbers,
        "distribution": {
            "odd_percent": odd_percent,
            "even_percent": even_percent,
            "small_percent": small_percent,
            "large_percent": large_percent,
            "sample_draws": recent_limit
        },
        "odd_even_splits": stats_odd_even_splits,
        "small_large_splits": stats_small_large_splits
    }

def calculate_3d_game_stats(product_id: str, draws: list):
    """
    Calculates digit frequencies (0-9) at each position (hundreds, tens, units) for Max 3D / Max 3D Pro.
    """
    # Max 3D results look like:
    # {"Giải Đặc biệt": ["015", "517"], "Giải Nhất": ["334", "279", ...]}
    # We aggregate all 3-digit strings across all prize tiers in a draw
    
    digit_positions = {
        "hundreds": {d: 0 for d in range(10)},
        "tens": {d: 0 for d in range(10)},
        "units": {d: 0 for d in range(10)}
    }
    
    # Recent limit: last 100 draws
    digit_positions_100 = {
        "hundreds": {d: 0 for d in range(10)},
        "tens": {d: 0 for d in range(10)},
        "units": {d: 0 for d in range(10)}
    }
    
    # Last appearance index for each digit (0-9) at each position
    last_seen = {
        "hundreds": {d: len(draws) for d in range(10)},
        "tens": {d: len(draws) for d in range(10)},
        "units": {d: len(draws) for d in range(10)}
    }
    
    total_3d_numbers = 0
    total_3d_numbers_100 = 0
    
    for idx, draw in enumerate(draws):
        result = draw.get("result", {})
        if not isinstance(result, dict):
            continue
            
        # Collect all 3-digit strings in this draw
        draw_3d_strings = []
        for prize, values in result.items():
            if isinstance(values, list):
                for val in values:
                    if isinstance(val, str) and len(val) == 3 and val.isdigit():
                        draw_3d_strings.append(val)
                        
        for val in draw_3d_strings:
            total_3d_numbers += 1
            if idx < 100:
                total_3d_numbers_100 += 1
                
            h = int(val[0])
            t = int(val[1])
            u = int(val[2])
            
            digit_positions["hundreds"][h] += 1
            digit_positions["tens"][t] += 1
            digit_positions["units"][u] += 1
            
            if idx < 100:
                digit_positions_100["hundreds"][h] += 1
                digit_positions_100["tens"][t] += 1
                digit_positions_100["units"][u] += 1
                
            # Track last seen
            if last_seen["hundreds"][h] == len(draws):
                last_seen["hundreds"][h] = idx
            if last_seen["tens"][t] == len(draws):
                last_seen["tens"][t] = idx
            if last_seen["units"][u] == len(draws):
                last_seen["units"][u] = idx

    # Compute final position stats list
    position_stats = {}
    for pos in ["hundreds", "tens", "units"]:
        pos_list = []
        for d in range(10):
            pos_list.append({
                "digit": d,
                "total_occurrences": digit_positions[pos][d],
                "freq_100": digit_positions_100[pos][d],
                "draws_since_last": last_seen[pos][d]
            })
        position_stats[pos] = pos_list
        
    return {
        "position_stats": position_stats,
        "total_numbers_analyzed": total_3d_numbers,
        "sample_draws": len(draws)
    }

def calculate_bingo18_stats(draws: list):
    """
    Calculates statistics for Bingo 18 (numbers 1-6, sums 3-18, and large/small ratio).
    """
    # Bingo 18 results look like:
    # {"date":"2024-12-03","id":"0083123","result":[2,6,1],"total":9,"large_small":"Nhỏ"}
    
    number_freq = {n: 0 for n in range(1, 7)}
    number_freq_100 = {n: 0 for n in range(1, 7)}
    number_last_seen = {n: len(draws) for n in range(1, 7)}
    
    sum_freq = {s: 0 for s in range(3, 19)}
    sum_freq_100 = {s: 0 for s in range(3, 19)}
    
    small_count = 0
    large_count = 0
    triple_count = 0
    double_count = 0
    total_recent = 0
    
    for idx, draw in enumerate(draws):
        result = draw.get("result", [])
        if not isinstance(result, list) or len(result) != 3:
            continue
            
        total = draw.get("total", sum(result))
        large_small = draw.get("large_small", "")
        
        # Track sum frequencies
        if total in sum_freq:
            sum_freq[total] += 1
            if idx < 100:
                sum_freq_100[total] += 1
                
        # Track number frequencies & last seen
        for num in result:
            if num in number_freq:
                number_freq[num] += 1
                if idx < 100:
                    number_freq_100[num] += 1
                if number_last_seen[num] == len(draws):
                    number_last_seen[num] = idx
                    
        # Track recent game distribution (last 100 draws)
        if idx < 100:
            total_recent += 1
            if large_small == "Nhỏ" or total <= 10:
                small_count += 1
            elif large_small == "Lớn" or total >= 11:
                large_count += 1
                
            # Triple check (3 identical numbers)
            if result[0] == result[1] == result[2]:
                triple_count += 1
            # Double check (2 identical numbers)
            elif result[0] == result[1] or result[1] == result[2] or result[0] == result[2]:
                double_count += 1

    # Form stats list
    number_stats = []
    for n in range(1, 7):
        number_stats.append({
            "number": n,
            "total_occurrences": number_freq[n],
            "freq_100": number_freq_100[n],
            "draws_since_last": number_last_seen[n]
        })
        
    sum_stats = []
    for s in range(3, 19):
        sum_stats.append({
            "sum": s,
            "total_occurrences": sum_freq[s],
            "freq_100": sum_freq_100[s]
        })
        
    return {
        "number_stats": number_stats,
        "sum_stats": sum_stats,
        "distribution": {
            "small_percent": round((small_count / total_recent) * 100, 1) if total_recent > 0 else 50.0,
            "large_percent": round((large_count / total_recent) * 100, 1) if total_recent > 0 else 50.0,
            "triple_percent": round((triple_count / total_recent) * 100, 1) if total_recent > 0 else 0.0,
            "double_percent": round((double_count / total_recent) * 100, 1) if total_recent > 0 else 0.0,
            "sample_draws": total_recent
        }
    }

def get_cached_vietlott_data(product_id: str):
    """
    Returns the cached data from MongoDB. If cache is empty, triggers a sync first.
    """
    if product_id not in PRODUCTS:
        raise ValueError(f"Sản phẩm {product_id} không được hỗ trợ.")
        
    cached = db.vietlott_cache.find_one({"_id": product_id})
    if not cached:
        print(f"Cache miss for {product_id}. Performing initial sync...")
        cached = sync_vietlott_data(product_id)
        
    # Remove _id from dict for JSON serialization compatibility
    result_data = cached.copy()
    result_data.pop("_id", None)
    return result_data

def sync_all():
    """
    Synchronizes all supported Vietlott products.
    """
    print("Vietlott Sync: Starting synchronization of all products...")
    results = {}
    for product in PRODUCTS:
        try:
            sync_vietlott_data(product)
            results[product] = "success"
        except Exception as e:
            print(f"Error syncing {product}: {e}")
            results[product] = f"error: {str(e)}"
    return results


def load_all_draws(product_id: str):
    """
    Loads all draws from local file or downloads from GitHub.
    """
    if product_id not in PRODUCTS:
        return []
        
    github_file = PRODUCTS[product_id]
    local_file_path = f"vietlott-data-crawler/data/{github_file}.jsonl"
    draws = []
    
    if os.path.exists(local_file_path):
        try:
            with open(local_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        draws.append(json.loads(line))
        except Exception as e:
            print(f"Error loading draws from {local_file_path}: {e}")
            
    if not draws:
        url = f"https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/{github_file}.jsonl"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                for line in r.text.strip().split('\n'):
                    if line.strip():
                        draws.append(json.loads(line))
        except Exception as e:
            print(f"Error downloading draws from GitHub: {e}")
            
    draws.sort(key=lambda x: (x.get("date", ""), x.get("id", "")), reverse=True)
    return draws


def backtest_user_numbers(product_id: str, user_nums: list):
    """
    Simulates buying a ticket with user_nums against all historical draws.
    """
    draws = load_all_draws(product_id)
    if not draws:
        return {"status": "error", "message": "Không thể tải dữ liệu lịch sử xổ số."}
        
    user_nums_set = set(user_nums)
    total_tickets = len(draws)
    ticket_price = 10000
    total_cost = total_tickets * ticket_price
    
    total_winnings = 0
    wins = {}
    high_prize_logs = []
    
    if product_id == "mega645":
        jackpot_wins = 0
        first_wins = 0
        second_wins = 0
        third_wins = 0
        
        for d in draws:
            res = d.get("result", [])
            if not res or len(res) < 6:
                continue
            matches = len(user_nums_set.intersection(res[:6]))
            if matches == 6:
                jackpot_wins += 1
                prize_amt = 12000000000  # 12 Billion average
                total_winnings += prize_amt
                high_prize_logs.append({
                    "date": d.get("date", ""),
                    "id": d.get("id", ""),
                    "prize": "Jackpot",
                    "amount": prize_amt,
                    "matches": 6,
                    "winning_numbers": res[:6]
                })
            elif matches == 5:
                first_wins += 1
                prize_amt = 10000000  # 10 Million
                total_winnings += prize_amt
                high_prize_logs.append({
                    "date": d.get("date", ""),
                    "id": d.get("id", ""),
                    "prize": "Giải Nhất",
                    "amount": prize_amt,
                    "matches": 5,
                    "winning_numbers": res[:6]
                })
            elif matches == 4:
                second_wins += 1
                total_winnings += 300000  # 300k
            elif matches == 3:
                third_wins += 1
                total_winnings += 30000   # 30k
                
        wins = {
            "jackpot": jackpot_wins,
            "first": first_wins,
            "second": second_wins,
            "third": third_wins
        }
        
    elif product_id == "power655":
        jackpot1_wins = 0
        jackpot2_wins = 0
        first_wins = 0
        second_wins = 0
        third_wins = 0
        
        for d in draws:
            res = d.get("result", [])
            if not res or len(res) < 7:
                continue
            reg_balls = res[:6]
            bonus_ball = res[6]
            
            matches_reg = len(user_nums_set.intersection(reg_balls))
            if matches_reg == 6:
                jackpot1_wins += 1
                prize_amt = 30000000000  # 30 Billion average
                total_winnings += prize_amt
                high_prize_logs.append({
                    "date": d.get("date", ""),
                    "id": d.get("id", ""),
                    "prize": "Jackpot 1",
                    "amount": prize_amt,
                    "matches": 6,
                    "winning_numbers": reg_balls
                })
            elif matches_reg == 5:
                if bonus_ball in user_nums_set:
                    jackpot2_wins += 1
                    prize_amt = 3000000000  # 3 Billion average
                    total_winnings += prize_amt
                    high_prize_logs.append({
                        "date": d.get("date", ""),
                        "id": d.get("id", ""),
                        "prize": "Jackpot 2",
                        "amount": prize_amt,
                        "matches": 5,
                        "winning_numbers": reg_balls + [bonus_ball]
                    })
                else:
                    first_wins += 1
                    prize_amt = 40000000  # 40 Million
                    total_winnings += prize_amt
                    high_prize_logs.append({
                        "date": d.get("date", ""),
                        "id": d.get("id", ""),
                        "prize": "Giải Nhất",
                        "amount": prize_amt,
                        "matches": 5,
                        "winning_numbers": reg_balls
                    })
            elif matches_reg == 4:
                second_wins += 1
                total_winnings += 500000  # 500k
            elif matches_reg == 3:
                third_wins += 1
                total_winnings += 50000   # 50k
                
        wins = {
            "jackpot1": jackpot1_wins,
            "jackpot2": jackpot2_wins,
            "first": first_wins,
            "second": second_wins,
            "third": third_wins
        }
    else:
        return {"status": "error", "message": f"Sản phẩm {product_id} chưa hỗ trợ mô phỏng giải thưởng."}
        
    net_profit = total_winnings - total_cost
    roi = ((total_winnings / max(1, total_cost)) * 100) - 100
    
    return {
        "status": "success",
        "product_id": product_id,
        "total_draws": total_tickets,
        "total_tickets": total_tickets,
        "total_cost": total_cost,
        "total_winnings": total_winnings,
        "net_profit": net_profit,
        "roi": round(roi, 2),
        "wins": wins,
        "high_prize_logs": high_prize_logs[:20]  # Return top 20 wins logs
    }


if __name__ == "__main__":
    # Test sync for one product
    sync_vietlott_data("power655")
    sync_vietlott_data("mega645")
    sync_vietlott_data("power535")
