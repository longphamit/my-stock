import requests
import json
import warnings
import hashlib
import math
import random
from itertools import combinations
from datetime import datetime, timedelta
from pymongo import MongoClient
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
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

LOTTERY_CONFIG = {
    "mega645": {
        "crawler": "power_645",
        "min_num": 1,
        "max_num": 45,
        "num_per_draw": 6,
        "split_num": 22,
        "training_draws": 120,
    },
    "power655": {
        "crawler": "power_655",
        "min_num": 1,
        "max_num": 55,
        "num_per_draw": 6,
        "split_num": 27,
        "training_draws": 120,
    },
    "power535": {
        "crawler": "power_535",
        "min_num": 1,
        "max_num": 35,
        "num_per_draw": 5,
        "split_num": 17,
        # Selected by chronological walk-forward validation on the complete
        # available history. It improved recent OOS coverage over 120 and 480.
        "training_draws": 240,
    },
}

# Offer explicit budget tiers. Power 5/35 has dedicated two- and three-ticket
# strategies because their objective is specifically >=3 matches on one ticket
# while keeping the purchase budget small.
COVERAGE_TICKET_BUDGETS = (2, 3, 5, 10)
DEFAULT_COVERAGE_TICKET_COUNT = 2
DEFAULT_BACKTEST_WINDOW = 60
MAX_BACKTEST_WINDOW = 100
RECENCY_TRANSFORM_SCALE = 3.0
RELATION_PRIOR_STRENGTH = 8.0
# A 4-number subset is rarer than a 3-number subset. Keep it in the joint
# objective, but do not let a handful of quadruples crowd out the primary
# >=3-hit objective for a small two-ticket budget.
QUADRUPLE_OBJECTIVE_WEIGHT = 4.0

VIETLOTT_ROUNDTABLE_WEIGHTS = {
    # Start neutral. Architecture names are not evidence of predictive skill;
    # rolling OOS Brier and prior hit history adjust these weights over time.
    "xgb_ranker": 1.0,
    "random_forest": 1.0,
    "xgboost": 1.0,
    "gradient_boosting": 1.0,
    "mlp": 1.0,
    "linear_regression": 1.0,
}


def _roundtable_model_weights(model_keys):
    """Return stable normalized chair weights for the available members."""
    active = [key for key in VIETLOTT_ROUNDTABLE_WEIGHTS if key in model_keys]
    total = sum(VIETLOTT_ROUNDTABLE_WEIGHTS[key] for key in active) or 1.0
    return {key: VIETLOTT_ROUNDTABLE_WEIGHTS[key] / total for key in active}


def _build_roundtable_vote_tickets(
    predictions,
    num_per_draw,
    max_num,
    ticket_count=2,
    relationship_scores=None,
    pattern_profile=None,
    quality_reports=None,
    performance_history=None,
):
    """Turn member rankings into a deterministic, auditable vote portfolio.

    This is deliberately deterministic so historical results can be reproduced.
    Codex may scientifically review the live decision, but is not replayed with
    today's model for old draws because that would make the backtest mutable.
    """
    weights = _roundtable_model_weights(predictions.keys())
    if not weights:
        return [], [], []
    quality_reports = quality_reports or {}
    baseline_brier = (num_per_draw / max_num) * (1.0 - num_per_draw / max_num)
    adjusted_weights = {}
    for model, base_weight in weights.items():
        report = quality_reports.get(model) or {}
        brier = report.get("calibrated_brier")
        if brier is None:
            brier = report.get("raw_brier")
        quality = float(np.clip(baseline_brier / float(brier), 0.5, 1.5)) if brier else 1.0
        prior_hits = list((performance_history or {}).get(model) or [])
        if len(prior_hits) >= 5:
            expected_hits = (num_per_draw * num_per_draw) / max_num
            observed_hits = float(np.mean(prior_hits))
            quality *= float(np.clip(observed_hits / max(expected_hits, 1e-6), 0.5, 1.5))
        adjusted_weights[model] = base_weight * quality
    adjusted_total = sum(adjusted_weights.values()) or 1.0
    weights = {model: value / adjusted_total for model, value in adjusted_weights.items()}
    aggregate = {number: 0.0 for number in range(1, max_num + 1)}
    top_vote = {number: 0.0 for number in aggregate}
    member_votes = []
    for model, weight in weights.items():
        ranked = sorted(
            predictions.get(model) or [],
            key=lambda item: float(item.get("prob", 0.0)),
            reverse=True,
        )
        top_numbers = [int(item["number"]) for item in ranked[:num_per_draw]]
        top_scores = {str(int(item["number"])): float(item.get("prob", 0.0)) for item in ranked[:10]}
        for item in ranked:
            number = int(item["number"])
            if number in aggregate:
                aggregate[number] += weight * float(item.get("prob", 0.0))
        for number in top_numbers:
            top_vote[number] += weight
        member_votes.append({
            "model": model,
            "weight": round(weight, 6),
            "ticket": top_numbers,
            "top_scores": top_scores,
            "argument": (
                f"{model} chọn {'-'.join(f'{number:02d}' for number in top_numbers)} "
                f"theo xác suất đã calibration; trọng số biểu quyết {weight:.1%}; phản biện: đây chỉ là xếp hạng OOS "
                "tại thời điểm dự báo, không phải quan hệ nhân quả."
            ),
        })
    consensus_scores = [
        {
            "number": number,
            "prob": float(np.clip(0.75 * aggregate[number] + 0.25 * top_vote[number], 0.0, 1.0)),
            "weighted_probability": round(aggregate[number], 8),
            "weighted_vote": round(top_vote[number], 8),
        }
        for number in aggregate
    ]
    tickets = build_vietlott_coverage_tickets(
        consensus_scores,
        num_per_draw,
        max_num,
        ticket_count=ticket_count,
        candidate_pool_size=14 if max_num == 35 else 12,
        preferred_max_shared=2,
        relationship_scores=relationship_scores,
        pattern_profile={
            **(pattern_profile or {}),
            "penalty_scale": 3.0 if max_num == 35 else 1.0,
            "strict_profile": max_num == 35,
        },
    )
    tally = sorted(
        ({
            "number": number,
            "weighted_vote_pct": round(top_vote[number] * 100.0, 2),
            "consensus_score": round(
                0.75 * aggregate[number] + 0.25 * top_vote[number], 8
            ),
        } for number in aggregate),
        key=lambda item: (-item["consensus_score"], item["number"]),
    )
    return tickets, member_votes, tally


def _safe_isotonic_calibrator(scores, labels):
    """Fit a leakage-safe probability calibrator on prior OOS observations."""
    if len(scores) < 120 or len(set(labels)) < 2 or len(set(scores)) < 3:
        return None
    try:
        calibrator = IsotonicRegression(
            y_min=0.001,
            y_max=0.999,
            out_of_bounds="clip",
        )
        calibrator.fit(np.asarray(scores, dtype=float), np.asarray(labels, dtype=float))
        return calibrator
    except Exception as calibration_err:
        print(f"Vietlott calibration skipped: {calibration_err}")
        return None


def _brier_score(scores, labels):
    if not labels:
        return None
    values = np.asarray(scores, dtype=float)
    target = np.asarray(labels, dtype=float)
    return float(np.mean((values - target) ** 2))

# Generic draw prediction function
def predict_draw_game(
    draws,
    min_num,
    max_num,
    num_per_draw,
    result_extractor,
    training_draws=200,
    enabled_models=None,
    ranking_models=None,
    progress_callback=None,
    progress_start=0,
    progress_end=100,
    include_diagnostics=True,
    feature_cache=None,
    source_offset=0,
):
    """Train leakage-safe per-number models with draw-level pattern features.

    ``draws`` is newest-first. For a target at index ``t``, only indices greater
    than ``t`` are known at prediction time. This rule is applied to frequency,
    pair, calendar and draw-structure features alike.
    """
    if len(draws) < 120:
        return None

    def report_local_progress(stage, fraction):
        """Report a bounded sub-stage without making callers know internals."""
        if not callable(progress_callback):
            return
        fraction = max(0.0, min(1.0, float(fraction)))
        progress = progress_start + (progress_end - progress_start) * fraction
        try:
            progress_callback(stage, int(round(progress)))
        except Exception as progress_err:
            print(f"Vietlott prediction progress callback failed: {progress_err}")

    report_local_progress("Đang chuẩn bị dữ liệu huấn luyện", 0.02)

    all_numbers = list(range(min_num, max_num + 1))
    split_num = (max_num + min_num) // 2
    PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79}

    feature_names = [
        "last_seen", "f100", "f50", "f30", "f10", "f5",
        "is_even", "is_small", "tail_digit", "head_digit", "digit_sum", "is_prime",
        "avg_gap", "gap_ratio", "streak", "min_dist_to_prev",
        "prev_cooccur_sum", "prev_cooccur_max", "rank_30", "rank_10", "is_hot", "is_cold",
        "prev_pair_rate", "prev_pair_jaccard", "prev_pair_lift",
        "prev3_pair_rate", "top_partner_rate", "top_partner_jaccard",
        "same_weekday_rate", "same_weekday_lift", "same_month_rate", "same_month_lift",
        "target_weekday_sin", "target_weekday_cos", "target_month_sin", "target_month_cos",
        "target_day_ratio", "recent_odd_mean", "recent_small_mean", "recent_sum_mean",
        "recent_sum_std", "recent_range_mean", "recent_consecutive_mean", "recent_tail_diversity_mean",
        "recent_repeat_mean", "number_rate_300", "number_rate_30", "number_vs_mean_position",
        "latest_odd", "latest_small", "latest_sum", "latest_range", "latest_consecutive", "latest_tail_diversity",
        "trend_odd", "trend_small", "trend_sum", "trend_range", "trend_consecutive", "trend_tail_diversity",
        "weekday_odd_mean", "weekday_small_mean", "weekday_sum_mean",
        "month_odd_mean", "month_small_mean", "month_sum_mean",
        "prev_same_tail_count", "prev_same_decade_count", "prev_near_count", "prev_same_number",
        "prev3_triple_support", "days_since_last", "avg_days_gap", "day_gap_ratio",
        "weighted_rate_30", "rate_trend_10_vs_30", "number_position_norm",
    ]

    def clean_result(draw):
        values = []
        seen = set()
        for value in result_extractor(draw) or []:
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if min_num <= number <= max_num and number not in seen:
                values.append(number)
                seen.add(number)
        return values

    def parse_date(value):
        if isinstance(value, datetime):
            return value
        if not value:
            return None
        text = str(value).strip()[:10]
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None

    draw_results = [clean_result(draw) for draw in draws]
    draw_sets = [set(result) for result in draw_results]
    draw_dates = [parse_date(draw.get("date")) for draw in draws]
    report_local_progress("Đang chuẩn hóa dữ liệu kỳ quay", 0.12)

    def target_date(t_target):
        if 0 <= t_target < len(draw_dates) and draw_dates[t_target] is not None:
            return draw_dates[t_target]
        if t_target == -1 and draw_dates and draw_dates[0] is not None:
            # Infer the next scheduled weekday from the observed calendar.
            # This keeps date features useful for Wed/Fri/Sun-style products
            # without reading any future result.
            weekday_counts = {}
            for draw_dt in draw_dates:
                if draw_dt is not None:
                    weekday_counts[draw_dt.weekday()] = weekday_counts.get(draw_dt.weekday(), 0) + 1
            latest_weekday = draw_dates[0].weekday()
            for day_offset in range(1, 8):
                candidate_weekday = (latest_weekday + day_offset) % 7
                if candidate_weekday in weekday_counts:
                    return draw_dates[0] + timedelta(days=day_offset)
            return draw_dates[0] + timedelta(days=1)
        return None

    def draw_structure(result):
        ordered = sorted(set(result))
        return {
            "odd": sum(number % 2 for number in ordered),
            "small": sum(number <= split_num for number in ordered),
            "sum": sum(ordered),
            "range": (ordered[-1] - ordered[0]) if ordered else 0,
            "consecutive": sum(ordered[index + 1] - ordered[index] == 1 for index in range(len(ordered) - 1)),
            "tail_diversity": len({number % 10 for number in ordered}),
        }

    structures = [draw_structure(result) for result in draw_results]

    # Appearance and suffix co-occurrence are both indexed newest-first.
    appearance_indices = {number: [] for number in all_numbers}
    for index, result in enumerate(draw_results):
        for number in result:
            appearance_indices[number].append(index)
    appearance_indices_set = {number: set(indices) for number, indices in appearance_indices.items()}

    effective_train_draws = min(training_draws, len(draws) - 25)
    required_cooccurrence_targets = set(range(1, effective_train_draws + 1)) | {-1, 0}
    cooccurrence_by_target = {}
    suffix_cooccurrence = np.zeros((max_num + 1, max_num + 1), dtype=np.int32)
    for target_idx in range(len(draws) - 1, -2, -1):
        if target_idx in required_cooccurrence_targets:
            cooccurrence_by_target[target_idx] = suffix_cooccurrence.copy()
        if target_idx >= 0:
            result = draw_results[target_idx]
            for number_a in result:
                for number_b in result:
                    if number_a != number_b:
                        suffix_cooccurrence[number_a, number_b] += 1
    report_local_progress("Đang dựng quan hệ cặp số và ngữ cảnh", 0.24)

    context_cache = {}

    def build_context(t_target):
        if t_target in context_cache:
            return context_cache[t_target]

        known_start = max(t_target + 1, 0)
        known_indices = list(range(known_start, len(draws)))
        recent_indices = known_indices[:30]
        target_dt = target_date(t_target)
        target_weekday = target_dt.weekday() if target_dt else None
        target_month = target_dt.month if target_dt else None
        target_day = target_dt.day if target_dt else None

        f30_all = {
            number: sum(1 for index in appearance_indices[number] if t_target < index <= t_target + 30)
            for number in all_numbers
        }
        f10_all = {
            number: sum(1 for index in appearance_indices[number] if t_target < index <= t_target + 10)
            for number in all_numbers
        }
        sorted_by_f30 = sorted(all_numbers, key=lambda number: (-f30_all[number], number))
        sorted_by_f10 = sorted(all_numbers, key=lambda number: (-f10_all[number], number))
        known_count = max(len(known_indices), 1)
        recent_count = max(len(recent_indices), 1)
        number_counts = {
            number: sum(1 for index in known_indices if number in draw_sets[index])
            for number in all_numbers
        }
        number_counts_300 = {
            number: sum(1 for index in known_indices[:300] if number in draw_sets[index])
            for number in all_numbers
        }

        recent_structures = [structures[index] for index in recent_indices]
        def structure_mean(key):
            return float(np.mean([item[key] for item in recent_structures])) if recent_structures else 0.0
        def structure_trend(key):
            if len(recent_structures) < 2:
                return 0.0
            return (recent_structures[0][key] - recent_structures[-1][key]) / float(len(recent_structures) - 1)
        recent_sums = [item["sum"] for item in recent_structures]
        recent_repeat = []
        for index in recent_indices:
            if index + 1 < len(draw_sets) and index + 1 in known_indices:
                recent_repeat.append(len(draw_sets[index].intersection(draw_sets[index + 1])))

        weekday_indices = [
            index for index in known_indices
            if target_weekday is not None and draw_dates[index] is not None and draw_dates[index].weekday() == target_weekday
        ]
        month_indices = [
            index for index in known_indices
            if target_month is not None and draw_dates[index] is not None and draw_dates[index].month == target_month
        ]
        def conditional_structure_mean(indices, key):
            return float(np.mean([structures[index][key] for index in indices])) if indices else 0.0
        C_feat = cooccurrence_by_target.get(
            t_target,
            np.zeros((max_num + 1, max_num + 1), dtype=np.int32),
        )
        prev_draws = [
            draw_results[index]
            for index in range(max(t_target + 1, 0), min(t_target + 4, len(draw_results)))
        ]

        context = {
            "t": t_target,
            "known_indices": known_indices,
            "known_count": known_count,
            "recent_count": recent_count,
            "f30_all": f30_all,
            "f10_all": f10_all,
            "rank_30": {number: sorted_by_f30.index(number) + 1 for number in all_numbers},
            "rank_10": {number: sorted_by_f10.index(number) + 1 for number in all_numbers},
            "number_counts": number_counts,
            "number_counts_300": number_counts_300,
            "weekday_indices": weekday_indices,
            "month_indices": month_indices,
            "target_dt": target_dt,
            "target_weekday": target_weekday,
            "target_month": target_month,
            "target_day": target_day,
            "C": C_feat,
            "prev_draws": prev_draws,
            "recent_odd_mean": structure_mean("odd"),
            "recent_small_mean": structure_mean("small"),
            "recent_sum_mean": float(np.mean(recent_sums)) if recent_sums else 0.0,
            "recent_sum_std": float(np.std(recent_sums)) if recent_sums else 0.0,
            "recent_range_mean": structure_mean("range"),
            "recent_consecutive_mean": structure_mean("consecutive"),
            "recent_tail_diversity_mean": structure_mean("tail_diversity"),
            "recent_repeat_mean": float(np.mean(recent_repeat)) if recent_repeat else 0.0,
            "latest_structure": recent_structures[0] if recent_structures else {},
            "latest_odd": recent_structures[0]["odd"] if recent_structures else 0.0,
            "latest_small": recent_structures[0]["small"] if recent_structures else 0.0,
            "latest_sum": recent_structures[0]["sum"] if recent_structures else 0.0,
            "latest_range": recent_structures[0]["range"] if recent_structures else 0.0,
            "latest_consecutive": recent_structures[0]["consecutive"] if recent_structures else 0.0,
            "latest_tail_diversity": recent_structures[0]["tail_diversity"] if recent_structures else 0.0,
            "trend_odd": structure_trend("odd"),
            "trend_small": structure_trend("small"),
            "trend_sum": structure_trend("sum"),
            "trend_range": structure_trend("range"),
            "trend_consecutive": structure_trend("consecutive"),
            "trend_tail_diversity": structure_trend("tail_diversity"),
            "weekday_odd_mean": conditional_structure_mean(weekday_indices, "odd"),
            "weekday_small_mean": conditional_structure_mean(weekday_indices, "small"),
            "weekday_sum_mean": conditional_structure_mean(weekday_indices, "sum"),
            "month_odd_mean": conditional_structure_mean(month_indices, "odd"),
            "month_small_mean": conditional_structure_mean(month_indices, "small"),
            "month_sum_mean": conditional_structure_mean(month_indices, "sum"),
        }
        context_cache[t_target] = context
        return context

    def pair_metrics(number, partner_draws, context):
        if not partner_draws:
            return 0.0, 0.0, 0.0, 0, 0.0
        prior_rate = num_per_draw / max_num
        number_count = context["number_counts"].get(number, 0)
        number_rate = number_count / context["known_count"]
        conditional = []
        jaccards = []
        lifts = []
        raw_counts = []
        for partner in sorted({value for result in partner_draws for value in result if value != number}):
            partner_count = context["number_counts"].get(partner, 0)
            co_count = int(context["C"][number, partner])
            if partner_count <= 0:
                continue
            # Smooth sparse pair counts toward the game-wide prior. This keeps
            # one accidental pair from dominating a feature or ticket score.
            cond = (
                co_count + prior_rate * RELATION_PRIOR_STRENGTH
            ) / (partner_count + RELATION_PRIOR_STRENGTH)
            union = number_count + partner_count - co_count
            jaccard = co_count / union if union > 0 else 0.0
            conditional.append(cond)
            jaccards.append(jaccard)
            lifts.append(cond / max(number_rate, 1.0 / context["known_count"]))
            raw_counts.append(co_count)
        if not conditional:
            return 0.0, 0.0, 0.0, 0, 0.0
        return (
            float(np.mean(conditional)),
            float(np.mean(jaccards)),
            float(np.mean(lifts)),
            max(raw_counts),
            max(jaccards),
        )

    def top_partner_metrics(number, context):
        number_count = context["number_counts"].get(number, 0)
        prior_rate = num_per_draw / max_num
        best_rate = 0.0
        best_jaccard = 0.0
        for partner in all_numbers:
            if partner == number:
                continue
            partner_count = context["number_counts"].get(partner, 0)
            co_count = int(context["C"][number, partner])
            if partner_count <= 0:
                continue
            cond = (
                co_count + prior_rate * RELATION_PRIOR_STRENGTH
            ) / (partner_count + RELATION_PRIOR_STRENGTH)
            union = number_count + partner_count - co_count
            jaccard = co_count / union if union > 0 else 0.0
            best_rate = max(best_rate, cond)
            best_jaccard = max(best_jaccard, jaccard)
        return best_rate, best_jaccard

    def make_feature_row(t_target, number):
        context = build_context(t_target)
        indices_past = [index for index in appearance_indices[number] if index > t_target]
        raw_last_seen = indices_past[0] - t_target - 1 if indices_past else len(draws) - t_target
        f100 = sum(1 for index in indices_past if index <= t_target + 100)
        f50 = sum(1 for index in indices_past if index <= t_target + 50)
        f30 = context["f30_all"][number]
        f10 = context["f10_all"][number]
        f5 = sum(1 for index in indices_past if index <= t_target + 5)

        gaps = [indices_past[index + 1] - indices_past[index] for index in range(len(indices_past) - 1)]
        avg_gap = float(np.mean(gaps)) if gaps else float(max_num / num_per_draw)
        # Compress long absences so they cannot dominate tree splits or
        # linear/MLP weights. Frequency, pairs and draw structure remain the
        # primary signals; absence is only a bounded supporting feature.
        last_seen = math.log1p(max(raw_last_seen, 0))
        gap_ratio = math.tanh((raw_last_seen / (avg_gap + 1.0)) / RECENCY_TRANSFORM_SCALE)
        streak = 0
        while t_target + 1 + streak in appearance_indices_set[number]:
            streak += 1

        prev_draw = context["prev_draws"][0] if context["prev_draws"] else []
        prev3_rate, prev3_jaccard, prev3_lift, prev3_raw_max, prev3_jaccard_max = pair_metrics(
            number, context["prev_draws"], context
        )
        prev_rate, prev_jaccard, prev_lift, prev_raw_max, prev_jaccard_max = pair_metrics(
            number, [prev_draw], context
        )
        best_partner_rate, best_partner_jaccard = top_partner_metrics(number, context)
        weekday_count = len(context["weekday_indices"])
        month_count = len(context["month_indices"])
        weekday_hits = sum(number in draw_sets[index] for index in context["weekday_indices"])
        month_hits = sum(number in draw_sets[index] for index in context["month_indices"])
        weekday_rate = weekday_hits / max(weekday_count, 1)
        month_rate = month_hits / max(month_count, 1)
        base_rate = context["number_counts"][number] / context["known_count"]
        expected_number = context["recent_sum_mean"] / max(num_per_draw, 1)
        target_dt = context["target_dt"]
        prev_draw_set = set(prev_draw)
        previous_overlap_draws = [
            index for index in context["known_indices"]
            if len(draw_sets[index].intersection(prev_draw_set)) >= 2
        ] if prev_draw_set else []
        triple_support = (
            (
                sum(number in draw_sets[index] for index in previous_overlap_draws)
                + (num_per_draw / max_num) * RELATION_PRIOR_STRENGTH
            ) / (len(previous_overlap_draws) + RELATION_PRIOR_STRENGTH)
        ) if previous_overlap_draws else 0.0
        weighted_window = context["known_indices"][:30]
        weighted_values = [
            np.exp(-position / 10.0) for position in range(len(weighted_window))
            if number in draw_sets[weighted_window[position]]
        ]
        weighted_rate_30 = sum(weighted_values) / max(sum(np.exp(-position / 10.0) for position in range(len(weighted_window))), 1.0)
        last_date = draw_dates[indices_past[0]] if indices_past else None
        raw_days_since_last = (target_dt - last_date).days if target_dt and last_date else float(raw_last_seen)
        days_since_last = math.log1p(max(raw_days_since_last, 0))
        day_gaps = [
            abs((draw_dates[indices_past[index]] - draw_dates[indices_past[index + 1]]).days)
            for index in range(len(indices_past) - 1)
            if draw_dates[indices_past[index]] is not None and draw_dates[indices_past[index + 1]] is not None
        ]
        avg_days_gap = float(np.mean(day_gaps)) if day_gaps else 1.0
        weekday_angle = 2.0 * np.pi * target_dt.weekday() / 7.0 if target_dt else 0.0
        month_angle = 2.0 * np.pi * ((target_dt.month - 1) / 12.0) if target_dt else 0.0

        return [
            last_seen, f100, f50, f30, f10, f5,
            int(number % 2 == 0), int(number <= split_num), number % 10, number // 10,
            (number % 10) + (number // 10), int(number in PRIMES),
            avg_gap, gap_ratio, streak,
            min((abs(number - partner) for partner in prev_draw), default=max_num - min_num),
            sum(int(context["C"][number, partner]) for partner in prev_draw),
            max((int(context["C"][number, partner]) for partner in prev_draw), default=0),
            context["rank_30"][number], context["rank_10"][number], int(f10 >= 2), int(raw_last_seen > 30),
            prev_rate, prev_jaccard, prev_lift,
            prev3_rate, best_partner_rate, best_partner_jaccard,
            weekday_rate, weekday_rate / max(base_rate, 1.0 / context["known_count"]),
            month_rate, month_rate / max(base_rate, 1.0 / context["known_count"]),
            np.sin(weekday_angle), np.cos(weekday_angle), np.sin(month_angle), np.cos(month_angle),
            (target_dt.day / 31.0) if target_dt else 0.0,
            context["recent_odd_mean"], context["recent_small_mean"], context["recent_sum_mean"],
            context["recent_sum_std"], context["recent_range_mean"], context["recent_consecutive_mean"],
            context["recent_tail_diversity_mean"], context["recent_repeat_mean"],
            context["number_counts_300"][number] / min(context["known_count"], 300),
            f30 / context["recent_count"],
            number / max(expected_number, 1.0),
            context["latest_odd"], context["latest_small"], context["latest_sum"], context["latest_range"],
            context["latest_consecutive"], context["latest_tail_diversity"],
            context["trend_odd"], context["trend_small"], context["trend_sum"], context["trend_range"],
            context["trend_consecutive"], context["trend_tail_diversity"],
            context["weekday_odd_mean"], context["weekday_small_mean"], context["weekday_sum_mean"],
            context["month_odd_mean"], context["month_small_mean"], context["month_sum_mean"],
            sum(number % 10 == partner % 10 for partner in prev_draw),
            sum(number // 10 == partner // 10 for partner in prev_draw),
            sum(abs(number - partner) <= 2 for partner in prev_draw),
            int(number in prev_draw_set),
            triple_support, days_since_last, avg_days_gap,
            math.tanh((raw_days_since_last / max(avg_days_gap, 1.0)) / RECENCY_TRANSFORM_SCALE),
            weighted_rate_30, (f10 / 10.0) - (f30 / 30.0),
            (number - min_num) / max(max_num - min_num, 1),
        ]

    def get_feature_rows(t_target):
        """Return one target's feature matrix, reusing walk-forward-safe rows.

        A backtest at ``source_offset`` sees exactly the same historical
        information as the corresponding target in the original newest-first
        dataset. Reusing only non-negative target rows therefore avoids
        rebuilding the same feature matrix dozens of times without changing
        the leakage boundary. The synthetic next-draw target (-1) is never
        shared because its calendar context is slice-specific.
        """
        cache_key = None
        if feature_cache is not None and t_target >= 0:
            cache_key = int(source_offset + t_target)
            cached = feature_cache.get(cache_key)
            if cached is not None:
                return cached
        rows = np.asarray([make_feature_row(t_target, number) for number in all_numbers], dtype=float)
        if cache_key is not None:
            feature_cache[cache_key] = rows
        return rows

    def get_features_labels(start_t, end_t):
        X, y = [], []
        end = min(end_t, len(draws))
        total = max(end - start_t, 1)
        update_every = max(1, total // 8)
        for row_index, t in enumerate(range(start_t, end), start=1):
            target_res = set(draw_results[t])
            feature_rows = get_feature_rows(t)
            for number_index, number in enumerate(all_numbers):
                X.append(feature_rows[number_index])
                y.append(int(number in target_res))
            if row_index == 1 or row_index == total or row_index % update_every == 0:
                report_local_progress(
                    f"Đang tạo feature ({row_index}/{total} kỳ)",
                    0.24 + 0.42 * (row_index / total),
                )
        return np.asarray(X, dtype=float), np.asarray(y, dtype=int)

    def get_features_pred(t_target):
        return get_feature_rows(t_target)

    def get_target_pattern(t_target):
        """Return a low-weight, data-derived profile for this target draw."""
        context = build_context(t_target)
        return {
            "pred_odd": round(context["recent_odd_mean"], 2),
            "pred_small": round(context["recent_small_mean"], 2),
            "pred_sum": round(context["recent_sum_mean"], 2),
            "pred_range": round(context["recent_range_mean"], 2),
            "pred_consecutive": round(context["recent_consecutive_mean"], 2),
            "pred_tail_diversity": round(context["recent_tail_diversity_mean"], 2),
        }

    def get_relationship_scores(t_target):
        """Build smoothed pair/triple signals for the ticket optimizer.

        Pair/triple counts are calculated only from draws known at the target
        time. A Bayesian prior prevents rare combinations from being treated
        as reliable evidence.
        """
        context = build_context(t_target)
        known_indices = context["known_indices"][:300]
        known_count = max(len(known_indices), 1)
        pair_prior = (num_per_draw * (num_per_draw - 1)) / max(max_num * (max_num - 1), 1)
        triple_prior = (
            math.comb(num_per_draw, 3) / max(math.comb(max_num, 3), 1)
            if num_per_draw >= 3 else 0.0
        )
        pair_scores = {}
        for first, second in combinations(all_numbers, 2):
            pair_count = int(context["C"][first, second])
            observed = (
                pair_count + pair_prior * RELATION_PRIOR_STRENGTH
            ) / (context["known_count"] + RELATION_PRIOR_STRENGTH)
            # Store a bounded relative signal. 0.25 is neutral; higher values
            # indicate above-prior support and lower values indicate caution.
            pair_scores[(first, second)] = float(
                np.clip(observed / max(pair_prior, 1e-9) / 4.0, 0.05, 1.0)
            )

        triple_counts = {}
        for index in known_indices:
            for triple in combinations(sorted(draw_sets[index]), 3):
                triple_counts[triple] = triple_counts.get(triple, 0) + 1
        triple_scores = {}
        for triple, count in triple_counts.items():
            observed = (
                count + triple_prior * RELATION_PRIOR_STRENGTH
            ) / (known_count + RELATION_PRIOR_STRENGTH)
            triple_scores[triple] = float(
                np.clip(observed / max(triple_prior, 1e-9) / 4.0, 0.02, 1.0)
            )
        return {
            "pair_scores": pair_scores,
            "triple_scores": triple_scores,
            "pair_neutral": 0.25,
            "triple_neutral": 0.25,
        }

    X_train, y_train = get_features_labels(1, effective_train_draws + 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    report_local_progress("Đã tạo xong feature, chuẩn bị huấn luyện model", 0.68)
    models = {
        "linear_regression": LogisticRegression(C=0.1, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=80, max_depth=6, min_samples_split=4, class_weight="balanced", random_state=42, n_jobs=-1),
        # Histogram boosting keeps the same probability-based role in the
        # ensemble but is substantially faster than the classic Python-tree
        # GradientBoostingClassifier on this repeated walk-forward workload.
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=60,
            max_depth=3,
            max_leaf_nodes=7,
            min_samples_leaf=20,
            learning_rate=0.04,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=False,
            random_state=42,
        ),
        "mlp": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=140, early_stopping=True, random_state=42),
    }
    ranker_names = set()

    active_ranking_models = (
        tuple(ranking_models)
        if ranking_models is not None
        else ("xgb_ranker",) if xgb else ()
    )
    for ranking_model in active_ranking_models:
        if ranking_model == "xgb_ranker" and xgb:
            models[ranking_model] = xgb.XGBRanker(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.03,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=5.0,
                objective="rank:ndcg",
                eval_metric="ndcg@5",
                random_state=42,
                n_jobs=1,
            )
            ranker_names.add(ranking_model)
        elif ranking_model == "catboost_ranker":
            try:
                from catboost import CatBoostRanker
            except ImportError:
                continue
            models[ranking_model] = CatBoostRanker(
                iterations=80,
                depth=4,
                learning_rate=0.04,
                l2_leaf_reg=8.0,
                loss_function="YetiRankPairwise",
                random_seed=42,
                thread_count=1,
                verbose=False,
                allow_writing_files=False,
            )
            ranker_names.add(ranking_model)
        elif ranking_model == "lightgbm_ranker":
            try:
                from lightgbm import LGBMRanker
            except ImportError:
                continue
            models[ranking_model] = LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                n_estimators=80,
                num_leaves=7,
                max_depth=4,
                learning_rate=0.03,
                min_child_samples=50,
                reg_lambda=5.0,
                verbosity=-1,
                random_state=42,
                n_jobs=1,
            )
            ranker_names.add(ranking_model)
    if xgb:
        models["xgboost"] = xgb.XGBClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.04,
            scale_pos_weight=float((max_num - num_per_draw) / num_per_draw),
            random_state=42, eval_metric="logloss", n_jobs=1,
        )

    # Offline walk-forward tuning can train only selected model families. The
    # production path leaves this unset and therefore keeps the full ensemble.
    if enabled_models is not None:
        requested_models = set(enabled_models)
        models = {
            name: model
            for name, model in models.items()
            if name in requested_models
        }
        if not models:
            raise ValueError(
                "Không có mô hình hợp lệ trong enabled_models: "
                f"{sorted(requested_models)}"
            )

    if len(np.unique(y_train)) < 2:
        class DummyModel:
            def fit(self, X, y):
                return self
            def predict(self, X):
                return np.zeros(len(X))
            def predict_proba(self, X):
                return np.column_stack([np.ones(len(X)), np.zeros(len(X))])
        for name in models:
            models[name] = DummyModel()
    else:
        neg_count = max(int(np.sum(y_train == 0)), 1)
        pos_count = max(int(np.sum(y_train == 1)), 1)
        w_neg = len(y_train) / (2.0 * neg_count)
        w_pos = len(y_train) / (2.0 * pos_count)
        sample_weight = np.asarray([w_pos if value else w_neg for value in y_train])
        ranking_groups = [len(all_numbers)] * effective_train_draws
        catboost_group_id = np.repeat(np.arange(effective_train_draws), len(all_numbers))
        model_count = max(len(models), 1)
        for model_index, (name, model) in enumerate(models.items(), start=1):
            model_fraction = 0.70 + 0.25 * ((model_index - 1) / model_count)
            report_local_progress(
                f"Đang huấn luyện model {name}",
                model_fraction,
            )
            if name in ranker_names:
                if name == "catboost_ranker":
                    from catboost import Pool
                    model.fit(
                        Pool(X_train_scaled, label=y_train, group_id=catboost_group_id)
                    )
                else:
                    model.fit(X_train_scaled, y_train, group=ranking_groups)
            elif name == "mlp":
                try:
                    model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
                except TypeError as exc:
                    if "sample_weight" not in str(exc):
                        raise
                    rng = np.random.RandomState(42)
                    class_indices = [np.flatnonzero(y_train == value) for value in (0, 1)]
                    target_size = max(len(indices) for indices in class_indices)
                    balanced_indices = np.concatenate([
                        rng.choice(indices, size=target_size, replace=True)
                        for indices in class_indices if len(indices) > 0
                    ])
                    model.fit(X_train_scaled[balanced_indices], y_train[balanced_indices])
            else:
                model.fit(X_train_scaled, y_train)
            report_local_progress(
                f"Đã huấn luyện model {name}",
                0.70 + 0.25 * (model_index / model_count),
            )

    report_local_progress("Đang tổng hợp xác suất dự báo", 0.96)
    def get_model_predictions(t_target, X_pred=None):
        if X_pred is None:
            X_pred = get_features_pred(t_target)
        X_pred_scaled = scaler.transform(X_pred)
        preds = {}
        for name, model in models.items():
            if name in ranker_names:
                raw_scores = np.asarray(model.predict(X_pred_scaled), dtype=float).reshape(-1)
                probs = (np.argsort(np.argsort(raw_scores)) + 1) / len(all_numbers)
            else:
                probabilities = model.predict_proba(X_pred_scaled)
                if probabilities.shape[1] == 1:
                    only_class = int(getattr(model, "classes_", [0])[0])
                    probs = probabilities[:, 0] if only_class == 1 else np.zeros(len(all_numbers))
                else:
                    classes = list(getattr(model, "classes_", [0, 1]))
                    positive_index = classes.index(1) if 1 in classes else min(1, probabilities.shape[1] - 1)
                    probs = probabilities[:, positive_index]
            preds[name] = [
                {"number": number, "prob": float(np.clip(probabilities_value, 0.0, 1.0))}
                for number, probabilities_value in zip(all_numbers, probs)
            ]

        active_models = list(preds)
        if "xgb_ranker" in active_models:
            weights = {
                "xgb_ranker": 0.25,
                "random_forest": 0.20,
                "xgboost": 0.20,
                "gradient_boosting": 0.15,
                "mlp": 0.12,
                "linear_regression": 0.08,
            }
        elif "xgboost" in active_models:
            weights = {
                "random_forest": 0.25,
                "xgboost": 0.25,
                "gradient_boosting": 0.25,
                "mlp": 0.15,
                "linear_regression": 0.10,
            }
        else:
            weights = {
                "random_forest": 0.35,
                "gradient_boosting": 0.35,
                "mlp": 0.20,
                "linear_regression": 0.10,
            }
        active_weight_sum = sum(weights.get(name, 0.0) for name in active_models)
        if active_weight_sum <= 0:
            active_weight_sum = float(max(len(active_models), 1))
        ensemble_raw = np.zeros(len(all_numbers))
        vote_counts = np.zeros(len(all_numbers), dtype=np.int32)
        for name in active_models:
            model_probs = np.asarray([item["prob"] for item in preds[name]])
            model_weight = weights.get(name, 1.0 / max(len(active_models), 1))
            if weights:
                model_weight /= active_weight_sum
            ensemble_raw += model_weight * model_probs
            vote_counts[np.argsort(model_probs)[-num_per_draw:]] += 1
        ensemble_rank = (np.argsort(np.argsort(ensemble_raw)) + 1) / len(all_numbers)
        consensus_scores = 0.60 * ensemble_rank + 0.40 * (vote_counts / max(len(active_models), 1))
        preds["ensemble"] = [
            {"number": number, "prob": float(consensus_scores[index]), "raw_prob": float(ensemble_raw[index]), "model_votes": int(vote_counts[index])}
            for index, number in enumerate(all_numbers)
        ]
        return preds

    back_preds = get_model_predictions(0)
    # Walk-forward evaluation only needs predictions for the held-out draw.
    # Keep the target's lightweight pattern/relationship evidence because the
    # roundtable ticket chair needs it; skip only the expensive next-draw
    # matrix and feature-importance diagnostics.
    if not include_diagnostics:
        return {
            "back": back_preds,
            "back_relationships": get_relationship_scores(0),
            "target_pattern": get_target_pattern(0),
        }

    next_X_pred = get_features_pred(-1)
    next_preds = get_model_predictions(-1, next_X_pred)
    next_features = {}
    integer_features = {"last_seen", "f100", "f50", "f30", "f10", "f5", "is_even", "is_small", "tail_digit", "head_digit", "digit_sum", "is_prime", "streak", "min_dist_to_prev", "prev_cooccur_sum", "prev_cooccur_max", "rank_30", "rank_10", "is_hot", "is_cold"}
    next_context = build_context(-1)
    for index, number in enumerate(all_numbers):
        values = {}
        for name, value in zip(feature_names, next_X_pred[index]):
            values[name] = int(round(value)) if name in integer_features else float(round(value, 4))
        values["weekday_sample_count"] = len(next_context["weekday_indices"])
        values["month_sample_count"] = len(next_context["month_indices"])
        values["previous_draw"] = list(next_context["prev_draws"][0]) if next_context["prev_draws"] else []
        next_features[str(number)] = values

    recent_pairs = []
    pair_matrix = next_context["C"]
    for first in all_numbers:
        for second in all_numbers:
            if first < second and pair_matrix[first, second] > 0:
                recent_pairs.append({"numbers": [first, second], "count": int(pair_matrix[first, second])})
    recent_pairs.sort(key=lambda item: (-item["count"], item["numbers"]))
    weekday_names = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"]
    target_dt = next_context["target_dt"]
    next_pattern = {
        "target_date": target_dt.strftime("%Y-%m-%d") if target_dt else None,
        "target_weekday": weekday_names[target_dt.weekday()] if target_dt else None,
        "known_draws": len(next_context["known_indices"]),
        "recent_draws": next_context["recent_count"],
        "previous_draw": list(next_context["prev_draws"][0]) if next_context["prev_draws"] else [],
        "sum_mean": round(next_context["recent_sum_mean"], 2),
        "sum_std": round(next_context["recent_sum_std"], 2),
        "odd_mean": round(next_context["recent_odd_mean"], 2),
        "even_mean": round(num_per_draw - next_context["recent_odd_mean"], 2),
        "small_mean": round(next_context["recent_small_mean"], 2),
        "large_mean": round(num_per_draw - next_context["recent_small_mean"], 2),
        "range_mean": round(next_context["recent_range_mean"], 2),
        "consecutive_mean": round(next_context["recent_consecutive_mean"], 2),
        "tail_diversity_mean": round(next_context["recent_tail_diversity_mean"], 2),
        "repeat_mean": round(next_context["recent_repeat_mean"], 2),
        "latest_draw_structure": {
            "odd": next_context["latest_odd"],
            "small": next_context["latest_small"],
            "sum": next_context["latest_sum"],
            "range": next_context["latest_range"],
            "consecutive": next_context["latest_consecutive"],
            "tail_diversity": next_context["latest_tail_diversity"],
        },
        "weekday_pattern": {
            "odd_mean": round(next_context["weekday_odd_mean"], 2),
            "small_mean": round(next_context["weekday_small_mean"], 2),
            "sum_mean": round(next_context["weekday_sum_mean"], 2),
        },
        "month_pattern": {
            "odd_mean": round(next_context["month_odd_mean"], 2),
            "small_mean": round(next_context["month_small_mean"], 2),
            "sum_mean": round(next_context["month_sum_mean"], 2),
        },
        "top_pairs": recent_pairs[:10],
    }
    feature_groups = {
        "recency_frequency": ["last_seen", "f100", "f50", "f30", "f10", "f5", "avg_gap", "gap_ratio", "streak", "rank_30", "rank_10"],
        "number_attributes": ["is_even", "is_small", "tail_digit", "head_digit", "digit_sum", "is_prime"],
        "draw_structure": ["recent_odd_mean", "recent_small_mean", "recent_sum_mean", "recent_sum_std", "recent_range_mean", "recent_consecutive_mean", "recent_tail_diversity_mean", "recent_repeat_mean", "latest_odd", "latest_small", "latest_sum", "latest_range", "latest_consecutive", "latest_tail_diversity", "trend_odd", "trend_small", "trend_sum", "trend_range", "trend_consecutive", "trend_tail_diversity"],
        "relationships": ["prev_cooccur_sum", "prev_cooccur_max", "prev_pair_rate", "prev_pair_jaccard", "prev_pair_lift", "prev3_pair_rate", "top_partner_rate", "top_partner_jaccard", "prev_same_tail_count", "prev_same_decade_count", "prev_near_count", "prev_same_number", "prev3_triple_support"],
        "calendar": ["same_weekday_rate", "same_weekday_lift", "same_month_rate", "same_month_lift", "weekday_odd_mean", "weekday_small_mean", "weekday_sum_mean", "month_odd_mean", "month_small_mean", "month_sum_mean", "target_weekday_sin", "target_weekday_cos", "target_month_sin", "target_month_cos", "target_day_ratio"],
        "time_distance_and_position": ["days_since_last", "avg_days_gap", "day_gap_ratio", "weighted_rate_30", "rate_trend_10_vs_30", "number_position_norm", "number_vs_mean_position"],
    }
    feature_importance = {}
    group_lookup = {
        feature: group
        for group, features in feature_groups.items()
        for feature in features
    }
    group_scores = {group: [] for group in feature_groups}
    for model_name in ("random_forest", "gradient_boosting", "xgboost", "xgb_ranker"):
        model = models.get(model_name)
        importances = getattr(model, "feature_importances_", None)
        if importances is None or len(importances) != len(feature_names):
            continue
        total_importance = float(np.sum(importances)) or 1.0
        ranked = sorted(
            ((feature_names[index], float(value) / total_importance) for index, value in enumerate(importances)),
            key=lambda item: item[1],
            reverse=True,
        )
        feature_importance[model_name] = [
            {"feature": feature, "importance_pct": round(value * 100.0, 2)}
            for feature, value in ranked[:15]
        ]
        model_group_scores = {group: 0.0 for group in feature_groups}
        for feature, value in ranked:
            group = group_lookup.get(feature)
            if group:
                model_group_scores[group] += value
        for group, value in model_group_scores.items():
            group_scores[group].append(value)
    feature_importance["group_importance_pct"] = {
        group: round(float(np.mean(values)) * 100.0, 2) if values else 0.0
        for group, values in group_scores.items()
    }
    return {
        "next": next_preds,
        "back": back_preds,
        "next_relationships": get_relationship_scores(-1),
        "back_relationships": get_relationship_scores(0),
        "target_pattern": get_target_pattern(0),
        "next_features": next_features,
        "feature_names": feature_names,
        "feature_groups": feature_groups,
        "feature_importance": feature_importance,
        "next_pattern": next_pattern,
    }

def predict_draw_profile(draws, num_per_draw, split_num):
    """Predict draw-level structure from prior draw structure and calendar.

    This model is separate from the per-number classifiers because sum,
    odd/even and small/large are properties of the complete ticket. Its
    features use only the five older draws for every training target.
    """
    from sklearn.ensemble import RandomForestRegressor

    def parse_date(value):
        if isinstance(value, datetime):
            return value
        if not value:
            return None
        try:
            return datetime.strptime(str(value)[:10], "%Y-%m-%d")
        except ValueError:
            return None

    def structure(draw):
        result = sorted(set(draw.get("result", [])[:num_per_draw]))
        return {
            "odd": sum(number % 2 for number in result),
            "small": sum(number <= split_num for number in result),
            "sum": sum(result),
            "range": result[-1] - result[0] if result else 0,
            "consecutive": sum(result[index + 1] - result[index] == 1 for index in range(len(result) - 1)),
            "tails": len({number % 10 for number in result}),
        }

    def date_features(target_index):
        if 0 <= target_index < len(draws):
            target_dt = parse_date(draws[target_index].get("date"))
        elif target_index == -1 and draws:
            latest = parse_date(draws[0].get("date"))
            if latest:
                weekday_counts = {}
                for draw in draws:
                    draw_dt = parse_date(draw.get("date"))
                    if draw_dt:
                        weekday_counts[draw_dt.weekday()] = weekday_counts.get(draw_dt.weekday(), 0) + 1
                target_dt = None
                for day_offset in range(1, 8):
                    if (latest.weekday() + day_offset) % 7 in weekday_counts:
                        target_dt = latest + timedelta(days=day_offset)
                        break
                target_dt = target_dt or (latest + timedelta(days=1))
            else:
                target_dt = None
        else:
            target_dt = None
        if not target_dt:
            return [0.0] * 5
        weekday_angle = 2.0 * np.pi * target_dt.weekday() / 7.0
        month_angle = 2.0 * np.pi * (target_dt.month - 1) / 12.0
        return [
            float(np.sin(weekday_angle)), float(np.cos(weekday_angle)),
            float(np.sin(month_angle)), float(np.cos(month_angle)),
            target_dt.day / 31.0,
        ]

    def row(target_index):
        values = []
        for index in range(target_index + 1, target_index + 6):
            if index < len(draws):
                item = structure(draws[index])
                values.extend([item["odd"], item["small"], item["sum"], item["range"], item["consecutive"], item["tails"]])
            else:
                values.extend([num_per_draw / 2.0, num_per_draw / 2.0, num_per_draw * split_num, split_num, 0.0, num_per_draw])
        values.extend(date_features(target_index))
        return values

    fallback = {
        "pred_odd": num_per_draw // 2,
        "pred_even": num_per_draw - (num_per_draw // 2),
        "pred_small": num_per_draw // 2,
        "pred_large": num_per_draw - (num_per_draw // 2),
        "pred_sum": int(num_per_draw * split_num),
        "pred_range": round(float(split_num * 1.5), 1),
        "pred_consecutive": 0,
        "pred_tail_diversity": num_per_draw,
        "odd_confidence": 50.0,
        "small_confidence": 50.0,
    }
    if len(draws) < 100:
        return fallback

    X = []
    y_odd, y_small, y_sum, y_range, y_consecutive, y_tails = [], [], [], [], [], []
    for target_index in range(len(draws) - 6, -1, -1):
        target = structure(draws[target_index])
        X.append(row(target_index))
        y_odd.append(target["odd"])
        y_small.append(target["small"])
        y_sum.append(target["sum"])
        y_range.append(target["range"])
        y_consecutive.append(target["consecutive"])
        y_tails.append(target["tails"])

    clf_odd = RandomForestClassifier(n_estimators=80, max_depth=5, random_state=42, n_jobs=-1)
    clf_small = RandomForestClassifier(n_estimators=80, max_depth=5, random_state=42, n_jobs=-1)
    clf_consecutive = RandomForestClassifier(n_estimators=80, max_depth=5, random_state=42, n_jobs=-1)
    clf_tails = RandomForestClassifier(n_estimators=80, max_depth=5, random_state=42, n_jobs=-1)
    reg_sum = RandomForestRegressor(n_estimators=80, max_depth=5, random_state=42, n_jobs=-1)
    reg_range = RandomForestRegressor(n_estimators=80, max_depth=5, random_state=42, n_jobs=-1)
    for model, target in ((clf_odd, y_odd), (clf_small, y_small), (clf_consecutive, y_consecutive), (clf_tails, y_tails)):
        model.fit(X, target)
    reg_sum.fit(X, y_sum)
    reg_range.fit(X, y_range)

    next_row = [row(-1)]
    pred_odd = int(clf_odd.predict(next_row)[0])
    pred_small = int(clf_small.predict(next_row)[0])
    pred_consecutive = int(clf_consecutive.predict(next_row)[0])
    pred_tails = int(clf_tails.predict(next_row)[0])
    pred_sum = float(reg_sum.predict(next_row)[0])
    pred_range = float(reg_range.predict(next_row)[0])

    def class_confidence(model, prediction):
        try:
            index = list(model.classes_).index(prediction)
            return round(float(model.predict_proba(next_row)[0][index]) * 100.0, 1)
        except Exception:
            return 50.0

    return {
        "pred_odd": pred_odd,
        "pred_even": num_per_draw - pred_odd,
        "pred_small": pred_small,
        "pred_large": num_per_draw - pred_small,
        "pred_sum": round(pred_sum, 1),
        "pred_range": round(pred_range, 1),
        "pred_consecutive": pred_consecutive,
        "pred_tail_diversity": pred_tails,
        "odd_confidence": class_confidence(clf_odd, pred_odd),
        "small_confidence": class_confidence(clf_small, pred_small),
    }

def run_monte_carlo_filter(product_id, ensemble_probabilities, profile, num_per_draw, max_num, split_num, draws, seed=None):
    import random
    import math
    from itertools import combinations

    # Monte Carlo remains a sampling step, but callers can provide a stable
    # seed so the same input data produces the same cached recommendation.
    rng = random.Random(seed)

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
    pred_range = profile.get("pred_range")
    pred_consecutive = profile.get("pred_consecutive")
    pred_tail_diversity = profile.get("pred_tail_diversity")
    
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
            n1 = rng.choices(available_nums, weights=available_weights, k=1)[0]
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
                    
                chosen = rng.choices(available_nums, weights=dyn_weights, k=1)[0]
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

            if pred_range is not None:
                range_dev = abs(max(ticket) - min(ticket) - pred_range)
                if range_dev > 12:
                    profile_penalty += (range_dev - 12) * 0.025

            if pred_consecutive is not None:
                profile_penalty += abs(consec - pred_consecutive) * 0.25

            if pred_tail_diversity is not None:
                tail_diversity = len({number % 10 for number in ticket})
                profile_penalty += abs(tail_diversity - pred_tail_diversity) * 0.18
                
            score -= profile_penalty
            
            local_qualified.append({
                "ticket": ticket,
                "score": score
            })
            seen_tickets.add(ticket_tuple)
            if len(local_qualified) >= 1000:
                break
        return local_qualified

    # Generate 40,000 samples and pick top candidates using soft constraints.
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

def rank_special_numbers_535(draws):
    """Rank all 1-12 special balls using only the supplied historical draws."""
    try:
        if not draws:
            return list(range(1, 13))
            
        special_numbers = []
        for d in draws:
            res = d.get("result", [])
            if len(res) >= 6:
                special_numbers.append(res[5])
                
        if not special_numbers:
            return list(range(1, 13))
            
        # Let's count frequencies in the last 100 draws
        recent_specials = special_numbers[:100]
        from collections import Counter
        counts = Counter(recent_specials)
        
        # We can also calculate last seen for each number
        last_seen = {}
        for idx, num in enumerate(special_numbers):
            if num not in last_seen:
                last_seen[num] = idx
                
        scored = []
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
            scored.append((score, n))
        return [number for _, number in sorted(scored, key=lambda item: (-item[0], item[1]))]
    except Exception as e:
        print(f"Error predicting special number for power535: {e}")
        return list(range(1, 13))


def predict_special_number_535(draws):
    """Return the highest-ranked Power 5/35 special ball."""
    return rank_special_numbers_535(draws)[0]


def format_vietlott_sms(product_id, numbers, ticket_index=1):
    """Format a ready-to-send MOMO Vietlott purchase message."""
    label = {"mega645": "645", "power655": "655", "power535": "535"}.get(product_id, product_id)
    config = LOTTERY_CONFIG[product_id]
    main_numbers = [int(number) for number in numbers[:config["num_per_draw"]]]
    main_text = " ".join(f"{number:02d}" for number in main_numbers)
    prefix = f"MOMO {label} K{ticket_index} S {main_text}"
    if product_id == "power535" and len(numbers) > config["num_per_draw"]:
        return f"{prefix}-{int(numbers[config['num_per_draw']]):02d}"
    return prefix


def build_vietlott_coverage_tickets(
    number_probabilities,
    num_per_draw,
    max_num,
    ticket_count=5,
    candidate_pool_size=None,
    preferred_max_shared=2,
    relationship_scores=None,
    pattern_profile=None,
):
    """Build a deterministic low-overlap portfolio for >=3 and >=4 hits.

    A single top-k ticket maximizes expected matches from marginal scores. To
    increase the chance that *one* ticket reaches three or four matches, a
    portfolio must cover different combinations. This helper makes that
    trade-off explicit instead of pretending one ticket suddenly has a
    guarantee.
    """
    probability_map = {
        int(item["number"]): max(float(item.get("prob", 0.0)), 1e-6)
        for item in (number_probabilities or [])
        if 1 <= int(item.get("number", 0)) <= max_num
    }
    ranked_numbers = sorted(
        range(1, max_num + 1),
        key=lambda number: (-probability_map.get(number, 1e-6), number),
    )
    pool_size = min(
        max_num,
        candidate_pool_size or max(12, 2 * num_per_draw + 2),
    )
    candidate_pool = ranked_numbers[:pool_size]

    relationship_scores = relationship_scores or {}
    pair_scores = relationship_scores.get("pair_scores", {})
    triple_scores = relationship_scores.get("triple_scores", {})
    pair_neutral = float(relationship_scores.get("pair_neutral", 0.25))
    triple_neutral = float(relationship_scores.get("triple_neutral", 0.25))
    pattern_profile = pattern_profile or {}
    predicted_odd = pattern_profile.get("pred_odd", pattern_profile.get("odd_mean"))
    predicted_small = pattern_profile.get("pred_small", pattern_profile.get("small_mean"))
    predicted_sum = pattern_profile.get("pred_sum", pattern_profile.get("sum_mean"))
    predicted_range = pattern_profile.get("pred_range", pattern_profile.get("range_mean"))
    predicted_consecutive = pattern_profile.get("pred_consecutive", pattern_profile.get("consecutive_mean"))
    predicted_tails = pattern_profile.get("pred_tail_diversity", pattern_profile.get("tail_diversity_mean"))
    penalty_scale = max(0.0, float(pattern_profile.get("penalty_scale", 1.0) or 1.0))
    strict_profile = bool(pattern_profile.get("strict_profile", False))

    def pair_signal(first, second):
        return pair_scores.get((min(first, second), max(first, second)), pair_neutral)

    def triple_signal(triple):
        return triple_scores.get(tuple(sorted(triple)), triple_neutral)

    candidates = []
    for ticket in combinations(candidate_pool, num_per_draw):
        ordered = tuple(sorted(ticket))
        log_score = sum(math.log(probability_map.get(number, 1e-6)) for number in ordered)
        ticket_triples = tuple(combinations(ordered, 3))
        pair_signal_mean = (
            sum(pair_signal(first, second) for first, second in combinations(ordered, 2))
            / max(math.comb(num_per_draw, 2), 1)
        )
        triple_signal_mean = (
            sum(triple_signal(triple) for triple in ticket_triples)
            / max(len(ticket_triples), 1)
        )
        # Relationship signals are deliberately small bonuses. They guide
        # combinations without overpowering the calibrated marginal scores.
        log_score += 0.35 * (pair_signal_mean - pair_neutral)
        log_score += 0.45 * (triple_signal_mean - triple_neutral)
        weighted_triples = {
            triple: math.prod(probability_map.get(number, 1e-6) for number in triple)
            * (1.0 + 0.8 * triple_signal(triple))
            for triple in ticket_triples
        }
        ticket_quadruples = tuple(combinations(ordered, 4)) if num_per_draw >= 4 else ()
        weighted_quadruples = {
            quadruple: math.prod(probability_map.get(number, 1e-6) for number in quadruple)
            * (
                1.0
                + 0.8 * float(np.mean([
                    triple_signal(triple)
                    for triple in combinations(quadruple, 3)
                ]))
            )
            for quadruple in ticket_quadruples
        }

        odds_count = sum(number % 2 for number in ordered)
        small_count = sum(number <= (max_num // 2) for number in ordered)
        consecutive_count = sum(
            ordered[index + 1] - ordered[index] == 1
            for index in range(len(ordered) - 1)
        )
        if strict_profile:
            if predicted_odd is not None and abs(odds_count - float(predicted_odd)) > 1:
                continue
            if predicted_small is not None and abs(small_count - float(predicted_small)) > 1:
                continue
            if predicted_sum is not None and abs(sum(ordered) - float(predicted_sum)) > 25:
                continue
            if predicted_range is not None and abs(ordered[-1] - ordered[0] - float(predicted_range)) > 14:
                continue
        profile_penalty = 0.0
        if predicted_odd is not None:
            profile_penalty += 0.12 * abs(odds_count - float(predicted_odd))
        if predicted_small is not None:
            profile_penalty += 0.12 * abs(small_count - float(predicted_small))
        if predicted_sum is not None:
            sum_deviation = abs(sum(ordered) - float(predicted_sum))
            profile_penalty += 0.002 * max(0.0, sum_deviation - 20.0)
        if predicted_range is not None:
            profile_penalty += 0.015 * max(
                0.0,
                abs(ordered[-1] - ordered[0] - float(predicted_range)) - 8.0,
            )
        if predicted_consecutive is not None:
            profile_penalty += 0.08 * abs(consecutive_count - float(predicted_consecutive))
        if predicted_tails is not None:
            profile_penalty += 0.08 * abs(
                len({number % 10 for number in ordered}) - float(predicted_tails)
            )
        log_score -= profile_penalty * penalty_scale
        candidates.append((log_score, ordered, weighted_triples, weighted_quadruples))

    # With exactly two tickets we can optimize the pair jointly instead of
    # greedily locking the first ticket before seeing the second. The objective
    # is the weighted union of their 3- and 4-number subsets: >=3/4 must occur
    # within either individual ticket; hits are never added across tickets.
    if ticket_count == 2 and 2 <= len(candidates) <= 500:
        overlap_limits = tuple(dict.fromkeys((preferred_max_shared, 3, num_per_draw - 1)))
        for max_shared in overlap_limits:
            best_pair = None
            best_pair_key = None
            for first_index, (first_log_score, first_ticket, first_triples, first_quadruples) in enumerate(candidates):
                first_set = set(first_ticket)
                first_weight = sum(first_triples.values())
                for second_log_score, second_ticket, second_triples, second_quadruples in candidates[first_index + 1:]:
                    shared_count = len(first_set.intersection(second_ticket))
                    if shared_count > max_shared:
                        continue
                    union_triples = dict(first_triples)
                    union_triples.update(second_triples)
                    union_quadruples = dict(first_quadruples)
                    union_quadruples.update(second_quadruples)
                    second_weight = sum(second_triples.values())
                    pair_key = (
                        # Prioritize the >=3 objective while retaining a
                        # meaningful secondary signal for >=4.
                        sum(union_triples.values())
                        + QUADRUPLE_OBJECTIVE_WEIGHT * sum(union_quadruples.values()),
                        sum(union_triples.values()),
                        sum(union_quadruples.values()),
                        min(first_weight, second_weight),
                        len(union_triples),
                        -shared_count,
                        math.exp((first_log_score + second_log_score) / (2 * num_per_draw)),
                        tuple(-number for number in first_ticket + second_ticket),
                    )
                    if best_pair_key is None or pair_key > best_pair_key:
                        best_pair_key = pair_key
                        best_pair = (first_ticket, second_ticket)
            if best_pair:
                return [list(ticket) for ticket in best_pair]

    # Greedy weighted maximum coverage: reaching >=3 matches means at least one
    # 3-number subset of a ticket is present in the result. Optimize that event
    # directly, rather than merely taking five variants of the top-k ticket.
    selected = []
    selected_numbers = set()
    covered_triples = set()
    covered_quadruples = set()
    while candidates and len(selected) < ticket_count:
        eligible_indices = []
        overlap_limits = tuple(dict.fromkeys((preferred_max_shared, 3, num_per_draw - 1)))
        for max_shared in overlap_limits:
            eligible_indices = [
                index for index, (_, ticket, _, _) in enumerate(candidates)
                if all(len(set(ticket).intersection(previous)) <= max_shared for previous in selected)
            ]
            if eligible_indices:
                break
        best_index = None
        best_key = None
        for index in eligible_indices:
            log_score, ticket, weighted_triples, weighted_quadruples = candidates[index]
            new_triple_weight = sum(
                weight for triple, weight in weighted_triples.items()
                if triple not in covered_triples
            )
            new_triple_count = sum(triple not in covered_triples for triple in weighted_triples)
            new_quadruple_weight = sum(
                weight for quadruple, weight in weighted_quadruples.items()
                if quadruple not in covered_quadruples
            )
            new_quadruple_count = sum(
                quadruple not in covered_quadruples
                for quadruple in weighted_quadruples
            )
            new_number_count = len(set(ticket).difference(selected_numbers))
            geometric_probability = math.exp(log_score / num_per_draw)
            key = (
                new_triple_weight + QUADRUPLE_OBJECTIVE_WEIGHT * new_quadruple_weight,
                new_triple_weight,
                new_quadruple_weight,
                new_quadruple_count,
                new_triple_count,
                new_number_count,
                geometric_probability,
                tuple(-number for number in ticket),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        _, ticket, weighted_triples, weighted_quadruples = candidates.pop(best_index)
        selected.append(ticket)
        selected_numbers.update(ticket)
        covered_triples.update(weighted_triples)
        covered_quadruples.update(weighted_quadruples)

    return [list(ticket) for ticket in selected[:ticket_count]]


def _portfolio_random_baseline(model_entries, num_per_draw, max_num, simulations=1200):
    """Estimate the random best-of-portfolio baseline for comparable metrics."""
    portfolios = [entry.get("all_tickets") or [] for entry in model_entries]
    portfolios = [portfolio for portfolio in portfolios if portfolio]
    if not portfolios:
        return {
            "avg_matched": 0.0,
            "at_least_2_pct": 0.0,
            "at_least_3_pct": 0.0,
            "at_least_4_pct": 0.0,
        }

    rng = random.Random(20260827 + max_num + num_per_draw)
    universe = list(range(1, max_num + 1))
    best_hits = []
    per_portfolio = max(200, int(simulations / len(portfolios)))
    for portfolio in portfolios:
        ticket_sets = [
            set((ticket.get("ticket", []) if isinstance(ticket, dict) else ticket)[:num_per_draw])
            for ticket in portfolio
        ]
        ticket_sets = [ticket for ticket in ticket_sets if len(ticket) == num_per_draw]
        if not ticket_sets:
            continue
        for _ in range(per_portfolio):
            actual = set(rng.sample(universe, num_per_draw))
            best_hits.append(max(len(actual.intersection(ticket)) for ticket in ticket_sets))
    if not best_hits:
        return {
            "avg_matched": 0.0,
            "at_least_2_pct": 0.0,
            "at_least_3_pct": 0.0,
            "at_least_4_pct": 0.0,
        }
    values = np.asarray(best_hits, dtype=int)
    return {
        "avg_matched": float(np.mean(values)),
        "at_least_2_pct": float(np.mean(values >= 2) * 100.0),
        "at_least_3_pct": float(np.mean(values >= 3) * 100.0),
        "at_least_4_pct": float(np.mean(values >= 4) * 100.0),
    }


def _apply_oos_calibration(predictions, calibration_history, min_samples=120):
    """Calibrate scores using only OOS observations available so far."""
    calibrated = {}
    reports = {}
    for model_name, items in (predictions or {}).items():
        history = calibration_history.get(model_name, {})
        scores = list(history.get("scores", []))
        labels = list(history.get("labels", []))
        calibrator = None
        if len(scores) >= min_samples:
            calibrator = _safe_isotonic_calibrator(scores, labels)
        result = []
        for item in items:
            raw_value = float(np.clip(item.get("prob", 0.0), 0.0, 1.0))
            value = raw_value
            if calibrator is not None:
                calibrated_value = float(np.clip(calibrator.predict([raw_value])[0], 0.001, 0.999))
                # Blend a small amount of the original rank signal so sparse
                # OOS bins do not flatten all candidates into a tie.
                value = 0.80 * calibrated_value + 0.20 * raw_value
            result.append({**item, "prob": float(np.clip(value, 0.0, 1.0)), "raw_prob": raw_value})
        calibrated[model_name] = result
        calibrated_history_scores = None
        if calibrator is not None:
            calibrated_history_scores = [
                0.80 * float(np.clip(calibrator.predict([score])[0], 0.001, 0.999))
                + 0.20 * float(score)
                for score in scores
            ]
        reports[model_name] = {
            "method": "isotonic_oos_blend" if calibrator is not None else "raw_until_oos_warmup",
            "samples": len(scores),
            "raw_brier": _brier_score(scores, labels),
            "calibrated_brier": _brier_score(calibrated_history_scores, labels) if calibrated_history_scores else None,
            "calibrator_ready": calibrator is not None,
        }
    return calibrated, reports


def predict_vietlott_game_ml(product_id, draws, progress_callback=None):
    """Predict one of the three supported 6/45, 6/55 or 5/35 games."""
    last_reported_progress = 0

    def report_progress(stage, progress):
        nonlocal last_reported_progress
        if callable(progress_callback):
            try:
                # Nested feature/model stages can round to a value just above
                # the next outer backtest boundary. Keep the source progress
                # monotonic so UIs and persisted worker status never appear
                # to go backwards at that boundary.
                safe_progress = max(
                    last_reported_progress,
                    min(100, max(0, int(progress))),
                )
                last_reported_progress = safe_progress
                progress_callback(stage, safe_progress)
            except Exception as progress_err:
                print(f"Vietlott progress callback failed: {progress_err}")

    config = LOTTERY_CONFIG.get(product_id)
    if config is None:
        raise ValueError(f"Sản phẩm {product_id} không được hỗ trợ.")

    draws = draws[:1000]
    num_per_draw = config["num_per_draw"]
    max_num = config["max_num"]
    split_num = config["split_num"]
    extractor = lambda draw: draw.get("result", [])[:num_per_draw]
    # Reuse only leakage-safe feature rows across the rolling OOS slices.
    # The cache is local to this training job, so it cannot retain stale data
    # after a new crawler sync.
    feature_cache = {}

    next_result = predict_draw_game(
        draws,
        config["min_num"],
        max_num,
        num_per_draw,
        extractor,
        training_draws=config["training_draws"],
        progress_callback=report_progress,
        progress_start=34,
        progress_end=45,
        feature_cache=feature_cache,
        source_offset=0,
    )
    if not next_result:
        return None
    report_progress("Đã huấn luyện dự báo kỳ kế tiếp", 45)

    base_model_names = ["linear_regression", "random_forest", "gradient_boosting", "mlp"]
    if xgb:
        base_model_names.extend(("xgboost", "xgb_ranker"))
    base_model_names.append("ensemble")

    def ticket_pattern(numbers):
        main_numbers = sorted(set(numbers[:num_per_draw]))
        return {
            "odd": sum(number % 2 for number in main_numbers),
            "even": num_per_draw - sum(number % 2 for number in main_numbers),
            "small": sum(number <= split_num for number in main_numbers),
            "large": num_per_draw - sum(number <= split_num for number in main_numbers),
            "sum": sum(main_numbers),
            "range": main_numbers[-1] - main_numbers[0] if main_numbers else 0,
            "consecutive": sum(main_numbers[index + 1] - main_numbers[index] == 1 for index in range(len(main_numbers) - 1)),
            "tail_diversity": len({number % 10 for number in main_numbers}),
        }

    backtest_entries = []
    # Keep a meaningful rolling validation window. The target draw is always
    # excluded from its training slice, and the window size is reported in the
    # API so the UI does not overstate the amount of validation performed.
    configured_backtest_window = os.getenv("VIETLOTT_BACKTEST_WINDOW", str(DEFAULT_BACKTEST_WINDOW))
    try:
        configured_backtest_window = int(configured_backtest_window)
    except (TypeError, ValueError):
        configured_backtest_window = DEFAULT_BACKTEST_WINDOW
    backtest_limit = max(20, min(MAX_BACKTEST_WINDOW, configured_backtest_window))
    num_backtests = min(backtest_limit, max(0, len(draws) - 119))

    # Each chronological OOS step may only use calibration samples from
    # earlier OOS steps. This prevents the calibration layer from peeking at
    # the target draw it is about to score.
    calibration_history = {
        model_name: {"scores": [], "labels": []}
        for model_name in base_model_names
    }
    calibration_reports = {}
    roundtable_performance_history = {model: [] for model in base_model_names if model != "ensemble"}

    # Walk forward from older draws to newer draws. Each prediction is scored
    # on untouched target data; backtest results are evaluation only and are
    # not fed back into the production forecast as post-hoc bias.
    for backtest_index, offset in enumerate(range(num_backtests - 1, -1, -1), start=1):
        step_start = 45 + ((backtest_index - 1) / max(num_backtests, 1)) * 45
        step_end = 45 + (backtest_index / max(num_backtests, 1)) * 45

        def report_backtest_progress(stage, progress):
            mapped_progress = step_start + (step_end - step_start) * (float(progress) / 100.0)
            report_progress(
                f"Backtest {backtest_index}/{num_backtests}: {stage}",
                min(int(round(mapped_progress)), 90),
            )

        report_backtest_progress("Đang chuẩn bị vòng kiểm định", 0)
        step_result = predict_draw_game(
            draws[offset:],
            config["min_num"],
            max_num,
            num_per_draw,
            extractor,
            training_draws=config["training_draws"],
            progress_callback=report_backtest_progress,
            progress_start=0,
            progress_end=100,
            include_diagnostics=False,
            feature_cache=feature_cache,
            source_offset=offset,
        )
        if not step_result:
            continue

        actual_len = 6 if product_id == "power535" else 7 if product_id == "power655" else 6
        actual_numbers = draws[offset].get("result", [])[:actual_len]
        actual_set = set(actual_numbers[:num_per_draw])
        actual_pattern = ticket_pattern(actual_numbers)
        raw_back_predictions = step_result["back"]
        calibrated_back_predictions, step_calibration_reports = _apply_oos_calibration(
            raw_back_predictions,
            calibration_history,
        )
        entry = {
            "draw_id": draws[offset].get("id"),
            "date": draws[offset].get("date"),
            "actual_numbers": actual_numbers,
            "actual_pattern": actual_pattern,
            "calibration": step_calibration_reports,
            "models": {},
        }

        for model in base_model_names:
            scored = calibrated_back_predictions[model]
            raw_scored = raw_back_predictions[model]
            predicted = [item["number"] for item in sorted(
                scored, key=lambda item: item["prob"], reverse=True
            )[:num_per_draw]]
            raw_predicted = [item["number"] for item in sorted(
                raw_scored, key=lambda item: item["prob"], reverse=True
            )[:num_per_draw]]
            matched = list(set(predicted).intersection(actual_set))
            raw_matched = list(set(raw_predicted).intersection(actual_set))
            predicted_pattern = ticket_pattern(predicted)
            pattern_errors = {
                "odd": abs(predicted_pattern["odd"] - actual_pattern["odd"]),
                "small": abs(predicted_pattern["small"] - actual_pattern["small"]),
                "sum": abs(predicted_pattern["sum"] - actual_pattern["sum"]),
                "range": abs(predicted_pattern["range"] - actual_pattern["range"]),
                "consecutive": abs(predicted_pattern["consecutive"] - actual_pattern["consecutive"]),
                "tail_diversity": abs(predicted_pattern["tail_diversity"] - actual_pattern["tail_diversity"]),
            }
            entry["models"][model] = {
                "predicted_numbers": predicted,
                "matched_numbers": matched,
                "matched_count": len(matched),
                "predicted_pattern": predicted_pattern,
                "pattern_errors": pattern_errors,
                "raw_predicted_numbers": raw_predicted,
                "raw_matched_numbers": raw_matched,
                "raw_matched_count": len(raw_matched),
            }

        # Record the reproducible common decision of the model meeting before
        # looking at this draw. This is separate from the plain Ensemble line:
        # every member casts a ticket, the chair combines weighted ranks, and
        # the two-ticket portfolio is then frozen and scored against reality.
        roundtable_tickets, member_votes, vote_tally = _build_roundtable_vote_tickets(
            calibrated_back_predictions,
            num_per_draw,
            max_num,
            ticket_count=2,
            relationship_scores=step_result.get("back_relationships"),
            pattern_profile=step_result.get("target_pattern"),
            quality_reports=step_calibration_reports,
            performance_history=roundtable_performance_history,
        )
        historical_specials = (
            rank_special_numbers_535(draws[offset + 1:])[:max(len(roundtable_tickets), 1)]
            if product_id == "power535" and len(draws[offset + 1:]) >= 1
            else []
        )
        roundtable_results = []
        for ticket_index, ticket in enumerate(roundtable_tickets):
            display_ticket = list(ticket)
            special_matched = False
            if historical_specials:
                historical_special = historical_specials[ticket_index % len(historical_specials)]
                display_ticket.append(int(historical_special))
                special_matched = (
                    len(actual_numbers) > num_per_draw
                    and int(actual_numbers[num_per_draw]) == int(historical_special)
                )
            matched = sorted(actual_set.intersection(ticket))
            ticket_result = {
                "ticket": display_ticket,
                "matched_numbers": matched,
                "matched_count": len(matched),
            }
            if historical_specials:
                ticket_result["special_matched"] = special_matched
            roundtable_results.append(ticket_result)
        best_roundtable = max(
            roundtable_results,
            key=lambda item: (item["matched_count"], bool(item.get("special_matched")), item["ticket"]),
            default={"ticket": [], "matched_numbers": [], "matched_count": 0, "special_matched": False},
        )
        roundtable_pattern = ticket_pattern(best_roundtable["ticket"])
        roundtable_record = {
            "predicted_numbers": best_roundtable["ticket"],
            "matched_numbers": best_roundtable["matched_numbers"],
            "matched_count": best_roundtable["matched_count"],
            "predicted_pattern": roundtable_pattern,
            "pattern_errors": {
                "odd": abs(roundtable_pattern["odd"] - actual_pattern["odd"]),
                "small": abs(roundtable_pattern["small"] - actual_pattern["small"]),
                "sum": abs(roundtable_pattern["sum"] - actual_pattern["sum"]),
                "range": abs(roundtable_pattern["range"] - actual_pattern["range"]),
                "consecutive": abs(roundtable_pattern["consecutive"] - actual_pattern["consecutive"]),
                "tail_diversity": abs(roundtable_pattern["tail_diversity"] - actual_pattern["tail_diversity"]),
            },
            "raw_predicted_numbers": best_roundtable["ticket"],
            "raw_matched_numbers": best_roundtable["matched_numbers"],
            "raw_matched_count": best_roundtable["matched_count"],
            "all_tickets": roundtable_results,
            "portfolio_size": len(roundtable_results),
            "metric_scope": "best_ticket_in_portfolio",
            "selection_mode": "roundtable_weighted_vote",
            "decision_rule": "75% xác suất calibration + 25% phiếu top; tối ưu phủ 2 vé trước khi biết kết quả",
            "member_votes": member_votes,
            "number_vote_tally": vote_tally[:15],
        }
        if historical_specials:
            roundtable_record["special_matched"] = any(
                bool(item.get("special_matched", False)) for item in roundtable_results
            )
        entry["models"]["roundtable_vote"] = roundtable_record

        # Only subsequent decisions may use the result of this target draw.
        for model in roundtable_performance_history:
            model_record = entry["models"].get(model) or {}
            roundtable_performance_history[model].append(int(model_record.get("matched_count", 0)))

        for ticket_budget in COVERAGE_TICKET_BUDGETS:
            concentrated_two_ticket = product_id == "power535" and ticket_budget == 2
            concentrated_three_ticket = product_id == "power535" and ticket_budget == 3
            # Use the calibrated ensemble as the common source for every
            # budget. A product-specific hard-coded source can silently become
            # stale after new features or recalibration are introduced.
            coverage_source_model = "ensemble"
            coverage_tickets = build_vietlott_coverage_tickets(
                calibrated_back_predictions[coverage_source_model],
                num_per_draw,
                max_num,
                ticket_count=ticket_budget,
                candidate_pool_size=(
                    8
                    if concentrated_two_ticket or concentrated_three_ticket
                    else 10
                    if ticket_budget == 2
                    else None
                ),
                preferred_max_shared=3 if concentrated_three_ticket else 2,
                relationship_scores=step_result.get("back_relationships"),
                pattern_profile=step_result.get("target_pattern"),
            )
            budget_results = []
            for ticket in coverage_tickets:
                matched = sorted(actual_set.intersection(ticket))
                budget_results.append({
                    "ticket": ticket,
                    "matched_numbers": matched,
                    "matched_count": len(matched),
                })
            best_coverage = max(
                budget_results,
                key=lambda item: (item["matched_count"], item["ticket"]),
                default={"ticket": [], "matched_numbers": [], "matched_count": 0},
            )
            coverage_pattern = ticket_pattern(best_coverage["ticket"])
            entry["models"][f"coverage_{ticket_budget}"] = {
                # This is explicitly the best ticket in a pre-generated
                # portfolio, not a hindsight-selected single-ticket forecast.
                "predicted_numbers": best_coverage["ticket"],
                "matched_numbers": best_coverage["matched_numbers"],
                "matched_count": best_coverage["matched_count"],
                "predicted_pattern": coverage_pattern,
                "pattern_errors": {
                    "odd": abs(coverage_pattern["odd"] - actual_pattern["odd"]),
                    "small": abs(coverage_pattern["small"] - actual_pattern["small"]),
                    "sum": abs(coverage_pattern["sum"] - actual_pattern["sum"]),
                    "range": abs(coverage_pattern["range"] - actual_pattern["range"]),
                    "consecutive": abs(coverage_pattern["consecutive"] - actual_pattern["consecutive"]),
                    "tail_diversity": abs(coverage_pattern["tail_diversity"] - actual_pattern["tail_diversity"]),
                },
                "raw_predicted_numbers": best_coverage["ticket"],
                "raw_matched_numbers": best_coverage["matched_numbers"],
                "raw_matched_count": best_coverage["matched_count"],
                "all_tickets": budget_results,
                "portfolio_size": len(budget_results),
                "metric_scope": "best_ticket_in_portfolio",
                "selection_mode": (
                    "joint_top8_two_ticket"
                    if concentrated_two_ticket
                    else "concentrated_top8_shared_core"
                    if concentrated_three_ticket
                    else "diversified_triple_coverage"
                ),
                "selection_source_model": coverage_source_model,
            }

        # Make this step available only to later OOS steps. The current target
        # is never used to calibrate its own prediction.
        for model in base_model_names:
            history = calibration_history[model]
            for item in raw_back_predictions.get(model, []):
                history["scores"].append(float(item.get("prob", 0.0)))
                history["labels"].append(int(item.get("number") in actual_set))
        backtest_entries.append(entry)
        if num_backtests:
            progress = 45 + int(backtest_index / num_backtests * 45)
            report_progress(
                f"Đang backtest dữ liệu lịch sử ({backtest_index}/{num_backtests} kỳ)",
                min(progress, 90),
            )

    backtest_summary = {}
    single_random_expected_avg = (num_per_draw * num_per_draw) / max_num
    combination_count = math.comb(max_num, num_per_draw)
    single_random_at_least_2 = sum(
        math.comb(num_per_draw, hits) * math.comb(max_num - num_per_draw, num_per_draw - hits)
        for hits in range(2, num_per_draw + 1)
        if num_per_draw - hits <= max_num - num_per_draw
    ) / combination_count * 100.0
    single_random_at_least_3 = sum(
        math.comb(num_per_draw, hits) * math.comb(max_num - num_per_draw, num_per_draw - hits)
        for hits in range(3, num_per_draw + 1)
        if num_per_draw - hits <= max_num - num_per_draw
    ) / combination_count * 100.0
    single_random_at_least_4 = sum(
        math.comb(num_per_draw, hits) * math.comb(max_num - num_per_draw, num_per_draw - hits)
        for hits in range(4, num_per_draw + 1)
        if num_per_draw - hits <= max_num - num_per_draw
    ) / combination_count * 100.0

    def wilson_interval(successes, total, z=1.96):
        if total <= 0:
            return [0.0, 0.0]
        proportion = successes / total
        denominator = 1.0 + z * z / total
        centre = (proportion + z * z / (2.0 * total)) / denominator
        margin = z * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        ) / denominator
        return [round(max(0.0, centre - margin) * 100.0, 1), round(min(1.0, centre + margin) * 100.0, 1)]

    coverage_model_names = [f"coverage_{budget}" for budget in COVERAGE_TICKET_BUDGETS]
    evaluation_model_names = base_model_names + ["roundtable_vote"] + coverage_model_names
    for model in evaluation_model_names:
        model_entries = [
            entry["models"][model]
            for entry in backtest_entries
            if model in entry.get("models", {})
        ]
        total_matches = sum(item.get("matched_count", 0) for item in model_entries)
        evaluated_draws = len(model_entries)
        avg_matched = total_matches / evaluated_draws if evaluated_draws else 0.0
        hit_distribution = {
            str(hits): sum(item.get("matched_count", 0) == hits for item in model_entries)
            for hits in range(num_per_draw + 1)
        }
        at_least_2_count = sum(item.get("matched_count", 0) >= 2 for item in model_entries)
        at_least_3_count = sum(item.get("matched_count", 0) >= 3 for item in model_entries)
        at_least_4_count = sum(item.get("matched_count", 0) >= 4 for item in model_entries)
        special_evaluated = sum("special_matched" in item for item in model_entries)
        special_match_count = sum(bool(item.get("special_matched")) for item in model_entries)
        special_random_rate = None
        if special_evaluated:
            covered_special_counts = []
            for item in model_entries:
                specials = {
                    int(ticket_info.get("ticket", [])[num_per_draw])
                    for ticket_info in item.get("all_tickets") or []
                    if len(ticket_info.get("ticket", [])) > num_per_draw
                }
                covered_special_counts.append(len(specials) or 1)
            special_random_rate = float(np.mean(covered_special_counts)) / 12.0 * 100.0
        at_least_3_ci95 = wilson_interval(at_least_3_count, evaluated_draws)
        at_least_4_ci95 = wilson_interval(at_least_4_count, evaluated_draws)
        if model.startswith("coverage_") or model == "roundtable_vote":
            random_baseline = _portfolio_random_baseline(model_entries, num_per_draw, max_num)
            random_expected_avg = random_baseline["avg_matched"]
            random_at_least_2 = random_baseline["at_least_2_pct"]
            random_at_least_3 = random_baseline["at_least_3_pct"]
            random_at_least_4 = random_baseline["at_least_4_pct"]
            ticket_count = max(
                (int(item.get("portfolio_size", 0)) for item in model_entries),
                default=DEFAULT_COVERAGE_TICKET_COUNT,
            ) or DEFAULT_COVERAGE_TICKET_COUNT
            metric_scope = f"best_of_{ticket_count}_tickets"
        else:
            random_expected_avg = single_random_expected_avg
            random_at_least_2 = single_random_at_least_2
            random_at_least_3 = single_random_at_least_3
            random_at_least_4 = single_random_at_least_4
            metric_scope = "single_ticket"
            ticket_count = 1
        pattern_keys = ("odd", "small", "sum", "range", "consecutive", "tail_diversity")
        pattern_mae = {
            key: round(
                sum(item.get("pattern_errors", {}).get(key, 0) for item in model_entries) / evaluated_draws,
                2,
            ) if evaluated_draws else 0.0
            for key in pattern_keys
        }
        exact_shape_rate = (
            sum(
                all(item.get("pattern_errors", {}).get(key, 1) == 0 for key in ("odd", "small"))
                for item in model_entries
            ) / evaluated_draws * 100.0
        ) if evaluated_draws else 0.0
        backtest_summary[model] = {
            "evaluated_draws": evaluated_draws,
            "total_matches": total_matches,
            "avg_matched": round(avg_matched, 3),
            "match_rate_pct": round(
                total_matches / (evaluated_draws * num_per_draw) * 100, 1
            ) if evaluated_draws else 0.0,
            "random_expected_avg": round(random_expected_avg, 3),
            "lift_vs_random_pct": round(
                (avg_matched / random_expected_avg - 1) * 100, 1
            ) if random_expected_avg else 0.0,
            "hit_distribution": hit_distribution,
            "at_least_2_count": at_least_2_count,
            "at_least_2_rate_pct": round(at_least_2_count / evaluated_draws * 100.0, 1) if evaluated_draws else 0.0,
            "at_least_3_count": at_least_3_count,
            "at_least_3_rate_pct": round(at_least_3_count / evaluated_draws * 100.0, 1) if evaluated_draws else 0.0,
            "at_least_3_ci95_pct": at_least_3_ci95,
            "at_least_4_count": at_least_4_count,
            "at_least_4_rate_pct": round(at_least_4_count / evaluated_draws * 100.0, 1) if evaluated_draws else 0.0,
            "at_least_4_ci95_pct": at_least_4_ci95,
            "random_at_least_2_pct": round(random_at_least_2, 1),
            "random_at_least_3_pct": round(random_at_least_3, 1),
            "random_at_least_4_pct": round(random_at_least_4, 1),
            "at_least_3_lift_vs_random_pct": round(
                ((at_least_3_count / evaluated_draws * 100.0) / random_at_least_3 - 1.0) * 100.0,
                1,
            ) if evaluated_draws and random_at_least_3 else 0.0,
            "at_least_4_lift_vs_random_pct": round(
                ((at_least_4_count / evaluated_draws * 100.0) / random_at_least_4 - 1.0) * 100.0,
                1,
            ) if evaluated_draws and random_at_least_4 else 0.0,
            "has_verified_at_least_3_edge": bool(
                evaluated_draws and at_least_3_ci95[0] > random_at_least_3
            ),
            "has_verified_at_least_4_edge": bool(
                evaluated_draws and at_least_4_ci95[0] > random_at_least_4
            ),
            "metric_scope": metric_scope,
            "ticket_count": ticket_count,
            "at_least_3_per_ticket_pct": round(
                (at_least_3_count / evaluated_draws * 100.0) / ticket_count,
                3,
            ) if evaluated_draws and ticket_count else 0.0,
            "random_at_least_3_per_ticket_pct": round(
                random_at_least_3 / ticket_count,
                3,
            ) if ticket_count else 0.0,
            "pattern_mae": pattern_mae,
            "exact_odd_small_rate_pct": round(exact_shape_rate, 1),
            "special_evaluated_draws": special_evaluated,
            "special_match_count": special_match_count,
            "special_match_rate_pct": round(
                special_match_count / special_evaluated * 100.0, 1
            ) if special_evaluated else None,
            "random_special_match_rate_pct": round(special_random_rate, 1) if special_random_rate is not None else None,
        }

    observed_budget_efficiency = {
        model: backtest_summary.get(model, {}).get("at_least_3_per_ticket_pct", 0.0)
        for model in coverage_model_names
    }
    recommended_coverage_model = max(
        coverage_model_names,
        key=lambda model: (
            observed_budget_efficiency[model],
            -backtest_summary.get(model, {}).get("ticket_count", 0),
            backtest_summary.get(model, {}).get("at_least_3_rate_pct", 0.0),
        ),
    )
    single_model_names = [
        model for model in base_model_names if model != "ensemble"
    ] + ["ensemble"]
    recommended_single_model = max(
        single_model_names,
        key=lambda model: (
            bool(backtest_summary.get(model, {}).get("has_verified_at_least_3_edge", False)),
            backtest_summary.get(model, {}).get("at_least_3_ci95_pct", [0.0, 0.0])[0],
            backtest_summary.get(model, {}).get("at_least_3_rate_pct", 0.0),
            backtest_summary.get(model, {}).get("at_least_2_rate_pct", 0.0),
            backtest_summary.get(model, {}).get("avg_matched", 0.0),
        ),
    )

    calibrated_next_predictions, calibration_reports = _apply_oos_calibration(
        next_result["next"],
        calibration_history,
    )

    forecast = {
        "models": {},
        "backtest_history": list(reversed(backtest_entries)),
        "backtest_summary": backtest_summary,
        "backtest_window": num_backtests,
        "backtest_mode": "walk_forward_out_of_sample",
        "ensemble_strategy": "rank_probability_consensus_v2_xgb_ranker_oos_calibrated",
        "coverage_strategy": "budgeted_joint_two_ticket_pair_triple_pattern_v2",
        "training_window_draws": config["training_draws"],
        "history_draw_count": len(draws),
        "recommended_coverage_model": recommended_coverage_model,
        "recommended_single_ticket_model": recommended_single_model,
        "goal_3_hits": {
            "mode": "budgeted_coverage",
            "default_ticket_count": DEFAULT_COVERAGE_TICKET_COUNT,
            "ticket_budgets": list(COVERAGE_TICKET_BUDGETS),
            "recommended_model": recommended_coverage_model,
            "warning": "Mỗi mức ngân sách phải được so với baseline ngẫu nhiên cùng số vé; không đảm bảo trúng từ 3 số.",
        },
        "number_features": next_result.get("next_features", {}),
        "feature_names": next_result.get("feature_names", []),
        "feature_groups": next_result.get("feature_groups", {}),
        "feature_importance": next_result.get("feature_importance", {}),
        "next_pattern": next_result.get("next_pattern", {}),
        "feature_engineering_version": "v4_bounded_recency_pair_triple_calendar",
        "calibration": calibration_reports,
    }
    report_progress("Đã hoàn tất backtest và tổng hợp dự báo", 94)

    for model in base_model_names:
        scored = []
        for item in calibrated_next_predictions.get(model, next_result["next"][model]):
            number = item["number"]
            raw_prob = item["prob"]
            scored.append({
                "number": number,
                "prob": float(np.clip(raw_prob, 0.0, 1.0)),
                "raw_prob": float(item.get("raw_prob", raw_prob)),
                "model_votes": item.get("model_votes"),
                "adjustment": 0.0,
            })

        lucky = [item["number"] for item in sorted(
            scored, key=lambda item: item["prob"], reverse=True
        )[:num_per_draw]]
        if product_id == "power535":
            lucky.append(predict_special_number_535(draws))
        forecast["models"][model] = {
            "lucky_numbers": lucky,
            "number_probabilities": scored,
            "pattern_profile": ticket_pattern(lucky),
        }

    profile = predict_draw_profile(draws, num_per_draw, split_num)
    forecast["profile_prediction"] = profile
    ensemble_probs = forecast["models"]["ensemble"]["number_probabilities"]

    live_member_predictions = {
        model: forecast["models"][model]["number_probabilities"]
        for model in _roundtable_model_weights(forecast["models"].keys())
        if model in forecast["models"]
    }
    roundtable_tickets, roundtable_member_votes, roundtable_tally = _build_roundtable_vote_tickets(
        live_member_predictions,
        num_per_draw,
        max_num,
        ticket_count=2,
        relationship_scores=next_result.get("next_relationships"),
        pattern_profile=profile,
        quality_reports=calibration_reports,
        performance_history=roundtable_performance_history,
    )
    if product_id == "power535":
        roundtable_specials = rank_special_numbers_535(draws)
        roundtable_tickets = [
            ticket + [roundtable_specials[index % len(roundtable_specials)]]
            for index, ticket in enumerate(roundtable_tickets)
        ]
    roundtable_probabilities = [
        {
            "number": item["number"],
            "prob": item["consensus_score"],
            "weighted_vote_pct": item["weighted_vote_pct"],
        }
        for item in roundtable_tally
    ]
    forecast["models"]["roundtable_vote"] = {
        "lucky_numbers": roundtable_tickets,
        "number_probabilities": roundtable_probabilities,
        "pattern_profile": ticket_pattern(roundtable_tickets[0]) if roundtable_tickets else None,
        "ticket_count": len(roundtable_tickets),
        "metric_scope": "portfolio",
        "cost_multiplier": len(roundtable_tickets),
        "selection_mode": "roundtable_weighted_vote",
        "selection_source_model": "model_roundtable",
        "decision_rule": "75% xác suất calibration + 25% phiếu top; tối ưu phủ 2 vé",
        "member_votes": roundtable_member_votes,
        "number_vote_tally": roundtable_tally[:15],
    }
    for ticket_budget in COVERAGE_TICKET_BUDGETS:
        concentrated_two_ticket = product_id == "power535" and ticket_budget == 2
        concentrated_three_ticket = product_id == "power535" and ticket_budget == 3
        coverage_source_model = "ensemble"
        coverage_source_probs = forecast["models"][coverage_source_model]["number_probabilities"]
        budget_tickets = build_vietlott_coverage_tickets(
            coverage_source_probs,
            num_per_draw,
            max_num,
            ticket_count=ticket_budget,
            candidate_pool_size=(
                8
                if concentrated_two_ticket or concentrated_three_ticket
                else 10
                if ticket_budget == 2
                else None
            ),
            preferred_max_shared=3 if concentrated_three_ticket else 2,
            relationship_scores=next_result.get("next_relationships"),
            pattern_profile=profile,
        )
        if product_id == "power535":
            special = predict_special_number_535(draws)
            budget_tickets = [ticket + [special] for ticket in budget_tickets]
        forecast["models"][f"coverage_{ticket_budget}"] = {
            "lucky_numbers": budget_tickets,
            "number_probabilities": coverage_source_probs,
            "pattern_profile": ticket_pattern(budget_tickets[0]) if budget_tickets else None,
            "ticket_count": len(budget_tickets),
            "metric_scope": "portfolio",
            "cost_multiplier": len(budget_tickets),
            "selection_mode": (
                "joint_top8_two_ticket"
                if concentrated_two_ticket
                else "concentrated_top8_shared_core"
                if concentrated_three_ticket
                else "diversified_triple_coverage"
            ),
            "selection_source_model": coverage_source_model,
        }

    # Public-facing recommendation: users should not have to choose between
    # several algorithms that naturally produce different tickets.  Keep the
    # individual model outputs above for diagnostics, but expose one stable
    # two-ticket policy selected by the out-of-sample coverage evaluation.
    # The two-ticket portfolio is deliberately used here because it is the
    # lowest-cost portfolio that matches the product's current user goal.
    auto_portfolio_key = "roundtable_vote"
    auto_portfolio = forecast["models"].get(auto_portfolio_key, {})
    auto_source_model = auto_portfolio.get("selection_source_model", "ensemble")
    auto_portfolio_summary = backtest_summary.get(auto_portfolio_key, {})
    auto_source_summary = backtest_summary.get(auto_source_model, {})
    auto_evaluated_draws = int(auto_portfolio_summary.get("evaluated_draws", 0) or 0)
    auto_p3 = float(auto_portfolio_summary.get("at_least_3_rate_pct", 0.0) or 0.0)
    auto_random_p3 = float(auto_portfolio_summary.get("random_at_least_3_pct", 0.0) or 0.0)
    auto_verified_edge = bool(auto_portfolio_summary.get("has_verified_at_least_3_edge", False))
    auto_edge_text = (
        "đã vượt baseline ngẫu nhiên với độ tin cậy thống kê"
        if auto_verified_edge
        else "chưa chứng minh được lợi thế thống kê so với baseline ngẫu nhiên"
    )
    auto_model = {
        **auto_portfolio,
        "display_name": "AI Tự Chọn (khuyến nghị)",
        "portfolio_key": auto_portfolio_key,
        "selection_mode": "ai_auto_roundtable_two_ticket",
        "selection_basis": "walk_forward_out_of_sample",
        "selection_source_model": auto_source_model,
        "selection_reason": (
            f"Đã đánh giá OOS {auto_evaluated_draws} kỳ; hệ thống dùng điểm "
            f"hội nghị biểu quyết để tạo 2 vé. Hiệu suất ≥3 là {auto_p3:.1f}% "
            f"so với baseline {auto_random_p3:.1f}%, {auto_edge_text}."
        ),
        "source_model_summary": auto_source_summary,
        "portfolio_summary": auto_portfolio_summary,
        "ticket_count": 2,
        "cost_multiplier": 2,
        "metric_scope": "best_of_2_tickets",
    }
    forecast["models"]["ai_recommended"] = auto_model
    forecast["recommended_model"] = "ai_recommended"
    forecast["ai_recommendation"] = {
        "mode": "AI Tự Chọn",
        "model_key": "ai_recommended",
        "portfolio_key": auto_portfolio_key,
        "source_model": auto_source_model,
        "selection_basis": "walk_forward_out_of_sample",
        "ticket_count": 2,
        "cost_multiplier": 2,
        "reason": auto_model["selection_reason"],
        "oos_summary": auto_portfolio_summary,
    }
    seed_material = f"{product_id}:{draws[0].get('id', '')}:{draws[0].get('date', '')}:{len(draws)}"
    monte_carlo_seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    tickets = run_monte_carlo_filter(
        product_id,
        ensemble_probs,
        profile,
        num_per_draw,
        max_num,
        split_num,
        draws,
        seed=monte_carlo_seed,
    )
    if product_id == "power535":
        special = predict_special_number_535(draws)
        tickets = [ticket + [special] for ticket in tickets]
    forecast["models"]["monte_carlo"] = {
        "lucky_numbers": tickets,
        "number_probabilities": ensemble_probs,
        "pattern_profile": ticket_pattern(tickets[0]) if tickets else None,
    }
    for model_data in forecast["models"].values():
        raw_tickets = model_data.get("lucky_numbers", [])
        tickets_for_sms = raw_tickets if raw_tickets and isinstance(raw_tickets[0], list) else [raw_tickets]
        model_data["sms_messages"] = [
            format_vietlott_sms(product_id, ticket, index)
            for index, ticket in enumerate(tickets_for_sms, start=1)
            if ticket
        ]
        model_data["sms_text"] = "\n".join(model_data["sms_messages"])
    return forecast


def build_vietlott_roundtable(product_id, forecast, draws, stats):
    """Build an auditable multi-model vote and the data packet for Codex."""
    config = LOTTERY_CONFIG[product_id]
    main_count = config["num_per_draw"]
    models = (forecast or {}).get("models") or {}
    labels = {
        "linear_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "mlp": "Mạng Nơ-ron MLP",
        "xgboost": "XGBoost Classifier",
        "xgb_ranker": "XGBoost Ranker",
    }
    active_keys = [key for key in labels if key in models]
    backtest_summary = (forecast or {}).get("backtest_summary") or {}
    prior_weights = _roundtable_model_weights(active_keys)
    skill_weights = {}
    for model_key in active_keys:
        summary = backtest_summary.get(model_key) or {}
        observed = float(summary.get("avg_matched", 0.0) or 0.0)
        baseline = float(summary.get("random_expected_avg", 0.0) or 0.0)
        skill = float(np.clip(observed / baseline, 0.5, 1.5)) if baseline else 1.0
        if summary.get("has_verified_at_least_3_edge"):
            skill *= 1.1
        skill_weights[model_key] = prior_weights.get(model_key, 0.0) * skill
    skill_total = sum(skill_weights.values()) or 1.0
    normalized_weights = {key: value / skill_total for key, value in skill_weights.items()}
    number_features = (forecast or {}).get("number_features") or {}
    stats_by_number = {
        int(item.get("number")): item
        for item in (stats or {}).get("number_stats") or []
        if item.get("number") is not None
    }

    members = []
    weighted_votes = {number: 0.0 for number in range(config["min_num"], config["max_num"] + 1)}
    vote_counts = {number: 0 for number in weighted_votes}
    model_votes = {number: [] for number in weighted_votes}
    for model_key in active_keys:
        model_data = models[model_key]
        raw_ticket = model_data.get("lucky_numbers") or []
        ticket = raw_ticket[0] if raw_ticket and isinstance(raw_ticket[0], list) else raw_ticket
        ticket = [int(value) for value in ticket[:main_count]]
        weight = normalized_weights.get(model_key, 0.0)
        for number in ticket:
            if number in weighted_votes:
                weighted_votes[number] += weight
                vote_counts[number] += 1
                model_votes[number].append(model_key)
        summary = backtest_summary.get(model_key) or {}
        calibration = ((forecast or {}).get("calibration") or {}).get(model_key) or {}
        number_reasons = []
        score_by_number = {
            int(item.get("number")): float(item.get("prob", 0.0))
            for item in model_data.get("number_probabilities") or []
            if item.get("number") is not None
        }
        for number in ticket:
            stat = stats_by_number.get(number, {})
            feature = number_features.get(str(number), number_features.get(number, {})) or {}
            number_reasons.append({
                "number": number,
                "score": round(score_by_number.get(number, 0.0), 6),
                "evidence": (
                    f"điểm {score_by_number.get(number, 0.0):.4f}; "
                    f"xuất hiện 30/60/100 kỳ = {int(stat.get('freq_30', 0) or 0)}/"
                    f"{int(stat.get('freq_60', 0) or 0)}/{int(stat.get('freq_100', 0) or 0)}; "
                    f"vắng {int(stat.get('draws_since_last', 0) or 0)} kỳ; "
                    f"same-weekday lift {float(feature.get('same_weekday_lift', 0.0) or 0.0):.3f}."
                ),
                "caveat": "Tần suất, độ vắng và quan hệ lịch chỉ là đặc trưng dự báo; không chứng minh số này phải xuất hiện.",
            })
        members.append({
            "model": model_key,
            "label": labels[model_key],
            "weight": float(weight),
            "ticket": ticket,
            "pattern": model_data.get("pattern_profile") or {},
            "oos": {
                "evaluated_draws": summary.get("evaluated_draws", 0),
                "avg_matched": summary.get("avg_matched", 0.0),
                "random_expected_avg": summary.get("random_expected_avg", 0.0),
                "at_least_3_rate_pct": summary.get("at_least_3_rate_pct", 0.0),
                "random_at_least_3_pct": summary.get("random_at_least_3_pct", 0.0),
                "at_least_3_ci95_pct": summary.get("at_least_3_ci95_pct", [0.0, 0.0]),
                "verified_edge": bool(summary.get("has_verified_at_least_3_edge", False)),
            },
            "calibration": {
                "method": calibration.get("method"),
                "samples": calibration.get("samples", 0),
                "raw_brier": calibration.get("raw_brier"),
                "calibrated_brier": calibration.get("calibrated_brier"),
                "ready": bool(calibration.get("calibrator_ready", False)),
            },
            "top_numbers": [
                {
                    "number": int(item.get("number")),
                    "score": float(item.get("prob", 0.0)),
                }
                for item in sorted(
                    model_data.get("number_probabilities") or [],
                    key=lambda item: float(item.get("prob", 0.0)),
                    reverse=True,
                )[:10]
            ],
            "number_reasons": number_reasons,
            "counter_argument": (
                "Nếu calibration chưa sẵn sàng hoặc khoảng tin cậy OOS còn chạm baseline ngẫu nhiên, "
                "phiếu này chỉ là tín hiệu tương đối và không đủ chứng minh lợi thế thống kê."
            ),
            "argument": (
                f"Đề xuất {'-'.join(f'{number:02d}' for number in ticket)}; "
                f"backtest OOS {int(summary.get('evaluated_draws', 0) or 0)} kỳ, "
                f"khớp TB {float(summary.get('avg_matched', 0.0) or 0.0):.2f}, "
                f"tỷ lệ ≥3 số {float(summary.get('at_least_3_rate_pct', 0.0) or 0.0):.1f}%. "
                "Các số được xếp theo điểm calibration và đặc trưng chỉ có trước kỳ dự báo."
            ),
        })

    number_vote_tally = [
        {
            "number": number,
            "weighted_vote_pct": round(weighted_votes[number] * 100.0, 2),
            "vote_count": vote_counts[number],
            "models": model_votes[number],
        }
        for number in weighted_votes
    ]
    number_vote_tally.sort(
        key=lambda item: (-item["weighted_vote_pct"], -item["vote_count"], item["number"])
    )

    candidates = []
    seen_tickets = set()

    def add_candidate(ticket, source, source_label):
        if not isinstance(ticket, list):
            return
        cleaned = [int(value) for value in ticket]
        required = main_count + (1 if product_id == "power535" else 0)
        if len(cleaned) < required:
            return
        cleaned = cleaned[:required]
        signature = tuple(cleaned)
        if signature in seen_tickets:
            return
        seen_tickets.add(signature)
        candidates.append({
            "id": f"C{len(candidates) + 1:02d}",
            "ticket": cleaned,
            "source": source,
            "source_label": source_label,
        })

    recommended = models.get("ai_recommended") or {}
    recommended_tickets = recommended.get("lucky_numbers") or []
    if recommended_tickets and not isinstance(recommended_tickets[0], list):
        recommended_tickets = [recommended_tickets]
    for ticket in recommended_tickets[:2]:
        add_candidate(ticket, "ai_recommended", "AI định lượng khuyến nghị")
    for member in members:
        source_ticket = list(member["ticket"])
        if product_id == "power535":
            special_values = models.get(member["model"], {}).get("lucky_numbers") or []
            if len(special_values) > main_count:
                source_ticket.append(int(special_values[main_count]))
        add_candidate(source_ticket, member["model"], member["label"])

    fallback_candidates = [candidate for candidate in candidates if candidate["source"] == "ai_recommended"][:2]
    if not fallback_candidates:
        fallback_candidates = candidates[:2]
    data_groups = [
        "30 kỳ quay gần nhất",
        "thống kê tần suất và số kỳ vắng của toàn bộ dải số",
        "quan hệ cặp số và cấu trúc kỳ quay",
        "đặc trưng của các số ứng viên",
        "xác suất và phiếu của từng mô hình",
        "walk-forward OOS, baseline ngẫu nhiên và calibration",
        "độ quan trọng các nhóm đặc trưng",
    ]
    roundtable = {
        "title": "Hội nghị bàn tròn dự báo Vietlott",
        "product": product_id,
        "quorum": len(members),
        "members": members,
        "number_vote_tally": number_vote_tally,
        "candidates": candidates,
        "decision": {
            "source": "ensemble_fallback",
            "selected_candidate_ids": [item["id"] for item in fallback_candidates],
            "selected_tickets": [item["ticket"] for item in fallback_candidates],
            "conclusion": "Bộ số định lượng dự phòng được chọn bằng walk-forward OOS và chiến lược phủ 2 vé.",
            "confidence": "low",
        },
        "data_groups": data_groups,
        "rule": "Codex giữ nguyên candidate nếu cả bộ hợp lý; nếu chỉ vài số mạnh thì phối hợp từng số. Backend tự xác minh nguồn và luật chơi.",
    }

    selected_numbers = sorted({
        number
        for candidate in candidates
        for number in candidate["ticket"][:main_count]
    })
    advisor_input = {
        "data_groups": data_groups,
        "product": product_id,
        "game_config": {
            "product": product_id,
            "min_number": config["min_num"],
            "max_number": config["max_num"],
            "main_numbers_per_draw": main_count,
            "training_draws": (forecast or {}).get("training_window_draws"),
            "backtest_window": (forecast or {}).get("backtest_window"),
        },
        "latest_draw": draws[0] if draws else None,
        "recent_draws": [
            {"id": item.get("id"), "date": item.get("date"), "result": item.get("result")}
            for item in (draws or [])[:30]
        ],
        "distribution": (stats or {}).get("distribution") or {},
        "odd_even_splits": ((stats or {}).get("odd_even_splits") or [])[:8],
        "small_large_splits": ((stats or {}).get("small_large_splits") or [])[:8],
        "number_statistics": [stats_by_number[number] for number in sorted(stats_by_number)],
        "next_draw_pattern": (forecast or {}).get("next_pattern") or {},
        "profile_prediction": (forecast or {}).get("profile_prediction") or {},
        "feature_importance": (forecast or {}).get("feature_importance") or {},
        "candidate_number_features": {
            str(number): number_features.get(str(number), {})
            for number in selected_numbers
        },
        "backtest_summary": {
            key: backtest_summary.get(key) or {}
            for key in active_keys + ["ensemble", "roundtable_vote", "coverage_2"]
        },
        "members": members,
        "number_vote_tally": number_vote_tally,
        "candidates": candidates,
    }
    return roundtable, advisor_input

def sync_vietlott_data(product_id: str, progress_callback=None, generate_forecast=True):
    """
    Runs the local crawler to fetch the latest draw results from the official Vietlott website,
    loads the data from the local JSONL file, calculates statistics, and caches results in MongoDB.

    ``generate_forecast=False`` is used by the low-cost background refresh. It
    updates draw data/statistics without retraining all model families. A
    forecast is still generated when no previous forecast exists, so a fresh
    installation remains usable on its first load.
    """
    if product_id not in PRODUCTS:
        raise ValueError(f"Sản phẩm {product_id} không được hỗ trợ.")

    def report_progress(stage, progress):
        if callable(progress_callback):
            try:
                progress_callback(stage, progress)
            except Exception as progress_err:
                print(f"Vietlott sync progress callback failed: {progress_err}")

    report_progress("Chuẩn bị đồng bộ dữ liệu xổ số", 5)
        
    github_file = PRODUCTS[product_id]
    
    # Run the local crawler for the selected supported product.
    crawler_prod = LOTTERY_CONFIG[product_id]["crawler"]
    import subprocess
    import sys
    cmd = [sys.executable, "-m", "vietlott.cli.crawl", crawler_prod]
        
    print(f"Vietlott Sync: Running local crawler for {crawler_prod}...")
    env = os.environ.copy()
    env["PYTHONSAFEPATH"] = "1"
    env["PYTHONPATH"] = "src"
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd="vietlott-data-crawler",
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"Local crawler warning (non-zero return code): {result.stderr.strip()}")
        else:
            print(f"Local crawler output: {result.stdout.strip()}")
    except Exception as e:
        print(f"Failed to run local crawler: {e}")
    report_progress("Đã lấy dữ liệu kỳ quay mới nhất", 18)

    # Read the data from the local crawler output.
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
    report_progress(f"Đã đọc {len(draws):,} kỳ quay lịch sử", 28)
        
    # Sort draws by date and ID descending (newest first).
    draws.sort(key=lambda x: (x.get("date", ""), x.get("id", "")), reverse=True)
    
    # All supported products use the standard numbered-draw statistics.
    stats = calculate_draw_game_stats(product_id, draws)
    report_progress("Đã tính thống kê tần suất và quan hệ số", 34)
        
    previous_forecast = None
    if not generate_forecast:
        try:
            previous_cache = db.vietlott_cache.find_one(
                {"_id": product_id},
                {"forecast": 1},
            )
            if previous_cache:
                previous_forecast = previous_cache.get("forecast")
        except Exception as cache_err:
            print(f"Vietlott: cannot read previous forecast for {product_id}: {cache_err}")

    should_generate_forecast = generate_forecast or previous_forecast is None
    if should_generate_forecast:
        try:
            print(f"Vietlott Sync: Generating AI forecast for {product_id}...")
            forecast = predict_vietlott_game_ml(
                product_id,
                draws,
                progress_callback=report_progress,
            )
        except Exception as e:
            print(f"Error generating AI forecast for {product_id}: {e}")
            forecast = None
    else:
        forecast = previous_forecast
        print(f"Vietlott Sync: Data-only refresh for {product_id}; skipping AI retraining.")
        report_progress("Đã cập nhật dữ liệu, giữ forecast AI hiện tại", 80)

    if forecast:
        try:
            roundtable, advisor_input = build_vietlott_roundtable(
                product_id,
                forecast,
                draws,
                stats,
            )
            previous_advisor = ((forecast.get("roundtable") or {}).get("chatgpt_advisor"))
            previous_is_codex = (
                isinstance(previous_advisor, dict)
                and previous_advisor.get("status") == "success"
                and previous_advisor.get("provider") == "Rùa AI Codex"
                and previous_advisor.get("advisor_version") == "codex-scientific-selection-v5"
            )
            if should_generate_forecast or not previous_is_codex:
                report_progress("Codex đang đọc dữ liệu và biên bản hội nghị Vietlott", 96)
                from openai_advisor import generate_vietlott_roundtable_advice

                chatgpt_advisor = generate_vietlott_roundtable_advice(
                    roundtable,
                    advisor_input,
                )
            else:
                chatgpt_advisor = previous_advisor
            roundtable["chatgpt_advisor"] = chatgpt_advisor
            roundtable["advisor_input_summary"] = {
                "data_groups": advisor_input.get("data_groups") or [],
                "recent_draw_count": len(advisor_input.get("recent_draws") or []),
                "number_stat_count": len(advisor_input.get("number_statistics") or []),
                "candidate_feature_count": len(advisor_input.get("candidate_number_features") or {}),
                "backtest_models": list((advisor_input.get("backtest_summary") or {}).keys()),
            }
            if chatgpt_advisor.get("status") == "success":
                final_tickets = chatgpt_advisor.get("selected_tickets") or []
                final_source = "codex"
                final_conclusion = chatgpt_advisor.get("conclusion") or ""
                final_confidence = chatgpt_advisor.get("confidence", "low")
            else:
                final_tickets = roundtable["decision"].get("selected_tickets") or []
                final_source = "ensemble_fallback"
                final_conclusion = roundtable["decision"].get("conclusion") or ""
                final_confidence = roundtable["decision"].get("confidence", "low")
            forecast["final_recommendation"] = {
                "source": final_source,
                "tickets": final_tickets,
                "conclusion": final_conclusion,
                "confidence": final_confidence,
                "advisor_model": chatgpt_advisor.get("model"),
            }
            forecast["roundtable"] = roundtable
            forecast["chatgpt_advisor"] = chatgpt_advisor
        except Exception as roundtable_error:
            print(f"Vietlott roundtable/Codex advisor failed for {product_id}: {roundtable_error}")
            forecast["chatgpt_advisor"] = {
                "status": "error",
                "provider": "Rùa AI Codex",
                "model": os.getenv("CODEX_MODEL", "") or "Codex mặc định",
                "generated_at": datetime.now().isoformat(),
                "message": str(roundtable_error)[:300],
            }
        
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
    report_progress("Đã lưu forecast và sẵn sàng hiển thị", 100)
    print(f"Vietlott Sync: Successfully cached {product_id} in MongoDB. Total draws: {len(draws)}.")
    return cache_doc

def calculate_draw_game_stats(product_id: str, draws: list):
    """
    Calculates number frequency, coldness, odd/even, and large/small stats for draw-style lotteries.
    """
    config = LOTTERY_CONFIG.get(product_id)
    if config is None:
        raise ValueError(f"Sản phẩm {product_id} không được hỗ trợ.")
    max_num = config["max_num"]
    split_num = config["split_num"]
    num_per_draw = config["num_per_draw"]

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
    forecast = result_data.get("forecast")
    if isinstance(forecast, dict) and not (forecast.get("roundtable") or {}).get("members"):
        try:
            roundtable, advisor_input = build_vietlott_roundtable(
                product_id,
                forecast,
                result_data.get("recent_draws") or [],
                result_data.get("stats") or {},
            )
            roundtable["chatgpt_advisor"] = forecast.get("chatgpt_advisor") or {
                "status": "skipped",
                "provider": "Rùa AI Codex",
                "model": os.getenv("CODEX_MODEL", "") or "Codex mặc định",
                "message": "Forecast cũ chưa có kết luận Codex; hãy bấm Huấn luyện lại.",
                "selected_tickets": [],
            }
            roundtable["advisor_input_summary"] = {
                "data_groups": advisor_input.get("data_groups") or [],
                "recent_draw_count": len(advisor_input.get("recent_draws") or []),
                "number_stat_count": len(advisor_input.get("number_statistics") or []),
                "candidate_feature_count": len(advisor_input.get("candidate_number_features") or {}),
                "backtest_models": list((advisor_input.get("backtest_summary") or {}).keys()),
            }
            forecast["roundtable"] = roundtable
        except Exception as roundtable_error:
            print(f"Cannot build cached Vietlott roundtable for {product_id}: {roundtable_error}")
    return result_data

def sync_all(generate_forecast=True):
    """
    Synchronizes all supported Vietlott products.

    The background loop passes ``generate_forecast=False`` to avoid starting
    three expensive ML retrains just because the periodic data refresh fired.
    """
    print("Vietlott Sync: Starting synchronization of all products...")
    results = {}
    for product in PRODUCTS:
        try:
            sync_vietlott_data(product, generate_forecast=generate_forecast)
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
