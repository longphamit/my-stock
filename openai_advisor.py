"""Rùa AI Codex-powered final reasoning for model roundtables.

The statistical models remain responsible for numeric forecasts. This module
uses their reports, votes, OOS weights and released market context to produce
an auditable qualitative conclusion for T+1 through T+5.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import PurePosixPath
from zoneinfo import ZoneInfo

import requests


DEFAULT_CODEX_API_BASE_URL = "https://api-ai.longpc.xyz"
MAX_CODEX_PROMPT_CHARS = 100000
VIETLOTT_ADVISOR_VERSION = "codex-scientific-selection-v5"
VIETNAM_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")


def _safe_codex_project():
    """Accept only a relative workspace path, as required by the public API."""
    project = str(os.getenv("CODEX_PROJECT", "")).strip()
    if not project:
        return ""
    path = PurePosixPath(project)
    if path.is_absolute() or ".." in path.parts:
        return ""
    return str(path)


def _response_error(response, fallback):
    try:
        payload = response.json()
    except Exception:
        payload = {}
    detail = payload.get("error") or payload.get("message") or fallback
    return f"{detail} (HTTP {response.status_code})"[:300]


def _run_codex(prompt, timeout_seconds=None, session=None):
    """Create one Codex run and poll the same runId until it finishes."""
    base_url = str(os.getenv("CODEX_API_BASE_URL", DEFAULT_CODEX_API_BASE_URL)).strip().rstrip("/")
    model = str(os.getenv("CODEX_MODEL", "")).strip()
    total_timeout = float(timeout_seconds or os.getenv("CODEX_TIMEOUT_SECONDS", "300"))
    request_timeout = float(os.getenv("CODEX_HTTP_TIMEOUT_SECONDS", "20"))
    poll_interval = max(0.25, float(os.getenv("CODEX_POLL_INTERVAL_SECONDS", "2")))
    http = session or requests
    created = http.post(
        f"{base_url}/api/public/codex",
        headers={"Content-Type": "application/json"},
        json={
            "prompt": prompt,
            "project": _safe_codex_project(),
            "sessionId": "",
            "model": model,
        },
        timeout=request_timeout,
    )
    if created.status_code != 202:
        raise RuntimeError(_response_error(created, "Không thể tạo yêu cầu Codex"))
    created_payload = created.json()
    run_id = str(created_payload.get("runId") or "").strip()
    if not run_id:
        raise ValueError("Rùa AI không trả về runId")

    started = time.monotonic()
    last_activity = "Đang chờ Codex xử lý"
    while True:
        if time.monotonic() - started >= total_timeout:
            try:
                http.post(
                    f"{base_url}/api/public/codex/{run_id}/stop",
                    headers={"Content-Type": "application/json"},
                    timeout=request_timeout,
                )
            except Exception:
                pass
            raise TimeoutError(f"Codex quá thời gian {total_timeout:.0f}s; runId={run_id}")

        time.sleep(poll_interval)
        status_response = http.get(
            f"{base_url}/api/public/codex/{run_id}",
            timeout=request_timeout,
        )
        if status_response.status_code in {429, 502, 503}:
            continue
        if status_response.status_code != 200:
            raise RuntimeError(_response_error(status_response, "Không lấy được trạng thái Codex"))
        payload = status_response.json()
        status = str(payload.get("status") or "").lower()
        last_activity = str(payload.get("activity") or last_activity)
        if status in {"queued", "running"}:
            continue
        if status == "completed":
            response_text = payload.get("response")
            if not isinstance(response_text, str) or not response_text.strip():
                raise ValueError("Codex hoàn tất nhưng không trả về nội dung")
            return {
                "run_id": run_id,
                "session_id": payload.get("sessionId"),
                "project": payload.get("project"),
                "url": payload.get("url"),
                "model": model or "Codex mặc định",
                "response": response_text.strip(),
            }
        if status in {"failed", "stopped"}:
            raise RuntimeError(str(payload.get("error") or f"Codex {status}: {last_activity}")[:300])
        raise RuntimeError(f"Trạng thái Codex không hợp lệ: {status or 'trống'}")


def _parse_codex_json(response_text):
    """Parse a JSON object even when a CLI wraps it in a Markdown fence."""
    text = str(response_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Codex không trả về JSON hợp lệ")
        parsed = json.loads(text[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Kết quả Codex phải là một JSON object")
    return parsed


def _compact_prompt_value(value, list_limit=12, string_limit=220):
    """Bound verbose evidence while preserving the structure Codex needs."""
    if isinstance(value, dict):
        return {
            str(key): _compact_prompt_value(item, list_limit, string_limit)
            for key, item in value.items()
        }
    if isinstance(value, list):
        if len(value) > list_limit:
            head_count = (list_limit + 1) // 2
            tail_count = list_limit // 2
            value = value[:head_count] + (value[-tail_count:] if tail_count else [])
        return [_compact_prompt_value(item, list_limit, string_limit) for item in value]
    if isinstance(value, str) and len(value) > string_limit:
        return value[:string_limit - 1].rstrip() + "…"
    return value


def _build_codex_prompt(instructions, schema, evidence_label, evidence):
    """Serialize a valid prompt that stays under the public API's size cap."""
    full_prompt = (
        instructions
        + "\nSCHEMA JSON:\n"
        + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        + f"\n{evidence_label}:\n"
        + json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
    )
    if len(full_prompt) <= MAX_CODEX_PROMPT_CHARS:
        return full_prompt
    for list_limit, string_limit in ((16, 240), (12, 200), (8, 160), (6, 120), (4, 90)):
        compact_evidence = _compact_prompt_value(evidence, list_limit, string_limit)
        prompt = (
            instructions
            + "\nSCHEMA JSON:\n"
            + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
            + f"\n{evidence_label}:\n"
            + json.dumps(compact_evidence, ensure_ascii=False, separators=(",", ":"))
        )
        if len(prompt) <= MAX_CODEX_PROMPT_CHARS:
            return prompt
    raise ValueError(
        f"Prompt Codex vượt giới hạn {MAX_CODEX_PROMPT_CHARS:,} ký tự sau khi thu gọn"
    )


def _compact_roundtable_input(analysis):
    roundtable = analysis.get("roundtable") or {}
    source_data = analysis.get("advisor_input_data") or {}
    members = []
    for member in roundtable.get("members") or []:
        members.append({
            "model": member.get("model"),
            "label": member.get("label"),
            "oos_weight": member.get("weight"),
            "r_squared_oos": member.get("r_squared"),
            "argument": member.get("reason"),
            "evidence": (member.get("evidence") or [])[:6],
            "counter_argument": member.get("counter_argument"),
            "fed_vote": member.get("fed_vote") or {},
            "horizons": [
                {
                    "horizon": item.get("horizon"),
                    "vote": item.get("vote"),
                    "price": item.get("price"),
                    "pct_change": item.get("pct_change"),
                    "reason": item.get("reason"),
                }
                for item in (member.get("horizons") or [])[:5]
            ],
        })

    signals = []
    for signal in (analysis.get("market_signals") or [])[:8]:
        if not isinstance(signal, dict):
            continue
        signals.append({
            "label": signal.get("label"),
            "short_term_bias": signal.get("short_term_bias"),
            "drivers": (signal.get("drivers") or [])[:4],
            "macro_values": signal.get("macro_values") or {},
        })

    fed = analysis.get("fed_policy_outlook") or {}
    compact_fed = {
        "as_of": fed.get("as_of"),
        "coverage": fed.get("coverage"),
        "confidence_pct": fed.get("confidence_pct"),
        "inflation": {
            key: (fed.get("inflation") or {}).get(key)
            for key in (
                "label", "score", "cpi_yoy", "pce_yoy", "core_pce_yoy", "core_pce_mom",
                "ppi_yoy", "core_ppi_yoy", "ppi_mom", "core_ppi_mom",
            )
        },
        "fed_action": {
            key: (fed.get("fed_action") or {}).get(key)
            for key in ("primary", "bias", "score", "probabilities", "horizon")
        },
        "gold_implication": fed.get("gold_implication") or {},
        "drivers": (fed.get("drivers") or [])[:4],
        "inputs": [
            {
                key: item.get(key)
                for key in ("key", "label", "value", "unit", "impact", "available")
                if item.get(key) is not None
            }
            for item in (fed.get("inputs") or [])[:12]
            if isinstance(item, dict)
        ],
        "components": (fed.get("components") or [])[:12],
        "market_confirmation": fed.get("market_confirmation") or {},
    }
    for event_key in ("latest_fed_event", "next_fed_event"):
        event = fed.get(event_key) or {}
        compact_fed[event_key] = {
            key: event.get(key)
            for key in ("date", "event", "impact", "policy_tone", "result", "days")
            if event.get(key) is not None
        }

    macro_snapshot = {}
    for key, item in (source_data.get("macro_snapshot") or {}).items():
        if not isinstance(item, dict):
            continue
        macro_snapshot[key] = {
            field: item.get(field)
            for field in (
                "name", "value", "prev_value", "date", "unit", "trend",
                "gold_impact", "impact_direction",
            )
            if item.get(field) is not None
        }
    recent_history = []
    for item in (source_data.get("recent_market_history") or [])[-5:]:
        if not isinstance(item, dict):
            continue
        recent_history.append({
            key: item.get(key)
            for key in (
                "date", "close", "world_price", "return_1d", "dxy", "us10y",
                "vix", "brent", "real_yield", "xagusd", "pce_headline_yoy",
                "pce_core_yoy", "ppi_yoy", "core_ppi_yoy", "ppi_mom", "core_ppi_mom",
                "days_to_fed", "days_to_cpi", "days_to_nfp", "days_to_pce", "days_to_ppi",
            )
            if item.get(key) is not None
        })
    compact_decisions = [
        {
            key: item.get(key)
            for key in (
                "horizon", "vote", "price", "pct_change", "confidence",
                "direction_confidence_score", "price_confidence",
                "price_oos_score", "context_overlay_pct",
            )
            if item.get(key) is not None
        }
        for item in (roundtable.get("decisions") or [])[:5]
        if isinstance(item, dict)
    ]
    compact_votes = [
        {
            key: item.get(key)
            for key in ("horizon", "weighted_tally", "vote_counts", "consensus", "signed_consensus")
            if item.get(key) is not None
        }
        for item in (roundtable.get("horizon_votes") or [])[:5]
        if isinstance(item, dict)
    ]

    return {
        "as_of": analysis.get("forecast_generated_at"),
        "current_price": analysis.get("current_price"),
        "forecast_date": analysis.get("forecast_date"),
        "model_version": analysis.get("model_version"),
        "technical_indicators": analysis.get("indicators") or {},
        "technical_and_market_reasons": (analysis.get("reasons") or [])[:12],
        "fed_policy_outlook": compact_fed,
        "market_signals": signals,
        "market_context": roundtable.get("context") or {},
        "members": members,
        "votes_by_horizon": compact_votes,
        "fed_vote_summary": roundtable.get("fed_vote_summary") or {},
        "fed_forecast_reference": roundtable.get("fed_forecast") or {},
        # This is only the shrunken numeric price path. It is not the chair's
        # directional vote: a tiny negative return may be displayed as flat
        # while the members still have a clear bearish directional bias.
        "ensemble_price_path_reference": compact_decisions,
        # This is the evidence packet used to form the model reports. Keeping
        # it beside the votes prevents the chair from merely repeating a
        # majority decision without checking the underlying observations.
        "source_data": {
            "recent_market_history": recent_history,
            # The crawler orders this as the latest released events followed
            # by upcoming releases; keep the tail so FOMC/CPI/PCE/PPI/NFP dates
            # near the forecast are not displaced by old January rows.
            "economic_calendar": (source_data.get("economic_calendar") or [])[-12:],
            "geopolitical_events": (source_data.get("geopolitical_events") or [])[:5],
            "macro_snapshot": macro_snapshot,
        },
    }


def _response_schema():
    decision_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "horizon", "direction", "confidence", "conclusion",
            "key_arguments", "dissenting_view", "risk_warning",
        ],
        "properties": {
            "horizon": {"type": "integer", "minimum": 1, "maximum": 5},
            "direction": {"type": "string", "enum": ["up", "down", "sideways"]},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "conclusion": {"type": "string"},
            "key_arguments": {
                "type": "array",
                "minItems": 2,
                "maxItems": 5,
                "items": {"type": "string"},
            },
            "dissenting_view": {"type": "string"},
            "risk_warning": {"type": "string"},
        },
    }
    fed_forecast_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "action", "confidence", "probabilities", "bias",
            "gold_implication", "conclusion", "key_arguments", "risk_warning",
        ],
        "properties": {
            "action": {"type": "string", "enum": ["hike", "hold", "cut"]},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "probabilities": {
                "type": "object",
                "additionalProperties": False,
                "required": ["hike", "hold", "cut"],
                "properties": {
                    "hike": {"type": "number", "minimum": 0, "maximum": 1},
                    "hold": {"type": "number", "minimum": 0, "maximum": 1},
                    "cut": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
            "bias": {"type": "string", "enum": ["hawkish", "neutral", "dovish"]},
            "gold_implication": {"type": "string", "enum": ["up", "down", "sideways"]},
            "conclusion": {"type": "string"},
            "key_arguments": {
                "type": "array",
                "minItems": 2,
                "maxItems": 5,
                "items": {"type": "string"},
            },
            "risk_warning": {"type": "string"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["meeting_summary", "fed_forecast", "decisions"],
        "properties": {
            "meeting_summary": {"type": "string"},
            "fed_forecast": fed_forecast_schema,
            "decisions": {
                "type": "array",
                "minItems": 5,
                "maxItems": 5,
                "items": decision_schema,
            },
        },
    }


def _normalize_advisor_result(parsed, roundtable):
    source_decisions = {
        int(item.get("horizon")): item
        for item in (roundtable.get("decisions") or [])
        if isinstance(item, dict) and item.get("horizon") is not None
    }
    raw_decisions = {
        int(item.get("horizon")): item
        for item in (parsed.get("decisions") or [])
        if isinstance(item, dict) and item.get("horizon") is not None
    }
    source_votes = {
        int(item.get("horizon")): item
        for item in (roundtable.get("horizon_votes") or [])
        if isinstance(item, dict) and item.get("horizon") is not None
    }
    normalized = []
    direction_labels = {"up": "Tăng", "down": "Giảm", "sideways": "Đi ngang"}
    for horizon in range(1, 6):
        if horizon not in raw_decisions:
            raise ValueError(f"Thiếu kết luận Codex cho T+{horizon}")
        item = raw_decisions[horizon]
        direction = item.get("direction")
        if direction not in direction_labels:
            raise ValueError(f"Hướng Codex không hợp lệ tại T+{horizon}")
        quantitative = source_decisions.get(horizon) or {}
        vote_summary = source_votes.get(horizon) or {}
        weighted_tally = vote_summary.get("weighted_tally") or {}
        directional_tally = {
            "up": float(weighted_tally.get("up", 0.0) or 0.0),
            "down": float(weighted_tally.get("down", 0.0) or 0.0),
        }
        dominant_direction = max(directional_tally, key=directional_tally.get)
        dominant_share = float(directional_tally[dominant_direction])
        context_values = ((roundtable.get("context") or {}).get("overlay_pct_by_horizon") or [])
        context_pct = float(context_values[horizon - 1]) if len(context_values) >= horizon else 0.0
        context_direction = "up" if context_pct > 0.02 else ("down" if context_pct < -0.02 else "sideways")
        direction_guard_applied = bool(
            direction == "sideways"
            and dominant_share >= 0.80
            and context_direction == dominant_direction
        )
        conclusion = str(item.get("conclusion") or "").strip()
        confidence = item.get("confidence", "low")
        if direction_guard_applied:
            direction = dominant_direction
            confidence = "low" if confidence not in {"medium", "high"} else confidence
            conclusion = (
                f"T+{horizon} nghiêng {direction_labels[direction].lower()} theo "
                f"{dominant_share * 100:.0f}% trọng số biểu quyết. Biên độ nhỏ hoặc OOS yếu "
                "chỉ làm giảm độ tin cậy, không đổi thiên hướng mạnh thành đi ngang."
            )
        normalized.append({
            "horizon": horizon,
            "direction": direction,
            "label": direction_labels[direction],
            "confidence": confidence,
            "conclusion": conclusion,
            "key_arguments": [str(value).strip() for value in (item.get("key_arguments") or []) if str(value).strip()][:5],
            "dissenting_view": str(item.get("dissenting_view") or "").strip(),
            "risk_warning": str(item.get("risk_warning") or "").strip(),
            # Numeric levels always come from Ensemble, never from generated text.
            "ensemble_price": quantitative.get("price"),
            "ensemble_pct_change": quantitative.get("pct_change"),
            "ensemble_direction": quantitative.get("vote"),
            "ensemble_direction_label": quantitative.get("label"),
            "dominant_vote_direction": dominant_direction,
            "dominant_vote_share": dominant_share,
            "direction_guard_applied": direction_guard_applied,
            "context_direction": context_direction,
        })
    return normalized


def _normalize_fed_forecast(parsed, roundtable):
    """Validate Codex's FED forecast and keep a deterministic fallback."""
    fallback = roundtable.get("fed_forecast") or {}
    raw = parsed.get("fed_forecast") or {}
    action_keys = {"hike", "hold", "cut"}
    action = raw.get("action") if raw.get("action") in action_keys else fallback.get("action_key")
    if action not in action_keys:
        action = "hold"

    raw_probabilities = raw.get("probabilities") or {}
    probabilities = {}
    for key in action_keys:
        try:
            value = float(raw_probabilities.get(key))
        except (TypeError, ValueError):
            value = None
        if value is None or value < 0:
            value = None
        if value is not None:
            probabilities[key] = min(1.0, value)
    if len(probabilities) != 3 or sum(probabilities.values()) <= 0:
        probabilities = {
            key: float((fallback.get("probabilities") or {}).get(key, 0.0) or 0.0)
            for key in action_keys
        }
    if sum(probabilities.values()) <= 0:
        probabilities = {"hike": 0.1, "hold": 0.8, "cut": 0.1}
    total = sum(probabilities.values())
    probabilities = {key: float(value / total) for key, value in probabilities.items()}

    bias = raw.get("bias") if raw.get("bias") in {"hawkish", "neutral", "dovish"} else fallback.get("bias", "neutral")
    gold_implication = raw.get("gold_implication") if raw.get("gold_implication") in {"up", "down", "sideways"} else fallback.get("gold_implication", "sideways")
    confidence = raw.get("confidence") if raw.get("confidence") in {"low", "medium", "high"} else "low"
    labels = {"hike": "Tăng lãi suất", "hold": "Giữ nguyên", "cut": "Giảm lãi suất"}
    gold_labels = {"up": "Tăng", "down": "Giảm", "sideways": "Đi ngang"}
    fallback_arguments = [str(value).strip() for value in (fallback.get("key_arguments") or []) if str(value).strip()]
    return {
        "action": action,
        "action_label": labels[action],
        "confidence": confidence,
        "probabilities": probabilities,
        "bias": bias,
        "gold_implication": gold_implication,
        "gold_implication_label": gold_labels[gold_implication],
        "conclusion": str(raw.get("conclusion") or fallback.get("conclusion") or "").strip(),
        "key_arguments": [
            str(value).strip()
            for value in (raw.get("key_arguments") or fallback_arguments)
            if str(value).strip()
        ][:5],
        "risk_warning": str(raw.get("risk_warning") or fallback.get("risk_warning") or "").strip(),
        "next_meeting": fallback.get("next_meeting"),
        "source": "Codex đối chiếu phiếu model + dữ liệu FED/vĩ mô",
    }


def generate_gold_roundtable_advice(analysis, timeout_seconds=None, session=None):
    """Return Codex conclusions, or a non-fatal error payload."""
    generated_at = datetime.now(VIETNAM_TIMEZONE).isoformat(timespec="seconds")
    report = _compact_roundtable_input(analysis)
    instructions = (
        "Đây chỉ là tác vụ suy luận dữ liệu. Không dùng công cụ, không chạy lệnh, không đọc/ghi "
        "file, không sửa source code và không tạo project. Bạn là chủ tọa độc lập của hội nghị "
        "dự báo giá vàng. Hãy suy luận bằng tiếng Việt "
        "chỉ từ báo cáo được cung cấp. Đánh giá cả phiếu số lượng, trọng số OOS, độ tin cậy, "
        "đường giá từng mô hình, FED/CPI/PCE/PPI/dữ liệu kinh tế, tin tức và địa chính trị. Không bịa "
        "thêm sự kiện hoặc con số. Không coi đồng thuận là chắc chắn; nêu ý kiến thiểu số và "
        "rủi ro. Phải kết luận riêng T+1 đến T+5 bằng tư duy khoa học: phiếu model là bằng chứng "
        "có tương quan, không phải chân lý; trước mỗi kết luận phải đối chiếu argument/evidence và "
        "counter_argument của từng model, chỉ ra luận điểm nào được dữ liệu độc lập xác nhận hoặc bác bỏ; "
        "đánh giá xem từng nhóm model có được kỹ thuật, vĩ mô, "
        "context và dữ liệu thực tế xác nhận hay không. Direction là thiên hướng tăng/giảm, không "
        "phải cam kết về biên độ. Khi một hướng đạt từ 60%, ưu tiên hướng đó nếu không có bằng chứng "
        "độc lập đối nghịch đủ mạnh. Được chọn sideways hoặc hướng khác khi đa số có điểm mù chung, "
        "nhưng phải chỉ rõ bằng chứng đối nghịch cụ thể. Biên độ nhỏ, giá Ensemble gần hiện tại hoặc "
        "OOS thấp chỉ làm giảm confidence, không đủ để tự động đổi thành sideways. Nếu đồng thuận từ "
        "80% và context cùng hướng thì không chọn sideways. Trường ensemble_price_path_reference chỉ "
        "là mốc giá đã co biên "
        "độ, không phải phiếu hướng bắt buộc. Đây là phân tích xác suất, không phải lời khuyên "
        "đầu tư. Nếu bằng chứng yếu hoặc mâu thuẫn, hạ confidence và nêu rõ rủi ro. "
        "Ngoài kết luận giá vàng T+1 đến T+5, bắt buộc đưa ra fed_forecast cho kỳ họp FED kế tiếp: "
        "chọn đúng một action hike/hold/cut, phân bổ xác suất hike/hold/cut, bias hawkish/neutral/dovish "
        "và gold_implication up/down/sideways. Phải đối chiếu fed_policy_outlook, dữ liệu CPI/PCE/PPI, việc làm, "
        "lợi suất, DXY, sự kiện FOMC và fed_vote của từng model; không được coi đây là thông báo chính thức của FED. "
        "Chỉ trả về đúng một JSON object, không Markdown, không giải thích ngoài JSON. JSON phải "
        "tuân thủ schema được cung cấp."
    )
    prompt = _build_codex_prompt(
        instructions,
        _response_schema(),
        "BÁO CÁO HỘI NGHỊ",
        report,
    )
    try:
        codex_result = _run_codex(prompt, timeout_seconds=timeout_seconds, session=session)
        parsed = _parse_codex_json(codex_result["response"])
        roundtable = analysis.get("roundtable") or {}
        decisions = _normalize_advisor_result(parsed, roundtable)
        fed_forecast = _normalize_fed_forecast(parsed, roundtable)
        return {
            "status": "success",
            "provider": "Rùa AI Codex",
            "model": codex_result["model"],
            "response_id": codex_result["run_id"],
            "run_id": codex_result["run_id"],
            "session_id": codex_result.get("session_id"),
            "generated_at": generated_at,
            "meeting_summary": str(parsed.get("meeting_summary") or "").strip(),
            "fed_forecast": fed_forecast,
            "decisions": decisions,
            "input_data_groups": list(
                ((analysis.get("advisor_input_data") or {}).get("data_groups") or [])
            ),
            "disclaimer": "Phân tích xác suất từ mô hình AI, không phải lời khuyên đầu tư.",
        }
    except Exception as error:
        return {
            "status": "error",
            "provider": "Rùa AI Codex",
            "model": str(os.getenv("CODEX_MODEL", "")).strip() or "Codex mặc định",
            "generated_at": generated_at,
            "message": str(error)[:300],
            "decisions": [],
        }


def _vietlott_response_schema(game_config=None):
    game_config = game_config or {}
    product = str(game_config.get("product") or "")
    main_count = int(game_config.get("main_numbers_per_draw") or 6)
    ticket_size = main_count + (1 if product == "power535" else 0)
    max_number = int(game_config.get("max_number") or 55)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "meeting_summary", "selected_candidate_ids", "selected_tickets", "confidence",
            "conclusion", "key_arguments", "ticket_explanations", "dissenting_view", "risk_warning",
        ],
        "properties": {
            "meeting_summary": {"type": "string"},
            "selected_candidate_ids": {
                "type": "array",
                "minItems": 0,
                "maxItems": 2,
                "items": {"type": "string"},
            },
            "selected_tickets": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
                "items": {
                    "type": "array",
                    "minItems": ticket_size,
                    "maxItems": ticket_size,
                    "items": {"type": "integer", "minimum": 1, "maximum": max_number},
                },
            },
            "ticket_explanations": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["ticket_index", "strategy", "rationale", "number_reasons"],
                    "properties": {
                        "ticket_index": {"type": "integer", "minimum": 1, "maximum": 2},
                        "strategy": {"type": "string", "enum": ["candidate_exact", "synthesized"]},
                        "rationale": {"type": "string"},
                        "number_reasons": {
                            "type": "array",
                            "minItems": ticket_size,
                            "maxItems": ticket_size,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["number", "position", "reason"],
                                "properties": {
                                    "number": {"type": "integer", "minimum": 1, "maximum": max_number},
                                    "position": {"type": "string", "enum": ["main", "special"]},
                                    "reason": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "conclusion": {"type": "string"},
            "key_arguments": {
                "type": "array",
                "minItems": 2,
                "maxItems": 6,
                "items": {"type": "string"},
            },
            "dissenting_view": {"type": "string"},
            "risk_warning": {"type": "string"},
        },
    }


def generate_vietlott_roundtable_advice(roundtable, advisor_input, timeout_seconds=None, session=None):
    """Let Codex synthesize final tickets from evidence and model votes."""
    generated_at = datetime.now(VIETNAM_TIMEZONE).isoformat(timespec="seconds")
    fallback = {
        "status": "error",
        "provider": "Rùa AI Codex",
        "model": str(os.getenv("CODEX_MODEL", "")).strip() or "Codex mặc định",
        "generated_at": generated_at,
        "message": "Codex chưa trả về kết luận; đang dùng bộ số Ensemble/AI định lượng.",
        "selected_candidate_ids": [],
        "selected_tickets": [],
        "advisor_version": VIETLOTT_ADVISOR_VERSION,
    }

    candidates = {
        str(item.get("id")): item
        for item in (roundtable.get("candidates") or [])
        if isinstance(item, dict) and item.get("id")
    }
    game_config = dict(advisor_input.get("game_config") or {})
    product = str(advisor_input.get("product") or roundtable.get("product") or "")
    game_config["product"] = product
    main_count = int(game_config.get("main_numbers_per_draw") or 6)
    min_number = int(game_config.get("min_number") or 1)
    max_number = int(game_config.get("max_number") or 55)
    ticket_size = main_count + (1 if product == "power535" else 0)

    instructions = (
        "Đây chỉ là tác vụ suy luận dữ liệu. Không dùng công cụ, không chạy lệnh, không đọc/ghi "
        "file, không sửa source code và không tạo project. Bạn là chủ tọa độc lập của hội nghị "
        "dự báo Vietlott. Hãy trả lời bằng tiếng Việt và "
        "chỉ dùng dữ liệu đầu vào: lịch sử kỳ quay, thống kê từng số, đặc trưng tần suất/chu kỳ, "
        "quan hệ cặp, cấu trúc bộ số, calibration và walk-forward OOS, cùng báo cáo/phiếu của "
        "các mô hình. Trước khi kết luận, phải đối chiếu lý do chọn từng số giữa các model, kiểm tra "
        "lập luận nào được nhiều nguồn độc lập hỗ trợ, phản biện bằng caveat/OOS/baseline và nêu rõ "
        "điểm bất đồng; không được chỉ đếm đa số phiếu. Xử lý đúng hai trường hợp: (1) nếu chỉ một số trong các candidate có bằng "
        "chứng tốt, giữ các số hợp lý rồi phối hợp với số khác từ dữ liệu/phiếu để tạo vé tổng hợp; "
        "(2) nếu toàn bộ một candidate đã hợp lý, giữ nguyên candidate đó. Không ưu tiên vé mới chỉ "
        "vì khác candidate và cũng không ưu tiên candidate nguyên bản nếu từng số không đủ bằng chứng. "
        "selected_candidate_ids ghi các candidate được giữ nguyên hoặc dùng làm nguồn tham khảo chính. "
        f"Mỗi vé phải có đúng {ticket_size} số; {main_count} số chính không trùng nhau trong "
        f"khoảng {min_number}-{max_number}. "
        + ("Số cuối là số đặc biệt trong khoảng 1-12 và có thể trùng số chính. Nếu chọn 2 vé mà "
           "dữ liệu chưa chứng minh một số đặc biệt có edge OOS, phải dùng 2 số đặc biệt khác nhau "
           "để tăng độ phủ; không dồn cả hai vé vào một bóng phụ chỉ vì tần suất gần đây. " if product == "power535" else "")
        + "Với mỗi vé, ticket_explanations phải nêu chiến lược giữ nguyên hay tổng hợp, lý do "
        "toàn bộ bộ số và lý do riêng cho từng số dựa trên dữ liệu cụ thể (phiếu model, xác suất, "
        "tần suất, chu kỳ vắng, cặp số, cấu trúc hoặc OOS). Không dùng lý do chung chung và không "
        "khẳng định quan hệ nhân quả từ dữ liệu xổ số ngẫu nhiên. "
        + "Không được tuyên bố có thể dự đoán chắc chắn xổ số; ưu tiên bằng chứng OOS "
        "và phải nêu rõ khi mô hình chưa chứng minh vượt baseline ngẫu nhiên. Chỉ trả về đúng "
        "một JSON object, không Markdown, không giải thích ngoài JSON và phải tuân thủ schema."
    )
    prompt = _build_codex_prompt(
        instructions,
        _vietlott_response_schema(game_config),
        "DỮ LIỆU VÀ BIÊN BẢN HỘI NGHỊ",
        advisor_input,
    )
    try:
        codex_result = _run_codex(prompt, timeout_seconds=timeout_seconds, session=session)
        parsed = _parse_codex_json(codex_result["response"])
        selected_ids = []
        for value in parsed.get("selected_candidate_ids") or []:
            candidate_id = str(value)
            if candidate_id in candidates and candidate_id not in selected_ids:
                selected_ids.append(candidate_id)
        selected_ids = selected_ids[:2]
        selected_tickets = []
        seen_tickets = set()
        for raw_ticket in parsed.get("selected_tickets") or []:
            if not isinstance(raw_ticket, list) or len(raw_ticket) != ticket_size:
                continue
            try:
                values = [int(value) for value in raw_ticket]
            except (TypeError, ValueError):
                continue
            main_numbers = values[:main_count]
            if len(set(main_numbers)) != main_count:
                continue
            if any(value < min_number or value > max_number for value in main_numbers):
                continue
            main_numbers = sorted(main_numbers)
            if product == "power535":
                special_number = values[-1]
                if special_number < 1 or special_number > 12:
                    continue
                normalized_ticket = main_numbers + [special_number]
            else:
                normalized_ticket = main_numbers
            signature = tuple(normalized_ticket)
            if signature in seen_tickets:
                continue
            seen_tickets.add(signature)
            selected_tickets.append(normalized_ticket)
        if not selected_tickets:
            raise ValueError("Codex không tạo được bộ số cuối hợp lệ")
        selected_tickets = selected_tickets[:2]
        if product == "power535" and len(selected_tickets) == 2 and selected_tickets[0][-1] == selected_tickets[1][-1]:
            alternative_specials = [
                int(item.get("ticket", [])[-1])
                for item in candidates.values()
                if len(item.get("ticket", [])) == ticket_size
                and 1 <= int(item.get("ticket", [])[-1]) <= 12
                and int(item.get("ticket", [])[-1]) != selected_tickets[0][-1]
            ]
            replacement = alternative_specials[0] if alternative_specials else (selected_tickets[0][-1] % 12) + 1
            selected_tickets[1][-1] = replacement
        candidate_signatures = {}
        for candidate_id, candidate in candidates.items():
            raw_candidate = candidate.get("ticket") or []
            if len(raw_candidate) != ticket_size:
                continue
            candidate_main = sorted(int(value) for value in raw_candidate[:main_count])
            normalized_candidate = (
                candidate_main + [int(raw_candidate[-1])]
                if product == "power535" else candidate_main
            )
            candidate_signatures[candidate_id] = tuple(normalized_candidate)
        ticket_provenance = []
        for ticket in selected_tickets:
            signature = tuple(ticket)
            exact_ids = [
                candidate_id
                for candidate_id, candidate_signature in candidate_signatures.items()
                if candidate_signature == signature
            ]
            if exact_ids:
                mode = "candidate_exact"
                source_ids = exact_ids
            else:
                mode = "synthesized"
                main_set = set(ticket[:main_count])
                overlap = sorted(
                    (
                        (len(main_set.intersection(signature_value[:main_count])), candidate_id)
                        for candidate_id, signature_value in candidate_signatures.items()
                    ),
                    reverse=True,
                )
                source_ids = [candidate_id for count, candidate_id in overlap if count > 0][:3]
            ticket_provenance.append({
                "ticket": ticket,
                "mode": mode,
                "mode_label": "Giữ nguyên candidate hợp lý" if mode == "candidate_exact" else "Tổng hợp các số có bằng chứng mạnh",
                "source_candidate_ids": source_ids,
            })
            for candidate_id in exact_ids:
                if candidate_id not in selected_ids:
                    selected_ids.append(candidate_id)
        selected_ids = selected_ids[:2]
        raw_explanations = {
            int(item.get("ticket_index")): item
            for item in (parsed.get("ticket_explanations") or [])
            if isinstance(item, dict) and item.get("ticket_index") is not None
        }
        ticket_explanations = []
        for index, ticket in enumerate(selected_tickets, start=1):
            raw = raw_explanations.get(index) or {}
            reason_pool = list(raw.get("number_reasons") or [])
            normalized_reasons = []
            used_reason_indices = set()
            for position_index, number in enumerate(ticket):
                expected_position = "special" if product == "power535" and position_index == main_count else "main"
                matched = None
                for reason_index, reason_item in enumerate(reason_pool):
                    if reason_index in used_reason_indices or not isinstance(reason_item, dict):
                        continue
                    try:
                        reason_number = int(reason_item.get("number", -1))
                    except (TypeError, ValueError):
                        continue
                    if reason_number == number and reason_item.get("position", "main") == expected_position:
                        matched = reason_item
                        used_reason_indices.add(reason_index)
                        break
                normalized_reasons.append({
                    "number": number,
                    "position": expected_position,
                    "reason": str((matched or {}).get("reason") or "Codex chưa nêu lý do riêng cho số này.").strip()[:500],
                })
            ticket_explanations.append({
                "ticket_index": index,
                # Backend-derived provenance is authoritative for exact vs synthesis.
                "strategy": ticket_provenance[index - 1]["mode"],
                "strategy_label": ticket_provenance[index - 1]["mode_label"],
                "rationale": str(raw.get("rationale") or "").strip()[:1200],
                "number_reasons": normalized_reasons,
            })
        return {
            "status": "success",
            "provider": "Rùa AI Codex",
            "advisor_version": VIETLOTT_ADVISOR_VERSION,
            "model": codex_result["model"],
            "response_id": codex_result["run_id"],
            "run_id": codex_result["run_id"],
            "session_id": codex_result.get("session_id"),
            "generated_at": generated_at,
            "meeting_summary": str(parsed.get("meeting_summary") or "").strip(),
            "selected_candidate_ids": selected_ids,
            "selected_tickets": selected_tickets,
            "ticket_provenance": ticket_provenance,
            "ticket_explanations": ticket_explanations,
            "confidence": parsed.get("confidence", "low"),
            "conclusion": str(parsed.get("conclusion") or "").strip(),
            "key_arguments": [str(value).strip() for value in (parsed.get("key_arguments") or []) if str(value).strip()][:6],
            "dissenting_view": str(parsed.get("dissenting_view") or "").strip(),
            "risk_warning": str(parsed.get("risk_warning") or "").strip(),
            "input_data_groups": list(advisor_input.get("data_groups") or []),
            "disclaimer": "Xổ số mang tính ngẫu nhiên; đây không phải cam kết trúng thưởng.",
        }
    except Exception as error:
        return {
            **fallback,
            "status": "error",
            "message": str(error)[:300],
        }
