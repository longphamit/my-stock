import asyncio
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
import os
import threading
import time
import uuid
from datetime import datetime, timedelta
from multiprocessing import Process, Queue
from queue import Empty as QueueEmpty
from zoneinfo import ZoneInfo

import crawler
import vietlott


VIETNAM_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")
GOLD_PREDICTION_HOUR = 5
GOLD_PREDICTION_MINUTE = 0
CALENDAR_RESULTS_UPDATE_HOUR = 3
CALENDAR_RESULTS_UPDATE_MINUTE = 0
BACKGROUND_ACTIVITY_COLLECTION = "background_activity_history"
GOLD_BACKTEST_RESULT_COLLECTION = "gold_backtest_results"
GOLD_CODEX_SNAPSHOT_COLLECTION = "gold_codex_retry_snapshots"

# Initialize MongoDB database
crawler.init_db()

app = FastAPI(title="Nền tảng Phân tích Giá Vàng và Vietlott")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create templates and static directories if they don't exist
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

class LearnRequest(BaseModel):
    date: str
    model: str


class ControlConfigPatch(BaseModel):
    """Whitelisted runtime controls for the pod API."""

    model_config = {"extra": "forbid"}
    backtest_window: Optional[int] = None
    training_draws: Optional[dict[str, int]] = None


CONTROL_ALLOWED_TRAINING_PRODUCTS = frozenset(vietlott.PRODUCTS)


def _control_runtime_config():
    return {
        "backtest_window": int(
            os.getenv("VIETLOTT_BACKTEST_WINDOW", str(vietlott.DEFAULT_BACKTEST_WINDOW))
        ),
        "training_draws": {
            product: int(vietlott.LOTTERY_CONFIG[product]["training_draws"])
            for product in vietlott.LOTTERY_CONFIG
        },
        "supported_products": sorted(vietlott.PRODUCTS),
    }


def _control_cache_snapshot():
    """Return safe, non-secret cache metadata for an operator/agent."""
    snapshot = {}
    for product in sorted(vietlott.PRODUCTS):
        try:
            document = vietlott.db.vietlott_cache.find_one(
                {"_id": product},
                {
                    "_id": 0,
                    "product_id": 1,
                    "updated_at": 1,
                    "total_draws": 1,
                    "latest_draw.id": 1,
                    "latest_draw.date": 1,
                    "forecast.backtest_window": 1,
                    "forecast.backtest_mode": 1,
                    "forecast.feature_engineering_version": 1,
                },
            )
            snapshot[product] = document or None
        except Exception as cache_err:
            snapshot[product] = {"error": str(cache_err)}
    return snapshot


# Gold prediction jobs can take longer than the market-data request.  Keep a
# small in-process status object so the UI can show the real stage while the
# worker is running instead of leaving the user with an unexplained spinner.
gold_prediction_status = {
    "status": "idle",
    "stage": "Chưa bắt đầu phân tích AI",
    "progress": 0,
    "started_at": None,
    "finished_at": None,
    "error": None,
}
gold_prediction_status_lock = threading.Lock()
gold_prediction_job_lock = threading.Lock()
gold_prediction_result_lock = threading.Lock()
gold_prediction_task = None
gold_prediction_result = None
codex_retry_status = {}
codex_retry_workers = {}
codex_retry_lock = threading.RLock()
gold_backtest_jobs = {}
gold_backtest_jobs_lock = threading.RLock()
GOLD_BACKTEST_JOB_RETENTION_SECONDS = 3600
vietlott_training_status = {}
vietlott_training_lock = threading.RLock()
vietlott_training_processes = {}
vietlott_training_progress_queues = {}
VIETLOTT_TRAINING_STATUS_COLLECTION = "vietlott_training_status"
# Progress is only observability; writing it on every nested model callback
# adds avoidable network latency to a CPU-bound training job.
VIETLOTT_STATUS_WRITE_INTERVAL_SECONDS = 1.0


def _background_activity_now():
    """Return an explicit Vietnam-time timestamp for activity records."""
    return datetime.now(VIETNAM_TIMEZONE).isoformat(timespec="seconds")


def _start_background_activity(
    *,
    task,
    kind,
    source,
    category="gold",
    product=None,
    db_module=None,
    details=None,
):
    """Create a durable activity record and return its in-process handle."""
    activity_id = uuid.uuid4().hex
    started_at = _background_activity_now()
    handle = {
        "activity_id": activity_id,
        "started_at": started_at,
        "started_monotonic": time.monotonic(),
        "kind": kind,
        "db_module": db_module or crawler,
    }
    document = {
        "activity_id": activity_id,
        "category": category,
        "task": task,
        "kind": kind,
        "source": source,
        "product": product,
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "duration_seconds": None,
        "crawl_started_at": started_at if kind in {"crawl", "sync_training"} else None,
        "crawl_finished_at": None,
        "crawl_duration_seconds": None,
        "training_started_at": started_at if kind == "training" else None,
        "training_finished_at": None,
        "training_duration_seconds": None,
        "details": details or {},
        "error": None,
    }
    try:
        handle["db_module"].db[BACKGROUND_ACTIVITY_COLLECTION].insert_one(document)
    except Exception as activity_err:
        # Observability must never prevent the actual crawler/training job.
        print(f"Background activity start log failed: {activity_err}")
    return handle


def _update_background_activity(activity, fields=None, details=None):
    if not activity:
        return
    update = {}
    if fields:
        update.update(fields)
    if details:
        update["details"] = details
    if not update:
        return
    try:
        activity["db_module"].db[BACKGROUND_ACTIVITY_COLLECTION].update_one(
            {"activity_id": activity["activity_id"]},
            {"$set": update},
        )
    except Exception as activity_err:
        print(f"Background activity update log failed: {activity_err}")


def _mark_background_activity_phase(activity, phase):
    """Mark crawl/training phase boundaries for combined jobs."""
    if not activity or activity.get(f"{phase}_finished_at"):
        return
    now = _background_activity_now()
    monotonic_now = time.monotonic()
    activity[f"{phase}_finished_at"] = now
    activity[f"{phase}_finished_monotonic"] = monotonic_now
    fields = {f"{phase}_finished_at": now}
    if phase == "crawl":
        fields["crawl_duration_seconds"] = round(
            monotonic_now - activity["started_monotonic"],
            2,
        )
    _update_background_activity(activity, fields=fields)


def _mark_background_training_start(activity):
    if not activity or activity.get("training_started_at"):
        return
    now = _background_activity_now()
    monotonic_now = time.monotonic()
    activity["training_started_at"] = now
    activity["training_started_monotonic"] = monotonic_now
    _update_background_activity(
        activity,
        fields={"training_started_at": now},
    )


def _finish_background_activity(activity, status, error=None, details=None):
    if not activity:
        return
    finished_at = _background_activity_now()
    monotonic_now = time.monotonic()
    fields = {
        "status": status,
        "finished_at": finished_at,
        "duration_seconds": round(
            monotonic_now - activity["started_monotonic"],
            2,
        ),
        "error": str(error) if error else None,
    }
    if activity.get("kind") == "sync_training":
        if not activity.get("crawl_finished_at"):
            _mark_background_activity_phase(activity, "crawl")
        if not activity.get("training_started_at"):
            activity["training_started_at"] = activity["started_at"]
            fields["training_started_at"] = activity["started_at"]
        fields["training_finished_at"] = finished_at
        training_start = activity.get("training_started_monotonic", activity["started_monotonic"])
        fields["training_duration_seconds"] = round(monotonic_now - training_start, 2)
    if details is not None:
        fields["details"] = details
    _update_background_activity(activity, fields=fields)


def _set_gold_prediction_status(status=None, stage=None, progress=None, error=None):
    with gold_prediction_status_lock:
        was_running = gold_prediction_status["status"] == "running"
        if status is not None:
            gold_prediction_status["status"] = status
        if stage is not None:
            gold_prediction_status["stage"] = stage
        if progress is not None:
            next_progress = max(0, min(100, int(progress)))
            # Several nested pipeline stages report their own percentages.
            # Never move the visible bar backwards while one job is running.
            if was_running or status == "running":
                next_progress = max(gold_prediction_status["progress"], next_progress)
            gold_prediction_status["progress"] = next_progress
        if error is not None:
            gold_prediction_status["error"] = str(error)
        elif status == "running":
            gold_prediction_status["error"] = None
        if status == "running" and not was_running:
            gold_prediction_status["started_at"] = datetime.now().isoformat()
            gold_prediction_status["finished_at"] = None
        if status in {"completed", "error"}:
            gold_prediction_status["finished_at"] = datetime.now().isoformat()


def _get_gold_prediction_status():
    with gold_prediction_status_lock:
        return dict(gold_prediction_status)


def _persist_gold_codex_retry_snapshot(result):
    """Keep the complete latest roundtable so Codex can be retried later.

    The retry action must not retrain the numeric models.  Persisting the
    already-built report also means a transient Codex outage does not discard
    the meeting packet when the web process is restarted.
    """
    if not isinstance(result, dict) or not result.get("roundtable"):
        return
    try:
        payload = jsonable_encoder(result)
        crawler.db[GOLD_CODEX_SNAPSHOT_COLLECTION].replace_one(
            {"_id": "latest"},
            {
                "_id": "latest",
                "payload": payload,
                "saved_at": datetime.now(VIETNAM_TIMEZONE).isoformat(timespec="seconds"),
            },
            upsert=True,
        )
    except Exception as snapshot_error:
        # Codex retry is an auxiliary capability; never turn a valid forecast
        # into a failed prediction because Mongo cannot write its snapshot.
        print(f"Gold Codex retry snapshot failed: {snapshot_error}")


def _get_gold_codex_retry_snapshot():
    with gold_prediction_result_lock:
        current = gold_prediction_result
    if isinstance(current, dict) and current.get("roundtable") and current.get("advisor_input_data"):
        return current
    try:
        saved = crawler.db[GOLD_CODEX_SNAPSHOT_COLLECTION].find_one(
            {"_id": "latest"},
            {"_id": 0, "payload": 1},
        ) or {}
        payload = saved.get("payload")
        return payload if isinstance(payload, dict) else None
    except Exception as snapshot_error:
        print(f"Gold Codex retry snapshot lookup failed: {snapshot_error}")
        return None


def _codex_retry_key(kind, product=None):
    return "gold" if kind == "gold" else f"vietlott:{product}"


def _get_codex_retry_status(key):
    with codex_retry_lock:
        return dict(codex_retry_status.get(key) or {
            "scope": key,
            "status": "idle",
            "stage": "Chưa gửi lại yêu cầu Codex",
            "progress": 0,
            "started_at": None,
            "finished_at": None,
            "error": None,
        })


def _set_codex_retry_status(key, status=None, stage=None, progress=None, error=None):
    with codex_retry_lock:
        current = codex_retry_status.setdefault(key, {
            "scope": key,
            "status": "idle",
            "stage": "Chưa gửi lại yêu cầu Codex",
            "progress": 0,
            "started_at": None,
            "finished_at": None,
            "error": None,
        })
        current["scope"] = key
        if status is not None:
            current["status"] = status
        if stage is not None:
            current["stage"] = stage
        if progress is not None:
            current["progress"] = max(0, min(100, int(progress)))
        if error is not None:
            current["error"] = str(error)[:500]
        elif status in {"queued", "running"}:
            current["error"] = None
        if status in {"queued", "running"} and not current.get("started_at"):
            current["started_at"] = datetime.now(VIETNAM_TIMEZONE).isoformat(timespec="seconds")
        if status in {"completed", "error"}:
            current["finished_at"] = datetime.now(VIETNAM_TIMEZONE).isoformat(timespec="seconds")
        snapshot = dict(current)
    return snapshot


def _run_codex_retry(kind, product=None):
    """Retry only the Codex chair using the existing roundtable report."""
    key = _codex_retry_key(kind, product)
    _set_codex_retry_status(
        key,
        status="running",
        stage="Đang gửi lại biên bản hội nghị sang Codex",
        progress=10,
    )
    try:
        from openai_advisor import (
            generate_gold_roundtable_advice,
            generate_vietlott_roundtable_advice,
        )

        if kind == "gold":
            analysis = _get_gold_codex_retry_snapshot()
            if not analysis:
                raise RuntimeError("Chưa có biên bản hội nghị vàng để gửi lại Codex.")
            advisor = generate_gold_roundtable_advice(analysis)
            analysis["chatgpt_advisor"] = advisor
            if isinstance(analysis.get("roundtable"), dict):
                analysis["roundtable"]["chatgpt_advisor"] = advisor
            with gold_prediction_result_lock:
                global gold_prediction_result
                gold_prediction_result = analysis
            _persist_gold_codex_retry_snapshot(analysis)
            if advisor.get("status") != "success":
                raise RuntimeError(advisor.get("message") or "Codex chưa trả về kết luận.")
        else:
            cached = vietlott.db.vietlott_cache.find_one({"_id": product})
            forecast = (cached or {}).get("forecast") or {}
            if not cached or not forecast:
                raise RuntimeError(f"Chưa có forecast {product} để gửi lại Codex.")
            roundtable, advisor_input = vietlott.build_vietlott_roundtable(
                product,
                forecast,
                cached.get("recent_draws") or [],
                cached.get("stats") or {},
            )
            advisor = generate_vietlott_roundtable_advice(roundtable, advisor_input)
            roundtable["chatgpt_advisor"] = advisor
            roundtable["advisor_input_summary"] = {
                "data_groups": advisor_input.get("data_groups") or [],
                "recent_draw_count": len(advisor_input.get("recent_draws") or []),
                "number_stat_count": len(advisor_input.get("number_statistics") or []),
                "candidate_feature_count": len(advisor_input.get("candidate_number_features") or {}),
                "backtest_models": list((advisor_input.get("backtest_summary") or {}).keys()),
            }
            forecast["roundtable"] = roundtable
            forecast["chatgpt_advisor"] = advisor
            if advisor.get("status") == "success":
                final_tickets = advisor.get("selected_tickets") or []
                final_source = "codex"
                final_conclusion = advisor.get("conclusion") or ""
                final_confidence = advisor.get("confidence", "low")
            else:
                final_tickets = roundtable.get("decision", {}).get("selected_tickets") or []
                final_source = "ensemble_fallback"
                final_conclusion = roundtable.get("decision", {}).get("conclusion") or ""
                final_confidence = roundtable.get("decision", {}).get("confidence", "low")
            forecast["final_recommendation"] = {
                "source": final_source,
                "tickets": final_tickets,
                "conclusion": final_conclusion,
                "confidence": final_confidence,
                "advisor_model": advisor.get("model"),
            }
            vietlott.db.vietlott_cache.update_one(
                {"_id": product},
                {"$set": {"forecast": forecast, "updated_at": datetime.now().isoformat()}},
            )
            if advisor.get("status") != "success":
                raise RuntimeError(advisor.get("message") or "Codex chưa trả về kết luận.")

        _set_codex_retry_status(
            key,
            status="completed",
            stage="Codex đã đọc lại biên bản và cập nhật kết luận",
            progress=100,
            error=None,
        )
    except Exception as retry_error:
        _set_codex_retry_status(
            key,
            status="error",
            stage="Không thể gửi lại yêu cầu Codex",
            progress=100,
            error=retry_error,
        )
        print(f"Codex retry failed for {key}: {retry_error}")


def _queue_codex_retry(kind, product=None):
    key = _codex_retry_key(kind, product)
    with codex_retry_lock:
        current = _get_codex_retry_status(key)
        if current.get("status") in {"queued", "running"}:
            return current
        if kind == "gold":
            if not _get_gold_codex_retry_snapshot():
                raise ValueError("Chưa có biên bản hội nghị vàng để gửi lại Codex.")
        else:
            cached = vietlott.db.vietlott_cache.find_one({"_id": product}, {"forecast": 1})
            if not cached or not cached.get("forecast"):
                raise ValueError(f"Chưa có forecast {product} để gửi lại Codex.")
        _set_codex_retry_status(
            key,
            status="queued",
            stage="Đang xếp hàng gửi lại biên bản hội nghị",
            progress=1,
        )
        worker = threading.Thread(
            target=_run_codex_retry,
            args=(kind, product),
            name=f"codex-retry-{key.replace(':', '-')}",
            daemon=True,
        )
        codex_retry_workers[key] = worker
        worker.start()
        return _get_codex_retry_status(key)


def _snapshot_gold_backtest_job(job):
    """Return a JSON-safe copy without exposing the worker thread handle."""
    if not job:
        return None
    return {
        key: value
        for key, value in job.items()
        if key not in {"thread", "finished_monotonic"}
    }


def _cleanup_gold_backtest_jobs_locked():
    now = time.monotonic()
    expired = [
        job_id
        for job_id, job in gold_backtest_jobs.items()
        if job.get("finished_monotonic")
        and now - job["finished_monotonic"] > GOLD_BACKTEST_JOB_RETENTION_SECONDS
    ]
    for job_id in expired:
        gold_backtest_jobs.pop(job_id, None)


def _get_gold_backtest_job(job_id):
    with gold_backtest_jobs_lock:
        _cleanup_gold_backtest_jobs_locked()
        return _snapshot_gold_backtest_job(gold_backtest_jobs.get(job_id))


def _persist_gold_backtest_result(result):
    """Persist the complete backtest payload so charts survive page reloads."""
    if not isinstance(result, dict):
        return
    status = result.get("status")
    if status not in {"success", "no_data"}:
        return
    year = result.get("year")
    month = result.get("month")
    if year is None or month is None:
        return
    try:
        payload = dict(result)
        payload["year"] = int(year)
        payload["month"] = int(month)
        payload["saved_at"] = datetime.now(VIETNAM_TIMEZONE).isoformat(timespec="seconds")
        crawler.db[GOLD_BACKTEST_RESULT_COLLECTION].replace_one(
            {"year": payload["year"], "month": payload["month"]},
            payload,
            upsert=True,
        )
    except Exception as persist_error:
        # Backtest result persistence is best-effort; it must not turn a
        # completed CPU-heavy backtest into a failed job.
        print(f"Gold backtest result persistence failed: {persist_error}")


def _validated_gold_backtest_result(result):
    """Reject cached results that claim an actual price after today."""
    if not isinstance(result, dict):
        return None
    today = datetime.now(VIETNAM_TIMEZONE).strftime("%Y-%m-%d")
    rows = result.get("rows") or []
    if any(str(row.get("date", "")) > today for row in rows if isinstance(row, dict)):
        return None
    payload = dict(result)
    payload.setdefault("as_of_date", today)
    return payload


def _latest_gold_backtest_result(year, month):
    """Return the latest in-memory or durable result for a month."""
    with gold_backtest_jobs_lock:
        _cleanup_gold_backtest_jobs_locked()
        matching_jobs = [
            job for job in gold_backtest_jobs.values()
            if job.get("year") == year
            and job.get("month") == month
            and job.get("status") in {"completed", "no_data"}
            and isinstance(job.get("result"), dict)
        ]
        if matching_jobs:
            matching_jobs.sort(key=lambda job: job.get("finished_at") or "", reverse=True)
            return _validated_gold_backtest_result(matching_jobs[0].get("result"))
    try:
        result = crawler.db[GOLD_BACKTEST_RESULT_COLLECTION].find_one(
            {"year": year, "month": month},
            {"_id": 0},
        )
        return _validated_gold_backtest_result(result)
    except Exception as read_error:
        print(f"Gold backtest result lookup failed: {read_error}")
        return None


def _run_gold_month_backtest_job(job_id, year, month):
    with gold_backtest_jobs_lock:
        job = gold_backtest_jobs.get(job_id)
        if not job:
            return
        job["status"] = "running"
        job["stage"] = "Đang tải dữ liệu và chạy backtest"
        job["started_at"] = datetime.now(VIETNAM_TIMEZONE).isoformat(timespec="seconds")

    try:
        result = _run_gold_month_backtest_for_activity(year, month)
        result_status = result.get("status")
        job_status = "completed" if result_status == "success" else result_status or "error"
        _persist_gold_backtest_result(result)
        with gold_backtest_jobs_lock:
            job = gold_backtest_jobs.get(job_id)
            if job:
                job.update({
                    "status": job_status,
                    "stage": "Đã hoàn tất" if job_status == "completed" else result.get("message", "Backtest không có kết quả."),
                    "finished_at": datetime.now(VIETNAM_TIMEZONE).isoformat(timespec="seconds"),
                    "finished_monotonic": time.monotonic(),
                    "result": result,
                })
    except Exception as exc:
        with gold_backtest_jobs_lock:
            job = gold_backtest_jobs.get(job_id)
            if job:
                job.update({
                    "status": "error",
                    "stage": "Backtest thất bại",
                    "error": str(exc),
                    "finished_at": datetime.now(VIETNAM_TIMEZONE).isoformat(timespec="seconds"),
                    "finished_monotonic": time.monotonic(),
                    "result": {
                        "status": "error",
                        "message": str(exc),
                        "year": year,
                        "month": month,
                        "models": [],
                        "rows": [],
                        "errors": [str(exc)],
                    },
                })


def _queue_gold_month_backtest_job(year, month):
    """Queue a long backtest and return immediately before proxy timeout."""
    with gold_backtest_jobs_lock:
        _cleanup_gold_backtest_jobs_locked()
        for job in gold_backtest_jobs.values():
            if (
                job.get("year") == year
                and job.get("month") == month
                and job.get("status") in {"queued", "running"}
            ):
                return _snapshot_gold_backtest_job(job)

        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "status": "queued",
            "stage": "Đang xếp hàng backtest",
            "year": year,
            "month": month,
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }
        worker = threading.Thread(
            target=_run_gold_month_backtest_job,
            args=(job_id, year, month),
            name=f"gold-backtest-{job_id[:8]}",
            daemon=True,
        )
        job["thread"] = worker
        gold_backtest_jobs[job_id] = job
        worker.start()
        return _snapshot_gold_backtest_job(job)


def _gold_prediction_progress(stage, progress):
    _set_gold_prediction_status(status="running", stage=stage, progress=progress)


def _run_gold_predictions_with_status(source="manual"):
    global gold_prediction_result
    activity = _start_background_activity(
        task="Dự đoán AI giá vàng",
        kind="training",
        source=source,
        category="gold",
        details={"model_version": "gold_prediction_pipeline"},
    )

    with gold_prediction_result_lock:
        gold_prediction_result = None

    _set_gold_prediction_status(
        status="running",
        stage="Khởi tạo mô hình và đọc dữ liệu giá vàng",
        progress=3,
        error=None,
    )
    try:
        result = crawler.get_gold_predictions(progress_callback=_gold_prediction_progress)
        _persist_gold_codex_retry_snapshot(result)
        # Store the payload before marking the status completed.  This avoids
        # a race where the UI sees 100% but the result is not ready yet.
        with gold_prediction_result_lock:
            gold_prediction_result = result
        if isinstance(result, dict) and result.get("status") == "error":
            _finish_background_activity(
                activity,
                "error",
                error=result.get("message", "Không rõ lỗi"),
                details={
                    "model_version": result.get("model_version"),
                    "forecast_date": result.get("forecast_date"),
                },
            )
            _set_gold_prediction_status(
                status="error",
                stage="Phân tích thất bại",
                progress=100,
                error=result.get("message", "Không rõ lỗi"),
            )
        else:
            _finish_background_activity(
                activity,
                "completed",
                details={
                    "model_version": result.get("model_version") if isinstance(result, dict) else None,
                    "forecast_date": result.get("forecast_date") if isinstance(result, dict) else None,
                    "chatgpt_advisor_status": (
                        (result.get("chatgpt_advisor") or {}).get("status")
                        if isinstance(result, dict) else None
                    ),
                    "chatgpt_advisor_model": (
                        (result.get("chatgpt_advisor") or {}).get("model")
                        if isinstance(result, dict) else None
                    ),
                },
            )
            _set_gold_prediction_status(
                status="completed",
                stage="Hoàn tất phân tích và cập nhật dự báo",
                progress=100,
                error=None,
            )
        return result
    except Exception as exc:
        with gold_prediction_result_lock:
            gold_prediction_result = None
        _finish_background_activity(activity, "error", error=exc)
        _set_gold_prediction_status(
            status="error",
            stage="Phân tích thất bại",
            progress=100,
            error=exc,
        )
        raise


def _consume_gold_prediction_task(task):
    """Read a background task exception so it cannot become an unhandled task."""
    try:
        task.result()
    except Exception as task_err:
        print(f"Gold prediction background task failed: {task_err}")


def _queue_gold_prediction_job(source="manual"):
    """Queue one in-process Gold prediction job and return its status.

    The caller must be inside the event loop.  A separate lock prevents two
    tabs from scheduling duplicate cold-start training at the same time.
    """
    global gold_prediction_task, gold_prediction_result

    with gold_prediction_job_lock:
        current_status = _get_gold_prediction_status()
        if current_status["status"] == "running":
            return current_status

        with gold_prediction_result_lock:
            gold_prediction_result = None
        _set_gold_prediction_status(
            status="running",
            stage="Đang xếp lịch phân tích AI",
            progress=1,
            error=None,
        )
        gold_prediction_task = asyncio.create_task(
            asyncio.to_thread(_run_gold_predictions_with_status, source)
        )
        gold_prediction_task.add_done_callback(_consume_gold_prediction_task)
        return _get_gold_prediction_status()


def _get_vietlott_training_status(product):
    default = {
        "status": "idle",
        "stage": "Chưa train lại model",
        "progress": 0,
        "started_at": None,
        "finished_at": None,
        "error": None,
    }
    with vietlott_training_lock:
        local_status = dict(vietlott_training_status.get(product, default))

    # The worker is a separate process, so its progress is persisted in
    # MongoDB rather than relying on the parent process's Python dictionary.
    # A short server-side timeout keeps this status endpoint from becoming a
    # second source of request stalls when Mongo itself is unavailable.
    try:
        persisted = vietlott.db[VIETLOTT_TRAINING_STATUS_COLLECTION].find_one(
            {"_id": product},
            {"_id": 0},
            max_time_ms=2000,
        )
        if persisted:
            local_updated_at = str(local_status.get("updated_at") or "")
            persisted_updated_at = str(persisted.get("updated_at") or "")
            # A parent watcher can receive a newer IPC update even when the
            # worker's Mongo write is delayed. Do not let an older document
            # overwrite that newer in-memory status.
            if not local_updated_at or persisted_updated_at >= local_updated_at:
                local_status.update(persisted)
                with vietlott_training_lock:
                    vietlott_training_status[product] = dict(local_status)
    except Exception as status_err:
        print(f"Vietlott status read failed for {product}: {status_err}")

    # Reconcile a worker that exited unexpectedly. This prevents a stale
    # "running" state after an OOM kill or a process crash.
    worker = vietlott_training_processes.get(product)
    scheduled_worker_active = bool(
        (active_vietlott_proc and active_vietlott_proc.is_alive())
        or (active_vietlott_535_proc and active_vietlott_535_proc.is_alive())
    )
    if worker is None and local_status.get("status") == "running" and not scheduled_worker_active:
        return _set_vietlott_training_status(
            product,
            status="error",
            stage="Train trước đã bị gián đoạn",
            progress=local_status.get("progress", 0),
            error="Không còn worker đang chạy cho job này.",
        )
    if worker is not None and not worker.is_alive() and local_status.get("status") == "running":
        worker.join(timeout=0)
        return _set_vietlott_training_status(
            product,
            status="error",
            stage="Worker train đã dừng bất thường",
            progress=local_status.get("progress", 0),
            error=f"Worker kết thúc với exit code {worker.exitcode}.",
        )
    return local_status


def _persist_vietlott_training_status(product, status):
    """Persist a JSON-safe training status document for the worker/API."""
    document = {"_id": product, **dict(status), "updated_at": datetime.now().isoformat()}
    try:
            vietlott.db[VIETLOTT_TRAINING_STATUS_COLLECTION].replace_one(
                {"_id": product},
                document,
                upsert=True,
            )
    except Exception as status_err:
        # Status persistence must never terminate the actual training job.
        print(f"Vietlott status write failed for {product}: {status_err}")


def _set_vietlott_training_status(product, status=None, stage=None, progress=None, error=None):
    with vietlott_training_lock:
        current = vietlott_training_status.setdefault(product, {
            "status": "idle",
            "stage": "Chưa train lại model",
            "progress": 0,
            "started_at": None,
            "finished_at": None,
            "error": None,
        })
        was_running = current["status"] == "running"
        if status is not None:
            current["status"] = status
        if stage is not None:
            current["stage"] = stage
        if progress is not None:
            next_progress = max(0, min(100, int(progress)))
            # A nested training phase can report a rounded value that is
            # slightly lower than the previous phase. Keep the UI monotonic
            # while one retraining job is running.
            if was_running or status == "running":
                next_progress = max(current["progress"], next_progress)
            current["progress"] = next_progress
        if error is not None:
            current["error"] = str(error)
        elif status == "running":
            current["error"] = None
        if status == "running" and not was_running:
            current["started_at"] = datetime.now().isoformat()
            current["finished_at"] = None
        if status in {"completed", "error"}:
            current["finished_at"] = datetime.now().isoformat()
        current["updated_at"] = datetime.now().isoformat()
        snapshot = dict(current)
    _persist_vietlott_training_status(product, snapshot)
    return snapshot


def _run_vietlott_training_process(product, progress_queue=None, activity_source="manual"):
    """Run one game's CPU-heavy retrain outside the Uvicorn process.

    ``asyncio.to_thread`` still lets Python-heavy feature engineering compete
    for the GIL with Uvicorn. A process gives the web server its own scheduler
    and keeps liveness/status requests responsive while backtests run.
    """
    worker_client = None
    status_collection = None
    activity = None
    started_at = datetime.now().isoformat()
    last_status_write = 0.0

    def persist_worker_status(status, stage, progress, error=None, finished_at=None, force=False):
        nonlocal last_status_write
        now = time.monotonic()
        if not force and (now - last_status_write) < VIETLOTT_STATUS_WRITE_INTERVAL_SECONDS:
            return
        last_status_write = now
        document = {
            "_id": product,
            "status": status,
            "stage": stage,
            "progress": max(0, min(100, int(progress))),
            "started_at": started_at,
            "finished_at": finished_at,
            "error": str(error) if error else None,
            "worker_pid": os.getpid(),
            "updated_at": datetime.now().isoformat(),
        }
        # IPC keeps progress visible even if the worker is blocked while
        # opening Mongo or if status persistence itself is unavailable.
        if progress_queue is not None:
            try:
                progress_queue.put_nowait(document)
            except Exception as queue_err:
                print(f"Vietlott worker progress queue failed for {product}: {queue_err}")
        if status_collection is None:
            return
        try:
            status_collection.replace_one(
                {"_id": product},
                document,
                upsert=True,
            )
        except Exception as status_err:
            print(f"Vietlott worker status write failed for {product}: {status_err}")

    # Report before touching Mongo so the parent can distinguish a worker that
    # started from one that never got scheduled.
    persist_worker_status("running", "Worker train đã khởi động", 2, force=True)

    try:
        # Do not use a MongoClient inherited across fork. It can deadlock when
        # the parent had an active socket at the moment the worker started.
        import vietlott as worker_vietlott
        from pymongo import MongoClient

        try:
            worker_vietlott.client.close()
        except Exception:
            pass
        worker_client = MongoClient(
            worker_vietlott.mongo_uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=3000,
            socketTimeoutMS=3000,
            waitQueueTimeoutMS=3000,
        )
        worker_vietlott.client = worker_client
        worker_vietlott.db = worker_client["stock_analytics"]
        status_collection = worker_vietlott.db[VIETLOTT_TRAINING_STATUS_COLLECTION]
        activity = _start_background_activity(
            task="Cào dữ liệu + training Vietlott",
            kind="sync_training",
            source=activity_source,
            category="vietlott",
            product=product,
            db_module=worker_vietlott,
            details={"progress": 2},
        )

        persist_worker_status(
            "running",
            "Đang bắt đầu train lại model",
            2,
            force=True,
        )
        def report_worker_progress(stage, progress):
            persist_worker_status("running", stage, progress)
            if progress >= 18:
                _mark_background_activity_phase(activity, "crawl")
            if progress >= 34:
                _mark_background_training_start(activity)

        result = worker_vietlott.sync_vietlott_data(
            product,
            progress_callback=report_worker_progress,
        )
        if not isinstance(result, dict) or result.get("forecast") is None:
            raise RuntimeError("Không tạo được forecast cho game này; kiểm tra dữ liệu hoặc log train.")
        _finish_background_activity(
            activity,
            "completed",
            details={
                "progress": 100,
                "total_draws": result.get("total_draws"),
                "chatgpt_status": ((result.get("forecast") or {}).get("chatgpt_advisor") or {}).get("status"),
                "chatgpt_model": ((result.get("forecast") or {}).get("chatgpt_advisor") or {}).get("model"),
            },
        )
        persist_worker_status(
            "completed",
            "Train lại hoàn tất và đã lưu kết quả",
            100,
            finished_at=datetime.now().isoformat(),
            force=True,
        )
        return result
    except Exception as exc:
        _finish_background_activity(activity, "error", error=exc)
        persist_worker_status(
            "error",
            "Train lại thất bại",
            100,
            error=exc,
            finished_at=datetime.now().isoformat(),
            force=True,
        )
        print(f"Vietlott retraining failed for {product}: {exc}")
        raise
    finally:
        if worker_client is not None:
            try:
                worker_client.close()
            except Exception:
                pass


def _watch_vietlott_training_process(product, worker, progress_queue):
    """Relay child progress, reap the worker, and surface crashes."""
    def apply_update(update):
        if not isinstance(update, dict):
            return
        _set_vietlott_training_status(
            product,
            status=update.get("status"),
            stage=update.get("stage"),
            progress=update.get("progress"),
            error=update.get("error"),
        )

    while worker.is_alive():
        try:
            apply_update(progress_queue.get(timeout=0.5))
        except QueueEmpty:
            continue
        except (EOFError, OSError):
            break
    worker.join()
    while True:
        try:
            apply_update(progress_queue.get_nowait())
        except QueueEmpty:
            break
        except (EOFError, OSError):
            break

    with vietlott_training_lock:
        if vietlott_training_processes.get(product) is worker:
            vietlott_training_processes.pop(product, None)
        if vietlott_training_progress_queues.get(product) is progress_queue:
            vietlott_training_progress_queues.pop(product, None)
    latest = _get_vietlott_training_status(product)
    if latest.get("status") == "running":
        _set_vietlott_training_status(
            product,
            status="error",
            stage="Worker train đã dừng bất thường",
            progress=latest.get("progress", 0),
            error=f"Worker kết thúc với exit code {worker.exitcode}.",
        )


def _queue_vietlott_training_job(product, activity_source="manual"):
    with vietlott_training_lock:
        current = _get_vietlott_training_status(product)
        if current["status"] == "running":
            worker = vietlott_training_processes.get(product)
            if worker is not None and worker.is_alive():
                return current
            # A persisted running state without a live worker belongs to a
            # previous/crashed app process. Mark it stale before allowing a
            # fresh job, rather than silently creating duplicate work.
            _set_vietlott_training_status(
                product,
                status="error",
                stage="Train trước đã bị gián đoạn",
                progress=current.get("progress", 0),
                error="Không còn worker đang chạy cho job này.",
            )
        _set_vietlott_training_status(
            product,
            status="running",
            stage="Đang xếp lịch train lại model",
            progress=1,
            error=None,
        )
        progress_queue = Queue()
        worker = Process(
            target=_run_vietlott_training_process,
            args=(product, progress_queue, activity_source),
        )
        try:
            worker.start()
        except Exception as start_err:
            _set_vietlott_training_status(
                product,
                status="error",
                stage="Không khởi động được worker train",
                progress=100,
                error=start_err,
            )
            raise
        vietlott_training_processes[product] = worker
        vietlott_training_progress_queues[product] = progress_queue
        threading.Thread(
            target=_watch_vietlott_training_process,
            args=(product, worker, progress_queue),
            name=f"vietlott-training-watch-{product}",
            daemon=True,
        ).start()
        return _get_vietlott_training_status(product)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/gold", response_class=HTMLResponse)
async def gold_price_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/gold/prediction", response_class=HTMLResponse)
async def gold_prediction_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")



@app.get("/vietlott", response_class=HTMLResponse)
async def vietlott_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")







def _refresh_gold_data_for_activity(source):
    """Refresh Gold data and persist the duration of the training-data crawl."""
    activity = _start_background_activity(
        task="Cào dữ liệu giá vàng để train",
        kind="crawl",
        source=source,
        category="gold",
        details={"bypass_cache": True},
    )
    try:
        result = crawler.fetch_gold_prices(bypass_cache=True)
        _finish_background_activity(
            activity,
            "completed",
            details={"has_data": bool(result)},
        )
        return result
    except Exception as exc:
        _finish_background_activity(activity, "error", error=exc)
        raise


def _refresh_calendar_results_for_activity(source):
    """Force-refresh released US economic-calendar results and log it."""
    activity = _start_background_activity(
        task="Cập nhật kết quả lịch sự kiện kinh tế Mỹ",
        kind="crawl",
        source=source,
        category="calendar",
        details={"force": True},
    )
    try:
        result = crawler.refresh_economic_calendar_results(force=True)
        if result.get("status") == "error":
            error = result.get("message", "Không thể cập nhật kết quả lịch sự kiện.")
            _finish_background_activity(activity, "error", error=error, details=result)
            return result
        _finish_background_activity(
            activity,
            "completed",
            details={
                "updated": int(result.get("updated", 0) or 0),
                "status": result.get("status"),
                "checked_at": result.get("checked_at"),
                "source_errors": result.get("source_errors", []),
            },
        )
        return result
    except Exception as exc:
        _finish_background_activity(activity, "error", error=exc)
        raise


def _run_gold_month_backtest_for_activity(year, month):
    """Run one month of walk-forward validation and record its duration."""
    activity = _start_background_activity(
        task="Backtest so sánh mô hình giá vàng",
        kind="training",
        source="manual_backtest",
        category="gold",
        details={"year": year, "month": month},
    )
    try:
        result = crawler.run_gold_month_backtest(year=year, month=month)
        if result.get("status") == "error":
            _finish_background_activity(
                activity,
                "error",
                error=result.get("message", "Backtest thất bại."),
                details=result,
            )
        else:
            _finish_background_activity(
                activity,
                "completed",
                details={
                    "year": result.get("year"),
                    "month": result.get("month"),
                    "evaluated_sessions": result.get("evaluated_sessions", 0),
                    "best_model": (result.get("best_model") or {}).get("model"),
                },
            )
        return result
    except Exception as exc:
        _finish_background_activity(activity, "error", error=exc)
        raise


def run_gold_crawler_sync(loop_count):
    try:
        print("Background crawler: Scraping live gold prices, crude oil, and macro indicators...")
        # bypass_cache=True forces crawler to crawl new data and update MongoDB cache doc
        crawler.fetch_gold_prices(bypass_cache=True)
        print("Background crawler: Successfully updated MongoDB cache.")
        
        # Update macro history from yfinance + FRED real yield every 60 minutes or on startup
        if loop_count % 60 == 0:
            try:
                print("Background crawler: Updating macro variables (DXY, US10Y, VIX, Brent, EURUSD, XAG, FRED real yield)...")
                crawler.update_all_macro(days=30)
            except Exception as macro_err:
                print(f"Error updating macro history in background: {macro_err}")
        

    except Exception as e:
        print(f"Error in background gold crawler worker thread: {e}")

active_gold_proc = None
active_vietlott_proc = None
active_vietlott_535_proc = None

def run_gold_crawler_process(loop_count):
    try:
        import crawler
        # Close inherited connection and open a fresh one in the child process to avoid deadlock
        try:
            crawler.client.close()
        except Exception:
            pass
        crawler.client = crawler.MongoClient(crawler.mongo_uri)
        crawler.db = crawler.client["stock_analytics"]
        
        run_gold_crawler_sync(loop_count)
    except Exception as e:
        print(f"Error in background gold crawler process: {e}")

def run_vietlott_crawler_process(activity_source="scheduled_00:00"):
    try:
        import vietlott
        # Close inherited connection and open a fresh one in the child process to avoid deadlock
        try:
            vietlott.client.close()
        except Exception:
            pass
        vietlott.client = vietlott.MongoClient(vietlott.mongo_uri)
        vietlott.db = vietlott.client["stock_analytics"]
        
        print("Background crawler: Synchronizing Vietlott statistics...")
        # Run the complete crawl + ML retrain sequentially so the VM does not
        # launch three CPU-heavy jobs at once. Reuse the isolated training
        # worker so each product's progress is visible through the status API.
        for product in ("power655", "mega645", "power535"):
            try:
                print(f"Background crawler: Full crawl + ML retrain started for {product}.")
                _run_vietlott_training_process(
                    product,
                    activity_source=activity_source,
                )
            except Exception as product_err:
                # Keep the nightly job moving if one product has a transient
                # source/model failure; that product's status is persisted as
                # error by the isolated worker above.
                print(f"Background crawler: Full retrain failed for {product}: {product_err}")
        print("Background crawler: Successfully finished the daily Vietlott job.")
    except Exception as e:
        print(f"Error in background Vietlott crawler process: {e}")


def run_vietlott_535_draw_refresh_process(activity_source="scheduled_13:30"):
    """Crawl, retrain and forecast the next Power 5/35 draw."""
    try:
        print(
            "Background Vietlott crawler: crawling, retraining and forecasting "
            f"Power 5/35 for the next draw ({activity_source})."
        )
        # Reuse the isolated full training worker so the crawl/training
        # timestamps, progress, forecast and Codex roundtable are recorded in
        # the same activity history as a manual retrain.
        result = _run_vietlott_training_process(
            "power535",
            activity_source=activity_source,
        )
        print(
            "Background Vietlott crawler: Power 5/35 retraining completed "
            f"with {result.get('total_draws', 0)} draws and a next-draw forecast."
        )
        return result
    except Exception as exc:
        print(f"Error retraining scheduled Power 5/35 forecast: {exc}")

# Background gold & macro crawler task running continuously in the background
async def gold_crawler_background_loop():
    global active_gold_proc
    print("Background gold & macro crawler loop registered.")
    # Wait 5 seconds to let uvicorn startup completely and open the port
    await asyncio.sleep(5)
    print("Background gold & macro crawler loop started...")
    loop_count = 0
    while True:
        try:
            if active_gold_proc and active_gold_proc.is_alive():
                print("Background gold crawler process from previous loop is still running. Skipping this iteration.")
            else:
                active_gold_proc = Process(target=run_gold_crawler_process, args=(loop_count,))
                active_gold_proc.start()
        except Exception as e:
            print(f"Error starting background gold crawler process: {e}")
        loop_count += 1
        # Sleep for 60 seconds (1 minute)
        await asyncio.sleep(60)


async def _wait_until_next_vietnam_time(hour, minute=0):
    """Wait for the next scheduled time in Vietnam, independent of VM timezone."""
    now_vn = datetime.now(VIETNAM_TIMEZONE)
    target = now_vn.replace(
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0,
    )
    # A restart in the first minute after the scheduled time should still run
    # today's job. Later restarts wait for the next day's scheduled run.
    if now_vn >= target + timedelta(minutes=1):
        target += timedelta(days=1)

    while True:
        seconds_until_target = (target - datetime.now(VIETNAM_TIMEZONE)).total_seconds()
        if seconds_until_target <= 0:
            return
        await asyncio.sleep(min(seconds_until_target, 60))


async def _wait_until_next_vietnam_midnight():
    """Wait until the next 00:00 in Vietnam."""
    await _wait_until_next_vietnam_time(0, 0)


async def gold_prediction_background_loop():
    """Run one fresh Gold AI prediction every day at 05:00 Vietnam time."""
    print(
        "Background Gold prediction scheduler started "
        f"({GOLD_PREDICTION_HOUR:02d}:{GOLD_PREDICTION_MINUTE:02d} Asia/Ho_Chi_Minh)..."
    )
    while True:
        await _wait_until_next_vietnam_time(
            GOLD_PREDICTION_HOUR,
            GOLD_PREDICTION_MINUTE,
        )

        print("Background Gold prediction: 05:00 reached; refreshing data and queueing AI prediction.")
        try:
            # Refresh the source data immediately before inference so a stale
            # cache cannot make the scheduled forecast use yesterday's price.
            try:
                await asyncio.to_thread(
                    _refresh_gold_data_for_activity,
                    "scheduled_05:00",
                )
            except Exception as refresh_err:
                # The prediction pipeline can still use the latest persisted
                # history/cache if this one source refresh fails.
                print(f"Background Gold prediction data refresh failed: {refresh_err}")

            scheduled_status = _queue_gold_prediction_job(source="scheduled_05:00")
            print(
                "Background Gold prediction: AI job queued "
                f"with status {scheduled_status.get('status')}."
            )
        except Exception as prediction_err:
            print(f"Error starting scheduled Gold prediction: {prediction_err}")

        # Avoid immediately treating the same 05:00 window as a second run if
        # the job was queued very quickly.
        await asyncio.sleep(61)


async def calendar_results_background_loop():
    """Refresh released US economic-calendar results every day at 03:00 VN."""
    print(
        "Background US economic calendar scheduler started "
        f"({CALENDAR_RESULTS_UPDATE_HOUR:02d}:{CALENDAR_RESULTS_UPDATE_MINUTE:02d} "
        "Asia/Ho_Chi_Minh)..."
    )
    while True:
        await _wait_until_next_vietnam_time(
            CALENDAR_RESULTS_UPDATE_HOUR,
            CALENDAR_RESULTS_UPDATE_MINUTE,
        )
        print("Background US economic calendar: 03:00 reached; refreshing released results.")
        try:
            result = await asyncio.to_thread(
                _refresh_calendar_results_for_activity,
                "scheduled_03:00",
            )
            print(
                "Background US economic calendar: refresh finished "
                f"with status {result.get('status')} and {result.get('updated', 0)} updates."
            )
        except Exception as refresh_err:
            print(f"Background US economic calendar refresh failed: {refresh_err}")
        await asyncio.sleep(61)


async def vietlott_crawler_background_loop():
    global active_vietlott_proc
    print("Background Vietlott crawler scheduler started (00:00 Asia/Ho_Chi_Minh)...")
    while True:
        await _wait_until_next_vietnam_midnight()

        print(
            "Background Vietlott crawler: midnight reached; "
            "starting full crawl + ML retraining for Power 6/55, Mega 6/45, and Power 5/35."
        )
        try:
            # Do not drop a scheduled run if a manual retrain or a previous
            # full sync is still running; wait until the resource is free.
            while True:
                manual_training_active = any(
                    worker.is_alive()
                    for worker in vietlott_training_processes.values()
                )
                previous_sync_active = bool(
                    active_vietlott_proc and active_vietlott_proc.is_alive()
                )
                if not manual_training_active and not previous_sync_active:
                    break
                active_work = "manual training" if manual_training_active else "previous full sync"
                print(f"Background Vietlott crawler: waiting for {active_work} to finish...")
                await asyncio.sleep(60)

            active_vietlott_proc = Process(
                target=run_vietlott_crawler_process,
                args=("scheduled_00:00",),
            )
            active_vietlott_proc.start()
            print("Background Vietlott crawler: full daily job started.")
        except Exception as e:
            print(f"Error starting scheduled Vietlott crawler: {e}")


async def vietlott_power535_draw_background_loop(hour, minute):
    """Retrain Power 5/35 after each of its two daily draw times."""
    global active_vietlott_535_proc
    schedule_label = f"scheduled_{hour:02d}:{minute:02d}"
    print(
        "Background Power 5/35 draw crawler scheduler started "
        f"({hour:02d}:{minute:02d} Asia/Ho_Chi_Minh)..."
    )
    while True:
        await _wait_until_next_vietnam_time(hour, minute)
        print(
            "Background Power 5/35 draw crawler: "
            f"{hour:02d}:{minute:02d} reached; refreshing result data."
        )
        try:
            # Never overlap this retrain with the midnight full sync, a manual
            # retrain, or the other scheduled draw retrain.
            while True:
                manual_training_active = any(
                    worker.is_alive()
                    for worker in vietlott_training_processes.values()
                )
                previous_full_active = bool(
                    active_vietlott_proc and active_vietlott_proc.is_alive()
                )
                previous_draw_active = bool(
                    active_vietlott_535_proc and active_vietlott_535_proc.is_alive()
                )
                if not manual_training_active and not previous_full_active and not previous_draw_active:
                    break
                print("Background Power 5/35 draw crawler: waiting for another Vietlott job...")
                await asyncio.sleep(60)

            active_vietlott_535_proc = Process(
                target=run_vietlott_535_draw_refresh_process,
                args=(schedule_label,),
            )
            active_vietlott_535_proc.start()
            print(
                "Background Power 5/35 draw crawler: retrain + next-draw forecast started "
                f"for {schedule_label}."
            )
        except Exception as e:
            print(f"Error starting scheduled Power 5/35 draw refresh: {e}")
        await asyncio.sleep(61)


@app.on_event("startup")
async def startup_event():
    # Start the background tasks
    asyncio.create_task(gold_crawler_background_loop())
    asyncio.create_task(gold_prediction_background_loop())
    asyncio.create_task(calendar_results_background_loop())
    asyncio.create_task(vietlott_crawler_background_loop())
    asyncio.create_task(vietlott_power535_draw_background_loop(13, 30))
    asyncio.create_task(vietlott_power535_draw_background_loop(21, 30))

@app.get("/api/gold/prediction-history")
async def get_gold_prediction_history():
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        history = await loop.run_in_executor(
            None,
            crawler.get_gold_prediction_history_with_explanations
        )
        return JSONResponse(content=jsonable_encoder(history))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gold/model-comparison")
async def get_gold_model_comparison(days: int = 30):
    """Return rolling accuracy metrics for the resolved Gold predictions."""
    try:
        result = await asyncio.to_thread(crawler.calculate_gold_model_comparison, days)
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message"))
        return JSONResponse(content=jsonable_encoder(result))
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/api/gold/backtest")
async def backtest_gold_month(year: Optional[int] = None, month: int = 9):
    """Queue a one-step walk-forward Gold backtest for a selected month.

    Backtesting all models for a month can exceed the reverse proxy timeout,
    so this endpoint returns a job id immediately.  The UI polls the status
    endpoint until the worker has stored the result.
    """
    try:
        if year is None:
            now_vn = datetime.now(VIETNAM_TIMEZONE)
            # Default to the current year. Partial current-month backtests are
            # valid and clearer than silently switching to last year.
            year = now_vn.year
        if month < 1 or month > 12 or year < 2000 or year > 2100:
            raise HTTPException(status_code=400, detail="Tháng hoặc năm backtest không hợp lệ.")
        job = _queue_gold_month_backtest_job(int(year), int(month))
        return JSONResponse(status_code=202, content=jsonable_encoder(job))
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/api/gold/backtest-status/{job_id}")
async def get_gold_month_backtest_status(job_id: str):
    """Return the status/result of a queued monthly Gold backtest."""
    job = _get_gold_backtest_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Không tìm thấy job backtest hoặc job đã hết hạn.")
    return JSONResponse(content=jsonable_encoder(job))


@app.get("/api/gold/backtest-latest")
async def get_latest_gold_month_backtest(year: Optional[int] = None, month: int = 9):
    """Load the last complete monthly backtest for the selected month."""
    try:
        if year is None:
            now_vn = datetime.now(VIETNAM_TIMEZONE)
            year = now_vn.year
        if month < 1 or month > 12 or year < 2000 or year > 2100:
            raise HTTPException(status_code=400, detail="Tháng hoặc năm backtest không hợp lệ.")
        result = _latest_gold_backtest_result(int(year), int(month))
        if not result:
            return JSONResponse(content={
                "status": "not_found",
                "year": int(year),
                "month": int(month),
                "month_label": f"{int(month):02d}/{int(year)}",
            })
        return JSONResponse(content=jsonable_encoder(result))
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

@app.post("/api/gold/backfill")
async def backfill_gold_history(days: int = 60):
    """Backfill T+1 prediction history for the last `days` trading days."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: crawler.backfill_gold_predictions_history(days=days)
        )
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message"))
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/gold/learn-from-mistake")
async def learn_from_mistake(req: LearnRequest):
    try:
        from datetime import datetime
        date_str = req.date.strip()
        model_key = req.model.strip()
        
        # 1. Fetch prediction record
        pred_record = crawler.db.gold_predictions_history.find_one({"date": date_str})
        if not pred_record:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy bản ghi dự đoán cho ngày {date_str}.")
            
        actual_price = pred_record.get("actual_price")
        if not actual_price or actual_price == 0.0:
            raise HTTPException(status_code=400, detail=f"Phiên giao dịch ngày {date_str} chưa có giá thực tế để đối chiếu.")
            
        # 2. Compute prediction error
        models_preds = pred_record.get("models", {})
        predicted_price = models_preds.get(model_key)
        if predicted_price is None:
            raise HTTPException(status_code=400, detail=f"Không có giá dự đoán của mô hình {model_key} cho ngày {date_str}.")
            
        error = actual_price - predicted_price
        
        # 3. Save bias correction adjustment
        adjustment_doc = {
            "_id": "latest_adjustment",
            "date": date_str,
            "model": model_key,
            "error": float(error),
            "timestamp": datetime.now()
        }
        crawler.db.gold_model_adjustments.replace_one({"_id": "latest_adjustment"}, adjustment_doc, upsert=True)
        
        # 4. Update tomorrow predictions history in database
        try:
            crawler.update_gold_predictions_history()
        except Exception as update_err:
            print(f"Error updating predictions after learning: {update_err}")
            
        return {
            "status": "success",
            "date": date_str,
            "model": model_key,
            "error": float(error),
            "adjustment": float(error * 0.8)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/gold/optimize")
async def optimize_gold_models():
    """Manually triggers time-series cross-validation hyperparameter search."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        tuning_result = await loop.run_in_executor(None, crawler.tune_and_save_gold_hyperparameters)
        if not isinstance(tuning_result, dict) or tuning_result.get("status") == "error":
            raise RuntimeError((tuning_result or {}).get("message", "Fine-tuning thất bại"))
        
        # Reload tomorrow predictions history using the newly optimized hyperparameters
        await loop.run_in_executor(None, crawler.update_gold_predictions_history)
        
        params_doc = crawler.db.gold_model_hyperparameters.find_one({"type": "gold_params"})
        if params_doc:
            return {
                "status": "success",
                "updated_at": params_doc.get("updated_at"),
                "params": params_doc.get("params"),
                "diagnostics": params_doc.get("diagnostics"),
            }
        return {"status": "error", "message": "Failed to retrieve optimized parameters."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _load_gold_data_sync():
    """Build the fast gold market payload outside the asyncio event loop."""
    # Load from MongoDB latest cache
    cached = crawler.db.gold_prices.find_one({"_id": "latest_prices"})

    # If cache is missing or invalid (missing key prices), force a crawl.
    if not cached or not cached.get("world", {}).get("price") or not cached.get("domestic", {}).get("sjc_bar", {}).get("buy"):
        print("Cached gold prices missing or invalid. Performing sync fetch...")
        cached = crawler.fetch_gold_prices(bypass_cache=True)

    if not cached or not cached.get("world", {}).get("price") or not cached.get("domestic", {}).get("sjc_bar", {}).get("buy"):
        raise HTTPException(status_code=500, detail="Không thể tải dữ liệu giá vàng.")

    data = cached.copy()
    data.pop("_id", None)
    data.pop("fetched_at", None)

    try:
        data["ticks"] = crawler.get_gold_ticks()
        # The DB tick writer may be a few minutes behind the live cache. Append
        # the current quote to the response so the 24-hour chart never ends at
        # an old price while the price cards already show a newer value.
        live_world = data.get("world") or {}
        try:
            live_price = float(str(live_world.get("price", "0")).replace(",", ""))
        except (TypeError, ValueError):
            live_price = 0.0
        live_time = str(live_world.get("time", ""))
        try:
            live_dt = datetime.strptime(live_time, "%H:%M:%S %d/%m/%Y")
        except (TypeError, ValueError):
            live_dt = None
        last_dt = None
        if data["ticks"]:
            try:
                last_dt = datetime.strptime(
                    str(data["ticks"][-1].get("timestamp", "")),
                    "%Y-%m-%d %H:%M:%S",
                )
            except (TypeError, ValueError):
                last_dt = None
        if live_price > 0 and live_dt is not None and (last_dt is None or live_dt > last_dt):
            data["ticks"].append({
                "timestamp": live_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "world_price": live_price,
                "time": live_dt.strftime("%d/%m"),
                "is_live": True,
            })
    except Exception as ticks_err:
        print(f"Error fetching gold ticks: {ticks_err}")
        data["ticks"] = []

    # Historical macro data is intentionally part of the fast market payload.
    # The charts must not wait for model training just to get DXY/Brent/DJI/SPX.
    def _clean_history(rows, fields):
        cleaned = []
        for row in rows:
            date_value = row.get("date")
            if not date_value:
                continue
            item = {"date": str(date_value)}
            for field in fields:
                value = row.get(field)
                try:
                    item[field] = float(value) if value is not None else None
                except (TypeError, ValueError):
                    item[field] = None
            cleaned.append(item)
        return cleaned

    try:
        macro_rows = list(
            crawler.db.macro_history.find({}, {"_id": 0})
            .sort("date", -1)
            .limit(180)
        )
        macro_rows.reverse()
        data["macro_history"] = _clean_history(
            macro_rows,
            [
                "dxy", "brent", "dji", "spx", "gld", "gld_trust", "us10y", "vix",
                "real_yield", "t10yie",
                "pce_headline_yoy", "pce_core_yoy", "pce_headline_mom", "pce_core_mom",
            ],
        )

        # Refresh the policy calendar shape for older cache documents that
        # were created before FED speeches/Jackson Hole were added.
        try:
            data["calendar"] = crawler.get_us_economic_calendar()
        except Exception as calendar_error:
            print(f"Error refreshing economic calendar for dashboard: {calendar_error}")

        macro_indicators = data.get("macro_indicators") or {}
        if not macro_indicators.get("pce") or not macro_indicators.get("core_pce"):
            try:
                macro_indicators = crawler.fetch_us_macro_indicators()
                data["macro_indicators"] = macro_indicators
            except Exception as indicators_error:
                print(f"Error refreshing macro indicators for FED outlook: {indicators_error}")
        data["fed_policy_outlook"] = crawler.build_fed_policy_outlook(
            macro_indicators=macro_indicators,
            macro_history=macro_rows[-2:],
            calendar=data.get("calendar") or [],
            market_signals=data.get("market_signals") or [],
        )
    except Exception as macro_history_err:
        print(f"Error loading macro history for dashboard: {macro_history_err}")
        data["macro_history"] = []

    try:
        gold_rows = list(
            crawler.db.gold_history.find({}, {"_id": 0})
            .sort("date", -1)
            .limit(180)
        )
        gold_rows.reverse()
        data["gold_history"] = _clean_history(
            gold_rows,
            ["world_price", "world_price_vnd", "sjc_bar_buy", "sjc_bar_sell"],
        )
    except Exception as gold_history_err:
        print(f"Error loading gold history for dashboard: {gold_history_err}")
        data["gold_history"] = []

    return data


def _gold_forecast_staleness(prediction):
    """Compare a forecast's training anchor with the latest cached spot price."""
    base_price = float(
        (prediction or {}).get("forecast_base_price")
        or (prediction or {}).get("current_price")
        or 0.0
    )
    live_price = 0.0
    try:
        cached = crawler.db.gold_prices.find_one(
            {"_id": "latest_prices"}, {"_id": 0, "world.price": 1}
        ) or {}
        raw_price = ((cached.get("world") or {}).get("price"))
        live_price = float(str(raw_price).replace(",", ""))
    except (TypeError, ValueError):
        live_price = 0.0

    drift_pct = (
        abs(live_price - base_price) / base_price * 100.0
        if base_price > 0 and live_price > 0 else 0.0
    )
    generated_at = (prediction or {}).get("forecast_generated_at")
    age_minutes = None
    if generated_at:
        try:
            age_minutes = max(
                0.0,
                (datetime.now() - datetime.fromisoformat(str(generated_at))).total_seconds() / 60.0,
            )
        except (TypeError, ValueError):
            age_minutes = None

    return {
        "base_price": base_price,
        "live_price": live_price,
        "drift_pct": float(drift_pct),
        "age_minutes": age_minutes,
        "is_stale": bool(drift_pct >= 0.15 or (age_minutes is not None and age_minutes >= 360.0)),
        # Retrain only after a meaningful market move. Age alone should not
        # repeatedly consume CPU when the price regime has not changed.
        "requires_refresh": bool(drift_pct >= 0.35),
    }


@app.get("/api/gold")
async def get_gold(model: str = "random_forest"):
    try:
        # Prices/news/calendar must be available immediately. Model training
        # is loaded by /api/gold/predictions so a slow ML job cannot keep the
        # whole Gold dashboard on its spinner.
        data = await asyncio.to_thread(_load_gold_data_sync)
        response = JSONResponse(content=data)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gold/predictions")
async def get_gold_predictions_api():
    """Start/return the heavier Gold models without holding an HTTP request open.

    The reverse proxy in front of the VM has a request timeout shorter than a
    cold run of all five models.  Always queue a cold run and let the frontend
    poll /api/gold/prediction-status, then return the cached result here.
    """
    try:
        current_status = _get_gold_prediction_status()
        with gold_prediction_result_lock:
            cached_result = gold_prediction_result

        if current_status["status"] == "completed" and cached_result is not None:
            staleness = _gold_forecast_staleness(cached_result)
            if staleness["requires_refresh"]:
                queued_status = _queue_gold_prediction_job(source="api_request")
                return JSONResponse(
                    status_code=202,
                    content={
                        "status": "running",
                        "reason": "Giá live đã lệch đáng kể so với giá gốc dự báo",
                        "staleness": staleness,
                        "progress": queued_status,
                    },
                )
            response_payload = dict(cached_result)
            response_payload["staleness"] = staleness
            response = JSONResponse(content=response_payload)
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            return response

        if current_status["status"] == "running":
            return JSONResponse(
                status_code=202,
                content={"status": "running", "progress": current_status},
            )

        queued_status = _queue_gold_prediction_job(source="api_request")
        return JSONResponse(
            status_code=202,
            content={"status": "running", "progress": queued_status},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gold/prediction-status")
async def get_gold_prediction_status_api():
    """Return the current stage of the background Gold AI analysis."""
    return JSONResponse(content=_get_gold_prediction_status())


@app.post("/api/gold/codex/retry", status_code=202)
async def retry_gold_codex():
    """Retry only the Codex chair using the latest saved Gold meeting."""
    try:
        status = _queue_codex_retry("gold")
        return JSONResponse(status_code=202, content=jsonable_encoder(status))
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/api/gold/codex/retry-status")
async def get_gold_codex_retry_status():
    return JSONResponse(content=_get_codex_retry_status("gold"))


@app.post("/api/gold/refresh")
async def refresh_gold():
    """Force-refresh market data and queue one fresh AI prediction job."""
    import asyncio
    try:
        print("Manual refresh triggered: force-crawling gold prices...")
        # Run in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        fresh = await loop.run_in_executor(
            None,
            _refresh_gold_data_for_activity,
            "manual_refresh",
        )
        if not fresh:
            raise HTTPException(status_code=500, detail="Không thể lấy dữ liệu mới nhất.")

        # Re-load from MongoDB (just written by fetch above)
        cached = crawler.db.gold_prices.find_one({"_id": "latest_prices"})
        if not cached:
            raise HTTPException(status_code=500, detail="Lỗi đọc dữ liệu sau khi cập nhật.")

        data = cached.copy()
        data.pop("_id", None)
        data.pop("fetched_at", None)

        # Queue the expensive model work after returning the fresh market
        # payload.  This keeps the manual refresh below reverse-proxy timeout
        # and lets the same status polling UI track the training job.
        data["predictions"] = None
        data["prediction_status"] = _queue_gold_prediction_job(source="manual_refresh")

        try:
            data["ticks"] = crawler.get_gold_ticks()
        except Exception as ticks_err:
            print(f"Error fetching gold ticks: {ticks_err}")
            data["ticks"] = []

        try:
            adj = crawler.db.gold_model_adjustments.find_one({"_id": "latest_adjustment"}, {"_id": 0})
            if adj and "timestamp" in adj and hasattr(adj["timestamp"], "isoformat"):
                adj["timestamp"] = adj["timestamp"].isoformat()
            data["latest_adjustment"] = adj
        except Exception as adj_err:
            print(f"Error fetching latest adjustment: {adj_err}")
            data["latest_adjustment"] = None

        response = JSONResponse(content=data)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/gold/calendar/refresh")
async def refresh_gold_calendar():
    """Force-refresh released economic-calendar results without retraining AI."""
    try:
        result = await asyncio.to_thread(
            _refresh_calendar_results_for_activity,
            "manual_api",
        )
        if result.get("status") == "error":
            raise HTTPException(status_code=502, detail=result.get("message"))
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/api/vietlott/{product}")
async def get_vietlott(product: str):
    product = product.strip().lower()
    try:
        # Cache misses can launch a full crawler + ML forecast. Run it away
        # from the event loop so Gold/API requests remain responsive.
        data = await asyncio.to_thread(vietlott.get_cached_vietlott_data, product)
        return JSONResponse(content=data)
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/vietlott/{product}/codex/retry", status_code=202)
async def retry_vietlott_codex(product: str):
    """Retry only the Codex chair for an existing Vietlott forecast."""
    product = product.strip().lower()
    if product not in vietlott.PRODUCTS:
        raise HTTPException(status_code=400, detail=f"Sản phẩm {product} không được hỗ trợ.")
    try:
        status = _queue_codex_retry("vietlott", product)
        return JSONResponse(status_code=202, content=jsonable_encoder({"product": product, **status}))
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/api/vietlott/{product}/codex/retry-status")
async def get_vietlott_codex_retry_status(product: str):
    product = product.strip().lower()
    if product not in vietlott.PRODUCTS:
        raise HTTPException(status_code=400, detail=f"Sản phẩm {product} không được hỗ trợ.")
    return JSONResponse(content={"product": product, **_get_codex_retry_status(_codex_retry_key("vietlott", product))})


@app.post("/api/vietlott/{product}/retrain")
async def retrain_vietlott(product: str):
    """Queue a fresh data sync and feature/model training for one game."""
    product = product.strip().lower()
    if product not in vietlott.PRODUCTS:
        raise HTTPException(status_code=400, detail=f"Sản phẩm {product} không được hỗ trợ.")
    status = _queue_vietlott_training_job(product)
    return JSONResponse(
        status_code=202,
        content={"status": "running", "product": product, "progress": status},
    )


@app.get("/api/vietlott/{product}/retrain-status")
async def get_vietlott_retrain_status(product: str):
    product = product.strip().lower()
    if product not in vietlott.PRODUCTS:
        raise HTTPException(status_code=400, detail=f"Sản phẩm {product} không được hỗ trợ.")
    status = await asyncio.to_thread(_get_vietlott_training_status, product)
    return JSONResponse(content=status)


class SimulateRequest(BaseModel):
    numbers: List[int]

@app.post("/api/vietlott/{product}/simulate")
async def simulate_vietlott(product: str, req: SimulateRequest):
    product = product.strip().lower()
    if product not in vietlott.PRODUCTS:
        raise HTTPException(status_code=400, detail=f"Sản phẩm {product} không được hỗ trợ.")
    if product not in {"mega645", "power655"}:
        raise HTTPException(status_code=400, detail=f"Sản phẩm {product} chưa hỗ trợ mô phỏng giải thưởng.")
    expected_numbers = vietlott.LOTTERY_CONFIG[product]["num_per_draw"]
    if not req.numbers or len(req.numbers) != expected_numbers:
        raise HTTPException(status_code=400, detail=f"Vui lòng chọn đúng {expected_numbers} con số.")
    try:
        result = vietlott.backtest_user_numbers(product, req.numbers)
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("message"))
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/vietlott/refresh")
async def refresh_vietlott(product: Optional[str] = None):
    """Force synchronize Vietlott products using the local crawler."""
    import asyncio
    try:
        if product:
            product = product.strip().lower()
            if product not in vietlott.PRODUCTS:
                raise HTTPException(status_code=400, detail=f"Sản phẩm {product} không được hỗ trợ.")
            print(f"Manual refresh triggered: force-syncing Vietlott product {product}...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, vietlott.sync_vietlott_data, product)
            return {"status": "success", "results": {product: "success"}}
        else:
            print("Manual refresh triggered: force-syncing all Vietlott products...")
            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(None, vietlott.sync_all)
            return {"status": "success", "results": res}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/control/health")
async def control_health():
    """Unauthenticated liveness endpoint for a VM/Kubernetes probe."""
    return {
        "status": "ok",
        "service": "ai-gold-vietlott",
        "control_api_auth": "disabled",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/control/status")
async def control_status(request: Request):
    """Return runtime, job and cache status without secrets."""
    cache_snapshot = await asyncio.to_thread(_control_cache_snapshot)
    return {
        "status": "ok",
        "runtime_config": _control_runtime_config(),
        "gold_prediction": _get_gold_prediction_status(),
        "vietlott_training": {
            product: _get_vietlott_training_status(product)
            for product in sorted(vietlott.PRODUCTS)
        },
        "cache": cache_snapshot,
    }


@app.get("/api/control/activity-history")
async def get_background_activity_history(limit: int = 80, category: Optional[str] = None):
    """Return recent background crawl/training activity with durations."""
    limit = max(1, min(int(limit), 200))
    query = {}
    if category in {"gold", "vietlott", "calendar"}:
        query["category"] = category
    try:
        records = list(
            crawler.db[BACKGROUND_ACTIVITY_COLLECTION]
            .find(query, {"_id": 0})
            .sort("started_at", -1)
            .limit(limit)
        )
        return JSONResponse(content=jsonable_encoder(records))
    except Exception as activity_err:
        raise HTTPException(status_code=500, detail=str(activity_err))


@app.get("/api/control/config")
async def control_config(request: Request):
    """Read the small, whitelisted set of live tuning controls."""
    return _control_runtime_config()


@app.patch("/api/control/config")
async def patch_control_config(request: Request, payload: ControlConfigPatch):
    """Apply safe tuning controls for subsequent training jobs only.

    This deliberately does not accept arbitrary Python/model parameters. It
    only changes the walk-forward validation window and the per-game training
    history length, both bounded to prevent an accidental runaway job.
    """
    if payload.backtest_window is not None:
        if not 20 <= payload.backtest_window <= vietlott.MAX_BACKTEST_WINDOW:
            raise HTTPException(
                status_code=422,
                detail=f"backtest_window phải nằm trong khoảng 20-{vietlott.MAX_BACKTEST_WINDOW}.",
            )
        os.environ["VIETLOTT_BACKTEST_WINDOW"] = str(payload.backtest_window)

    if payload.training_draws is not None:
        for product, training_draws in payload.training_draws.items():
            product = product.strip().lower()
            if product not in CONTROL_ALLOWED_TRAINING_PRODUCTS:
                raise HTTPException(status_code=422, detail=f"Game không được hỗ trợ: {product}.")
            if not 120 <= training_draws <= 1000:
                raise HTTPException(
                    status_code=422,
                    detail=f"training_draws của {product} phải nằm trong khoảng 120-1000.",
                )
            vietlott.LOTTERY_CONFIG[product]["training_draws"] = int(training_draws)

    return {
        "status": "updated",
        "applies_to": "next training job",
        "runtime_config": _control_runtime_config(),
    }


@app.post("/api/control/gold/retrain", status_code=202)
async def control_retrain_gold(request: Request):
    """Queue a Gold retraining/forecast job."""
    return {
        "status": "accepted",
        "job": _queue_gold_prediction_job(source="control_api"),
    }


@app.get("/api/control/gold/retrain-status")
async def control_gold_retrain_status(request: Request):
    return _get_gold_prediction_status()


@app.post("/api/control/gold/calendar/refresh")
async def control_refresh_gold_calendar(request: Request):
    """Refresh released calendar results through the control API."""
    try:
        result = await asyncio.to_thread(
            _refresh_calendar_results_for_activity,
            "control_api",
        )
        if result.get("status") == "error":
            raise HTTPException(status_code=502, detail=result.get("message"))
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/control/vietlott/{product}/retrain", status_code=202)
async def control_retrain_vietlott(product: str, request: Request):
    """Queue isolated data sync + training for exactly one lottery game."""
    product = product.strip().lower()
    if product not in vietlott.PRODUCTS:
        raise HTTPException(status_code=400, detail=f"Sản phẩm {product} không được hỗ trợ.")
    return {
        "status": "accepted",
        "product": product,
        "job": _queue_vietlott_training_job(product),
    }


@app.get("/api/control/vietlott/{product}/retrain-status")
async def control_vietlott_retrain_status(product: str, request: Request):
    product = product.strip().lower()
    if product not in vietlott.PRODUCTS:
        raise HTTPException(status_code=400, detail=f"Sản phẩm {product} không được hỗ trợ.")
    status = await asyncio.to_thread(_get_vietlott_training_status, product)
    return {
        "product": product,
        **status,
    }


@app.get("/api/app/version")
async def get_app_version():
    """Returns the latest app version information for update checking."""
    return {
        "version": "1.0.1",
        "build_number": 2,
        "download_url": "/static/app-release.apk",
        "change_log": "Cập nhật giao diện Widget 2x2, sửa lỗi tràn chữ, hỗ trợ mở ứng dụng trực tiếp từ widget, và tối ưu hóa hiệu suất mạng ngày cuối tuần."
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=3000, reload=False)
