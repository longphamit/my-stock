import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
import os

import crawler
import predictor
import vietlott

# Initialize MongoDB database
crawler.init_db()

app = FastAPI(title="Hệ thống Khuyến nghị Đầu tư Cổ phiếu Việt Nam")

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

class BatchRequest(BaseModel):
    tickers: List[str]

class SingleRequest(BaseModel):
    ticker: str

class LearnRequest(BaseModel):
    date: str
    model: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/gold", response_class=HTMLResponse)
async def gold_price_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/gold/prediction", response_class=HTMLResponse)
async def gold_prediction_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/stocks", response_class=HTMLResponse)
async def stocks_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/vietlott", response_class=HTMLResponse)
async def vietlott_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/vnall")
async def get_vnall():
    data = crawler.fetch_vnall_tickers()
    if not data:
        data = crawler.get_snapshots_from_db("VNALL")
    if not data:
        raise HTTPException(status_code=500, detail="Không thể tải danh sách cổ phiếu VNAllShare từ cả máy chủ và cơ sở dữ liệu.")
    return data

@app.get("/api/vn100")
async def get_vn100():
    data = crawler.fetch_vn100_tickers()
    if not data:
        data = crawler.get_snapshots_from_db("VN100")
    if not data:
        raise HTTPException(status_code=500, detail="Không thể tải danh sách cổ phiếu VN100 từ cả máy chủ và cơ sở dữ liệu.")
    return data

@app.get("/api/watchlist")
async def get_watchlist():
    return crawler.get_watchlist()

@app.post("/api/watchlist")
async def add_watchlist(req: SingleRequest):
    ticker = req.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Mã cổ phiếu không hợp lệ")
    success = crawler.add_to_watchlist(ticker)
    if not success:
        raise HTTPException(status_code=500, detail=f"Không thể lưu {ticker} vào cơ sở dữ liệu.")
    return {"status": "success", "ticker": ticker}

@app.delete("/api/watchlist/{ticker}")
async def delete_watchlist(ticker: str):
    ticker = ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Mã cổ phiếu không hợp lệ")
    success = crawler.remove_from_watchlist(ticker)
    if not success:
        raise HTTPException(status_code=500, detail=f"Không thể xóa {ticker} khỏi cơ sở dữ liệu.")
    return {"status": "success", "ticker": ticker}

def get_single_prediction_analysis(ticker):
    df, source = crawler.get_stock_data(ticker)
    if df is None or len(df) < 20:
        return {
            "status": "error",
            "message": f"Không thể lấy đủ dữ liệu lịch sử cho mã {ticker}. Vui lòng thử lại hoặc kiểm tra mã cổ phiếu."
        }
        
    from datetime import datetime
    import pandas as pd
    
    # 1. Determine if df already contains today's date
    today_str = datetime.now().strftime("%Y-%m-%d")
    latest_date_in_df = df["date"].max()
    
    df_old = None
    df_new = None
    
    if latest_date_in_df == today_str:
        # Today is already in history (e.g. after-hours)
        df_new = df.copy()
        df_old = df.iloc[:-1].copy()
    else:
        # Today is not in history (e.g. during trading hours)
        df_old = df.copy()
        
        # Try to find today's live snapshot
        try:
            from crawler import db as mongo_db
            snap = mongo_db.ticker_snapshots.find_one({"ticker": ticker})
            if snap:
                new_row = {
                    "date": today_str,
                    "open": float(snap.get("basic_price", snap.get("current_price", 0.0))),
                    "high": float(snap.get("highest", snap.get("current_price", 0.0))),
                    "low": float(snap.get("lowest", snap.get("current_price", 0.0))),
                    "close": float(snap["current_price"]),
                    "volume": float(snap["volume"]),
                    "change": float(snap.get("change", 0.0)),
                    "pct_change": float(snap.get("pct_change", 0.0))
                }
                df_new = pd.concat([df_old, pd.DataFrame([new_row])], ignore_index=True)
        except Exception as e:
            print(f"Error fetching snapshot for extending history: {e}")
            
        if df_new is None:
            # Fallback when there is no today's snapshot:
            # We treat the current session as the last historical session (df_old).
            # The previous session is df_old.iloc[:-1].
            df_new = df_old.copy()
            df_old = df_old.iloc[:-1].copy()
            
    # Calculate predictions
    # Main prediction defaults to df_new if available, else df_old
    if df_new is not None and len(df_new) >= 20:
        analysis = predictor.analyze_and_recommend(df_new)
        # Add old_analysis
        if len(df_old) >= 20:
            analysis["old_analysis"] = predictor.analyze_and_recommend(df_old)
        else:
            analysis["old_analysis"] = None
    else:
        analysis = predictor.analyze_and_recommend(df_old)
        analysis["old_analysis"] = None
        
    analysis["crawler_source"] = source
    analysis["ticker"] = ticker
    return analysis

@app.post("/api/predict/single")
async def predict_single(req: SingleRequest):
    ticker = req.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Mã cổ phiếu không hợp lệ")
    
    res = get_single_prediction_analysis(ticker)
    if res.get("status") == "error":
        return JSONResponse(status_code=400, content=res)
    return res

@app.post("/api/predict/batch")
async def predict_batch(req: BatchRequest):
    results = {}
    for ticker in req.tickers:
        ticker = ticker.strip().upper()
        if not ticker:
            continue
        try:
            res = get_single_prediction_analysis(ticker)
            results[ticker] = res
        except Exception as e:
            results[ticker] = {
                "status": "error",
                "ticker": ticker,
                "message": str(e)
            }
    return results

import asyncio

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
        
        # Fetch latest stock prices to update snapshots in DB
        print("Background crawler: Fetching latest VNALL & VN100 stock snapshots...")
        crawler.fetch_index_tickers("VNALL")
        crawler.fetch_index_tickers("VN100")
        print("Background crawler: Successfully updated stock snapshots.")
    except Exception as e:
        print(f"Error in background gold crawler worker thread: {e}")

from multiprocessing import Process

active_gold_proc = None
active_vietlott_proc = None

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

def run_vietlott_crawler_process():
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
        vietlott.sync_all()
        print("Background crawler: Successfully synchronized Vietlott statistics.")
    except Exception as e:
        print(f"Error in background Vietlott crawler process: {e}")

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


async def vietlott_crawler_background_loop():
    global active_vietlott_proc
    print("Background Vietlott crawler loop started...")
    # Wait 10 seconds after startup to not block the main startup flow
    await asyncio.sleep(10)
    while True:
        try:
            if active_vietlott_proc and active_vietlott_proc.is_alive():
                print("Background Vietlott crawler process is still running. Skipping.")
            else:
                active_vietlott_proc = Process(target=run_vietlott_crawler_process)
                active_vietlott_proc.start()
        except Exception as e:
            print(f"Error in background Vietlott crawler loop: {e}")
        # Sleep for 12 hours
        await asyncio.sleep(12 * 3600)


@app.on_event("startup")
async def startup_event():
    # Start the background tasks
    asyncio.create_task(gold_crawler_background_loop())
    asyncio.create_task(vietlott_crawler_background_loop())

@app.get("/api/gold/prediction-history")
async def get_gold_prediction_history():
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        history = await loop.run_in_executor(
            None,
            crawler.get_gold_prediction_history_with_explanations
        )
        return JSONResponse(content=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        await loop.run_in_executor(None, crawler.tune_and_save_gold_hyperparameters)
        
        # Reload tomorrow predictions history using the newly optimized hyperparameters
        await loop.run_in_executor(None, crawler.update_gold_predictions_history)
        
        params_doc = crawler.db.gold_model_hyperparameters.find_one({"type": "gold_params"})
        if params_doc:
            return {
                "status": "success",
                "updated_at": params_doc.get("updated_at"),
                "params": params_doc.get("params")
            }
        return {"status": "error", "message": "Failed to retrieve optimized parameters."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gold")
async def get_gold(model: str = "random_forest"):
    try:
        from datetime import datetime
        # Load from MongoDB latest cache
        cached = crawler.db.gold_prices.find_one({"_id": "latest_prices"})
        
        # If cache is missing or invalid (missing key prices), force a crawl
        if not cached or not cached.get("world", {}).get("price") or not cached.get("domestic", {}).get("sjc_bar", {}).get("buy"):
            print("Cached gold prices missing or invalid. Performing sync fetch...")
            cached = crawler.fetch_gold_prices(bypass_cache=True)
            
        if not cached or not cached.get("world", {}).get("price") or not cached.get("domestic", {}).get("sjc_bar", {}).get("buy"):
            raise HTTPException(status_code=500, detail="Không thể tải dữ liệu giá vàng.")
            
        data = cached.copy()
        data.pop("_id", None)
        data.pop("fetched_at", None)
        
        # Add predictions and intraday ticks
        try:
            predictions = crawler.get_gold_predictions()
            data["predictions"] = predictions
            data["latest_adjustments"] = predictions.get("latest_adjustments", {})
            data["latest_adjustment"] = predictions.get("latest_adjustments", {}).get(model)
        except Exception as pred_err:
            print(f"Error predicting gold: {pred_err}")
            data["predictions"] = None
            data["latest_adjustments"] = {}
            data["latest_adjustment"] = None

        try:
            data["ticks"] = crawler.get_gold_ticks()
        except Exception as ticks_err:
            print(f"Error fetching gold ticks: {ticks_err}")
            data["ticks"] = []
        
        response = JSONResponse(content=data)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/gold/refresh")
async def refresh_gold():
    """Force re-crawl latest gold prices + re-run all AI prediction models."""
    import asyncio
    try:
        print("Manual refresh triggered: force-crawling gold prices...")
        # Run in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        fresh = await loop.run_in_executor(
            None,
            lambda: crawler.fetch_gold_prices(bypass_cache=True)
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

        # Re-run prediction models with fresh data and fetch ticks
        try:
            predictions = crawler.get_gold_predictions()
            data["predictions"] = predictions
        except Exception as pred_err:
            print(f"Error re-predicting gold after refresh: {pred_err}")
            data["predictions"] = None

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




@app.get("/api/vietlott/{product}")
async def get_vietlott(product: str):
    product = product.strip().lower()
    try:
        data = vietlott.get_cached_vietlott_data(product)
        return JSONResponse(content=data)
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SimulateRequest(BaseModel):
    numbers: List[int]

@app.post("/api/vietlott/{product}/simulate")
async def simulate_vietlott(product: str, req: SimulateRequest):
    product = product.strip().lower()
    if not req.numbers or len(req.numbers) != 6:
        raise HTTPException(status_code=400, detail="Vui lòng chọn đúng 6 con số.")
    try:
        result = vietlott.backtest_user_numbers(product, req.numbers)
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("message"))
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/vietlott/refresh")
async def refresh_vietlott(product: Optional[str] = None):
    """Force synchronize Vietlott products using the local crawler."""
    import asyncio
    try:
        if product:
            product = product.strip().lower()
            print(f"Manual refresh triggered: force-syncing Vietlott product {product}...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, vietlott.sync_vietlott_data, product)
            return {"status": "success", "results": {product: "success"}}
        else:
            print("Manual refresh triggered: force-syncing all Vietlott products...")
            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(None, vietlott.sync_all)
            return {"status": "success", "results": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

