import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pymongo import MongoClient, UpdateOne

# Initialize MongoDB client
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client["stock_analytics"]

def init_db():
    """
    Initializes the MongoDB indexes and seeds default watchlist.
    """
    try:
        # Create unique indexes
        db.stock_history.create_index([("ticker", 1), ("date", 1)], unique=True)
        db.ticker_snapshots.create_index([("ticker", 1)], unique=True)
        db.watchlist.create_index([("ticker", 1)], unique=True)
        db.gold_history.create_index([("date", 1)], unique=True)
        db.us_economic_calendar.create_index([("date", 1), ("event", 1)], unique=True)
        db.macro_history.create_index([("date", 1)], unique=True)
        
        # Seed default watchlist if empty
        if db.watchlist.count_documents({}) == 0:
            now = datetime.now()
            default_tickers = ['TCB', 'FPT', 'VNM', 'HPG', 'SSI']
            docs = []
            for idx, ticker in enumerate(default_tickers):
                added_time = (now + timedelta(seconds=idx)).strftime("%Y-%m-%d %H:%M:%S")
                docs.append({"ticker": ticker, "added_at": added_time})
            db.watchlist.insert_many(docs)

        # Seed gold price history if empty or insufficient for predictions
        if db.gold_history.count_documents({}) < 100:
            seed_gold_history()

        # Seed US economic calendar if empty
        if db.us_economic_calendar.count_documents({}) == 0:
            seed_us_economic_calendar()

        # Seed macro history if empty
        if db.macro_history.count_documents({}) == 0:
            update_macro_history(days=365)
    except Exception as e:
        print(f"Error initializing MongoDB: {e}")


def get_last_trading_day():
    """
    Returns the YYYY-MM-DD of the most recent trading session.
    If today is a weekday and current local time is after 15:30, returns today.
    Otherwise, returns the previous weekday (taking into account weekends).
    """
    now = datetime.now()
    weekday = now.weekday() # 0 = Monday, 6 = Sunday
    
    # Check if we expect today's data to be finalized (after 15:30)
    is_after_market = now.hour > 15 or (now.hour == 15 and now.minute >= 30)
    
    if weekday == 5: # Saturday -> expected last is Friday
        target = now - timedelta(days=1)
    elif weekday == 6: # Sunday -> expected last is Friday
        target = now - timedelta(days=2)
    elif weekday == 0: # Monday
        if is_after_market:
            target = now
        else:
            target = now - timedelta(days=3) # Previous Friday
    else: # Tue, Wed, Thu, Fri
        if is_after_market:
            target = now
        else:
            target = now - timedelta(days=1) # Yesterday
            
    return target.strftime("%Y-%m-%d")

def save_history_to_db(ticker, df):
    """
    Saves a DataFrame of historical prices to MongoDB.
    """
    if df is None or df.empty:
        return
    try:
        operations = []
        for _, row in df.iterrows():
            doc = {
                "ticker": ticker.upper(),
                "date": row["date"],
                "open": float(row["open"]) if pd.notna(row["open"]) else 0.0,
                "high": float(row["high"]) if pd.notna(row["high"]) else 0.0,
                "low": float(row["low"]) if pd.notna(row["low"]) else 0.0,
                "close": float(row["close"]) if pd.notna(row["close"]) else 0.0,
                "volume": float(row["volume"]) if pd.notna(row["volume"]) else 0.0,
                "change": float(row["change"]) if "change" in row and pd.notna(row["change"]) else 0.0,
                "pct_change": float(row["pct_change"]) if "pct_change" in row and pd.notna(row["pct_change"]) else 0.0
            }
            operations.append(UpdateOne(
                {"ticker": doc["ticker"], "date": doc["date"]},
                {"$set": doc},
                upsert=True
            ))
        if operations:
            db.stock_history.bulk_write(operations)
    except Exception as e:
        print(f"Error saving history to DB for {ticker}: {e}")

def get_history_from_db(ticker):
    """
    Loads historical prices from MongoDB for a given ticker as a DataFrame.
    """
    ticker = ticker.upper()
    try:
        cursor = db.stock_history.find({"ticker": ticker}, {"_id": 0}).sort("date", 1)
        df = pd.DataFrame(list(cursor))
        if df.empty:
            return None
        # Return columns in specific order matching original schema
        cols = ["date", "open", "high", "low", "close", "volume", "change", "pct_change"]
        for col in cols:
            if col not in df.columns:
                df[col] = 0.0
        return df[cols]
    except Exception as e:
        print(f"Error reading history from DB for {ticker}: {e}")
        return None

def save_snapshots_to_db(stocks, index_code="VNALL"):
    """
    Saves a list of real-time price snapshots to MongoDB, tagging them with index_code.
    """
    if not stocks:
        return
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        operations = []
        for s in stocks:
            doc = {
                "ticker": s["ticker"].upper(),
                "basic_price": float(s["basic_price"]),
                "floor_price": float(s["floor_price"]),
                "ceiling_price": float(s["ceiling_price"]),
                "current_price": float(s["current_price"]),
                "change": float(s["change"]),
                "pct_change": float(s["pct_change"]),
                "highest": float(s["highest"]),
                "lowest": float(s["lowest"]),
                "volume": float(s["volume"]),
                "value": float(s["value"]),
                "last_updated": now_str
            }
            operations.append(UpdateOne(
                {"ticker": doc["ticker"]},
                {"$set": doc, "$addToSet": {"indices": index_code.upper()}},
                upsert=True
            ))
        if operations:
            db.ticker_snapshots.bulk_write(operations)
    except Exception as e:
        print(f"Error saving snapshots to DB: {e}")

def get_snapshots_from_db(index_code="VNALL"):
    """
    Loads snapshots for a specific index from MongoDB.
    """
    try:
        cursor = db.ticker_snapshots.find({"indices": index_code.upper()}, {"_id": 0})
        return list(cursor)
    except Exception as e:
        print(f"Error reading snapshots from DB: {e}")
        return []

def get_watchlist():
    """
    Retrieves the list of tickers in the watchlist.
    """
    try:
        cursor = db.watchlist.find({}, {"_id": 0, "ticker": 1}).sort("added_at", 1)
        return [doc["ticker"] for doc in cursor]
    except Exception as e:
        print(f"Error reading watchlist from DB: {e}")
        return []

def add_to_watchlist(ticker):
    """
    Adds a ticker to the watchlist in MongoDB.
    """
    try:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db.watchlist.update_one(
            {"ticker": ticker.upper()},
            {"$set": {"ticker": ticker.upper(), "added_at": now_str}},
            upsert=True
        )
        return True
    except Exception as e:
        print(f"Error adding to watchlist: {e}")
        return False

def remove_from_watchlist(ticker):
    """
    Removes a ticker from the watchlist in MongoDB.
    """
    try:
        result = db.watchlist.delete_one({"ticker": ticker.upper()})
        return result.deleted_count > 0
    except Exception as e:
        print(f"Error removing from watchlist: {e}")
        return False


def format_date(date_str, to_format):
    """
    Helper to convert date formats.
    Supported inputs: 'YYYY-MM-DD'
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime(to_format)
    except Exception:
        # Fallback to current time if parse fails
        return datetime.now().strftime(to_format)

def fetch_vndirect_data(ticker, start_date, end_date):
    """
    Fetches historical stock data from VNDirect's public REST API.
    NOTE: Deprecated/offline. Returns None immediately.
    """
    print(f"VNDirect historical API is offline. Skipping fetch for {ticker}.")
    return None


def parse_float(val, default=0.0):
    try:
        return float(val) if val else default
    except ValueError:
        return default

def fetch_index_tickers(index_code="VNALL"):
    """
    Fetches the real-time price snapshot for a specific index (VNALL, VN100, etc.) from VNDirect.
    Decodes the data using a character shift cipher: chr(ord(char) + idx % 5).
    """
    index_code = index_code.upper()
    url = f"https://price-streaming-api.vndirect.com.vn/v2/stocks/snapshot?indexCodes={index_code}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://banggia.vndirect.com.vn/"
    }
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return []
            
        encoded_list = r.json()
        stocks = []
        for item in encoded_list:
            decoded = "".join(chr(ord(char) + idx % 5) for idx, char in enumerate(item))
            parts = decoded.split("|")
            
            if len(parts) > 54:  # Ensure we have enough elements to read
                ticker = parts[1]
                r_parts = parts[1:]
                
                # Extract fields safely using helper
                basic_price = parse_float(r_parts[3]) * 1000
                floor_price = parse_float(r_parts[4]) * 1000
                ceiling_price = parse_float(r_parts[5]) * 1000
                
                current_val = parse_float(r_parts[52]) * 1000
                current_price = current_val if current_val > 0 else basic_price
                
                change = current_price - basic_price
                pct_change = (change / basic_price * 100) if basic_price > 0 else 0.0
                
                highest_val = parse_float(r_parts[48]) * 1000
                highest = highest_val if highest_val > 0 else current_price
                
                lowest_val = parse_float(r_parts[49]) * 1000
                lowest = lowest_val if lowest_val > 0 else current_price
                
                volume = parse_float(r_parts[51])
                # Value is in million VND, multiply by 1M to get VND
                value = parse_float(r_parts[50]) * 1000000
                
                stocks.append({
                    "ticker": ticker,
                    "basic_price": basic_price,
                    "floor_price": floor_price,
                    "ceiling_price": ceiling_price,
                    "current_price": current_price,
                    "change": change,
                    "pct_change": pct_change,
                    "highest": highest,
                    "lowest": lowest,
                    "volume": volume,
                    "value": value
                })
        if stocks:
            save_snapshots_to_db(stocks, index_code)
        return stocks
    except Exception as e:
        print(f"Error fetching {index_code} snapshot: {e}")
        cached = get_snapshots_from_db(index_code)
        if cached:
            print(f"Network error. Fallback to {len(cached)} cached snapshots from DB for {index_code}.")
            return cached
        return []

def fetch_vnall_tickers():
    return fetch_index_tickers("VNALL")

def fetch_vn100_tickers():
    return fetch_index_tickers("VN100")

def fetch_cafef_data(ticker, start_date, end_date, max_pages=6):
    """
    Fetches historical stock data from CafeF's redirect endpoint in pages of 20.
    Gathers enough pages to satisfy the start_date range (up to 200 trading days).
    """
    # Convert 'YYYY-MM-DD' to 'dd/mm/yyyy'
    try:
        sd = datetime.strptime(start_date, "%Y-%m-%d").strftime("%d/%m/%Y")
        ed = datetime.strptime(end_date, "%Y-%m-%d").strftime("%d/%m/%Y")
    except Exception:
        sd = (datetime.now() - timedelta(days=365)).strftime("%d/%m/%Y")
        ed = datetime.now().strftime("%d/%m/%Y")
        
    url = "https://cafef.vn/du-lieu/ajax/pagenew/datahistory/pricehistory.ashx"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": f"https://cafef.vn/du-lieu/lich-su-giao-dich-{ticker.lower()}-1.chn"
    }
    
    all_data = []
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def fetch_page(p):
        params = {
            "Symbol": ticker.upper(),
            "StartDate": sd,
            "EndDate": ed,
            "PageIndex": p,
            "PageSize": 20
        }
        try:
            r = requests.get(url, params=params, headers=headers, timeout=10)
            if r.status_code == 200:
                res_data = r.json()
                if res_data.get("Success") and res_data.get("Data"):
                    return res_data["Data"].get("Data", [])
        except Exception as e:
            print(f"Error fetching page {p} from CafeF: {e}")
        return []

    # Run fetching in parallel threads
    results_by_page = {}
    with ThreadPoolExecutor(max_workers=max_pages) as executor:
        futures = {executor.submit(fetch_page, p): p for p in range(1, max_pages + 1)}
        for future in as_completed(futures):
            p = futures[future]
            try:
                rows = future.result()
                if rows:
                    results_by_page[p] = rows
            except Exception as e:
                print(f"Page {p} thread error: {e}")
                
    # Merge results in order of pages (1 to max_pages)
    for p in sorted(results_by_page.keys()):
        rows = results_by_page[p]
        all_data.extend(rows)
        if len(rows) < 20:
            break
            
    if not all_data:
        return None
        
    try:
        parsed = []
        for item in all_data:
            # Format date: 'dd/mm/yyyy' -> 'YYYY-MM-DD'
            dt = datetime.strptime(item["Ngay"], "%d/%m/%Y")
            date_str = dt.strftime("%Y-%m-%d")
            
            # Parse change & pct_change
            thay_doi = item["ThayDoi"].replace(",", ".")
            parts = thay_doi.split()
            try:
                change_val = float(parts[0]) * 1000
            except:
                change_val = 0.0
            try:
                pct_change_val = float(parts[1].strip("()%"))
            except:
                pct_change_val = 0.0
                
            parsed.append({
                "date": date_str,
                "open": float(item["GiaMoCua"]) * 1000,
                "high": float(item["GiaCaoNhat"]) * 1000,
                "low": float(item["GiaThapNhat"]) * 1000,
                "close": float(item["GiaDongCua"]) * 1000,
                "volume": float(item["KhoiLuongKhopLenh"]),
                "change": change_val,
                "pct_change": pct_change_val
            })
            
        df = pd.DataFrame(parsed)
        # Sort ascending by date
        df = df.sort_values(by="date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error parsing CafeF data: {e}")
        return None

def update_cache_bg(ticker, start_date, end_date, max_pages):
    try:
        print(f"Background update starting for {ticker}...")
        df_new = fetch_cafef_data(ticker, start_date, end_date, max_pages=max_pages)
        if df_new is not None and not df_new.empty:
            save_history_to_db(ticker, df_new)
            print(f"Background update successful for {ticker}.")
        else:
            print(f"Background update failed/empty for {ticker}.")
    except Exception as e:
        print(f"Error in background update for {ticker}: {e}")

def get_stock_data(ticker, start_date=None, end_date=None):
    """
    Main entry point for stock daily history data crawling.
    Checks MongoDB first. If data exists and is up-to-date, returns from DB.
    Otherwise, fetches from online sources, saves/merges to DB, and returns.
    """
    init_db() # Ensure tables exist
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
        
    ticker = ticker.strip().upper()
    
    # 1. Check local DB cache
    df_cached = get_history_from_db(ticker)
    
    # Check if cache is valid (not stale and has enough data)
    is_cache_valid = False
    if df_cached is not None and len(df_cached) >= 20:
        latest_date = df_cached["date"].max()
        last_trading = get_last_trading_day()
        if latest_date >= last_trading:
            is_cache_valid = True
            
    if is_cache_valid:
        print(f"Cache hit for {ticker}. Latest date in DB: {latest_date}. Returning from DB.")
        df_filtered = df_cached[(df_cached["date"] >= start_date) & (df_cached["date"] <= end_date)].reset_index(drop=True)
        if len(df_filtered) >= 20:
            return df_filtered, "Database (Cached)"
        else:
            print(f"Filtered cache has only {len(df_filtered)} rows, which is less than 20. Fetching online instead.")

    # SWR (Stale-While-Revalidate): If cache exists but is stale, return stale cache immediately and update in background.
    if df_cached is not None and len(df_cached) >= 20:
        print(f"Cache stale for {ticker}. Returning stale cache and spawning background update thread...")
        import threading
        t = threading.Thread(target=update_cache_bg, args=(ticker, start_date, end_date, 1))
        t.daemon = True
        t.start()
        
        df_filtered = df_cached[(df_cached["date"] >= start_date) & (df_cached["date"] <= end_date)].reset_index(drop=True)
        return df_filtered, "Database (Stale-While-Revalidate)"


    # 2. Cache is missing or stale. Fetch from online.
    print(f"Cache miss/stale for {ticker}. Fetching online...")
    df_new = None
    source = "None"
    
    # Optimize: If we already have history, we only need to fetch the first page (20 records) to get up to date.
    max_pages = 6
    if df_cached is not None and len(df_cached) >= 20:
        max_pages = 1
        
    # Try CafeF
    print(f"Fetching from CafeF for ticker {ticker} (max_pages={max_pages})...")
    df_new = fetch_cafef_data(ticker, start_date, end_date, max_pages=max_pages)
    if df_new is not None:
        source = "CafeF (Synced to DB)"
        print(f"Successfully fetched from CafeF. Row count: {len(df_new)}")
    else:
        print("CafeF fetch failed. Skip VNDirect fallback (offline).")

            
    if df_new is not None and not df_new.empty:
        # Save new records to database
        save_history_to_db(ticker, df_new)
        # Reload combined data from DB to ensure complete dataset (e.g. merge past history with new rows)
        df_combined = get_history_from_db(ticker)
        if df_combined is not None:
            df_filtered = df_combined[(df_combined["date"] >= start_date) & (df_combined["date"] <= end_date)].reset_index(drop=True)
            return df_filtered, source
            
    # 3. Online fetch failed. Fallback to stale database cache if available
    if df_cached is not None and len(df_cached) >= 20:
        print(f"Online fetch failed for {ticker}. Falling back to stale DB cache.")
        df_filtered = df_cached[(df_cached["date"] >= start_date) & (df_cached["date"] <= end_date)].reset_index(drop=True)
        return df_filtered, source
        
    return None, "None"

def fetch_gold_prices(bypass_cache=False):
    """
    Fetches gold prices from giavang.org (world and domestic) and caches in MongoDB.
    Cache duration is 15 minutes.
    """
    # 1. Check cache first
    now = datetime.now()
    if not bypass_cache:
        try:
            cached = db.gold_prices.find_one({"_id": "latest_prices"})
            if cached:
                # Check if cache is still valid (< 15 mins)
                last_fetched = cached.get("fetched_at")
                if last_fetched and (now - last_fetched) < timedelta(minutes=2):
                    print("Gold prices cache hit. Returning cached data.")
                    data = cached.copy()
                    data.pop("fetched_at", None)
                    data.pop("_id", None)
                    return data
        except Exception as e:
            print(f"Error checking gold cache: {e}")


    # 2. Fetch new data
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    world_price = ""
    world_change = ""
    world_change_pct = ""
    world_time = ""
    converted_ounce = ""
    converted_cay = ""
    
    sjc_bar = {"buy": "", "sell": ""}
    sjc_ring = {"buy": "", "sell": ""}
    dom_time = ""
    brand_prices = []

    try:
        from bs4 import BeautifulSoup
        import re

        # Fetch world gold prices
        r_world = requests.get("https://giavang.org/the-gioi/", headers=headers, timeout=10)
        if r_world.status_code == 200:
            soup_world = BeautifulSoup(r_world.text, 'html.parser')
            
            price_box = soup_world.find(class_="crypto-price-box")
            if price_box:
                price_el = price_box.find(class_="crypto-price")
                if price_el:
                    world_price = price_el.text.strip()
                    
                change_el = price_box.find(class_="crypto-change")
                if change_el:
                    change_text = change_el.text.strip()
                    match = re.search(r"([\d.,-]+)[^\(]*\(([\d.,%+-]+)\)", change_text)
                    if match:
                        world_change = match.group(1).strip()
                        world_change_pct = match.group(2).strip()
                    else:
                        world_change = change_text
                        
                time_el = price_box.find(class_="box-headline")
                if time_el and time_el.find("small"):
                    world_time = time_el.find("small").text.replace("Cập nhật lúc", "").strip()

            content_box = soup_world.find(class_="box-content")
            if content_box:
                text = content_box.text
                match_ounce = re.search(r"1\s+Ounce\s*=\s*([\d\.]+)\s*VNĐ", text, re.IGNORECASE)
                if match_ounce:
                    converted_ounce = match_ounce.group(1).strip()
                match_cay = re.search(r"1\s+cây\s+vàng\s+[^.]+có\s+giá\s+là\s*([\d\.]+)\s*VNĐ", text, re.IGNORECASE)
                if match_cay:
                    converted_cay = match_cay.group(1).strip()
    except Exception as e:
        print(f"Error fetching world gold prices: {e}")

    try:
        from bs4 import BeautifulSoup
        import re

        # Fetch domestic gold prices
        r_dom = requests.get("https://giavang.org/trong-nuoc/", headers=headers, timeout=10)
        if r_dom.status_code == 200:
            soup_dom = BeautifulSoup(r_dom.text, 'html.parser')
            
            headline = soup_dom.find(class_="box-headline highlight")
            if headline and headline.find("small"):
                dom_time = headline.find("small").text.replace("Cập nhật lúc", "").strip()
                
            price_boxes = soup_dom.find_all(class_="gold-price-box")
            for box in price_boxes:
                headers_h2 = box.find_all("h2")
                for h2 in headers_h2:
                    title = h2.text.strip()
                    if "Miếng SJC" in title:
                        buy_el = box.find(class_="box-cgre")
                        sell_el = box.find(class_="box-cred")
                        if buy_el and buy_el.find(class_="gold-price"):
                            sjc_bar["buy"] = buy_el.find(class_="gold-price").text.split()[0].strip()
                        if sell_el and sell_el.find(class_="gold-price"):
                            sjc_bar["sell"] = sell_el.find(class_="gold-price").text.split()[0].strip()
                    elif "Nhẫn SJC" in title:
                        buy_el = box.find(class_="box-cgre")
                        sell_el = box.find(class_="box-cred")
                        if buy_el and buy_el.find(class_="gold-price"):
                            sjc_ring["buy"] = buy_el.find(class_="gold-price").text.split()[0].strip()
                        if sell_el and sell_el.find(class_="gold-price"):
                            sjc_ring["sell"] = sell_el.find(class_="gold-price").text.split()[0].strip()

            table = soup_dom.find("table", class_="table")
            if table:
                tbody = table.find("tbody")
                if tbody:
                    current_region = ""
                    rows = tbody.find_all("tr")
                    for row in rows:
                        th_list = row.find_all("th")
                        td_list = row.find_all("td")
                        if not td_list:
                            continue
                        if th_list:
                            current_region = th_list[0].text.strip()
                        brand_td = td_list[0]
                        brand_name = brand_td.text.strip()
                        if len(td_list) >= 3:
                            buy_price = td_list[1].text.strip()
                            sell_price = td_list[2].text.strip()
                            if current_region in ["TP. Hồ Chí Minh", "Hà Nội"]:
                                brand_prices.append({
                                    "region": current_region,
                                    "brand": brand_name,
                                    "buy": buy_price,
                                    "sell": sell_price
                                })
    except Exception as e:
        print(f"Error fetching domestic gold prices: {e}")

    # Fetch news, economic calendar, macro indicators, crude oil, and geopolitical conflicts
    news = fetch_macro_news()
    calendar = get_us_economic_calendar()
    macro_indicators = fetch_us_macro_indicators()
    crude_oil = fetch_crude_oil_price()
    conflict_events = fetch_geopolitical_conflicts()

    # Prepare data dict
    gold_data = {
        "world": {
            "price": world_price,
            "change": world_change,
            "change_pct": world_change_pct,
            "converted_ounce": converted_ounce,
            "converted_cay": converted_cay,
            "time": world_time
        },
        "domestic": {
            "sjc_bar": sjc_bar,
            "sjc_ring": sjc_ring,
            "time": dom_time,
            "brands": brand_prices
        },
        "news": news,
        "calendar": calendar,
        "macro_indicators": macro_indicators,
        "crude_oil": crude_oil,
        "conflict_events": conflict_events
    }

    # Save cache if we fetched valid data
    if world_price or sjc_bar["buy"]:
        try:
            # 1. Update gold prices cache
            cache_doc = gold_data.copy()
            cache_doc["_id"] = "latest_prices"
            cache_doc["fetched_at"] = now
            db.gold_prices.replace_one({"_id": "latest_prices"}, cache_doc, upsert=True)
            print("Successfully cached gold prices to MongoDB.")

            # 2. Save today's prices to gold_history daily table
            today_str = now.strftime("%Y-%m-%d")
            today_doc = {
                "date": today_str,
                "sjc_bar_buy": clean_vnd_price(sjc_bar["buy"]),
                "sjc_bar_sell": clean_vnd_price(sjc_bar["sell"]),
                "sjc_ring_buy": clean_vnd_price(sjc_ring["buy"]),
                "sjc_ring_sell": clean_vnd_price(sjc_ring["sell"]),
                "world_price": float(world_price.replace(",", "")) if world_price else 0.0,
                "world_price_vnd": clean_vnd_price(converted_cay)
            }
            db.gold_history.update_one({"date": today_str}, {"$set": today_doc}, upsert=True)
            print(f"Successfully saved gold history point for {today_str}.")
            
            try:
                update_gold_predictions_history()
            except Exception as hist_err:
                print(f"Error updating gold prediction history in fetch: {hist_err}")

            # 3. Save to intraday ticks table to capture fluctuations during the day
            try:
                sjc_buy_val = clean_vnd_price(sjc_bar["buy"])
                sjc_sell_val = clean_vnd_price(sjc_bar["sell"])
                sjc_ring_buy_val = clean_vnd_price(sjc_ring["buy"])
                sjc_ring_sell_val = clean_vnd_price(sjc_ring["sell"])
                world_price_val = float(world_price.replace(",", "")) if world_price else 0.0
                world_price_vnd_val = clean_vnd_price(converted_cay)
                
                # Check the last tick to avoid duplicates
                last_tick = db.gold_ticks.find_one(sort=[("timestamp", -1)])
                if not last_tick or \
                   last_tick.get("sjc_bar_buy") != sjc_buy_val or \
                   last_tick.get("sjc_bar_sell") != sjc_sell_val or \
                   last_tick.get("sjc_ring_buy") != sjc_ring_buy_val or \
                   last_tick.get("sjc_ring_sell") != sjc_ring_sell_val or \
                   last_tick.get("world_price") != world_price_val or \
                   last_tick.get("world_price_vnd") != world_price_vnd_val:
                    
                    tick_doc = {
                        "timestamp": now,
                        "sjc_bar_buy": sjc_buy_val,
                        "sjc_bar_sell": sjc_sell_val,
                        "sjc_ring_buy": sjc_ring_buy_val,
                        "sjc_ring_sell": sjc_ring_sell_val,
                        "world_price": world_price_val,
                        "world_price_vnd": world_price_vnd_val
                    }
                    db.gold_ticks.insert_one(tick_doc)
                    print(f"Successfully saved gold tick at {now}.")
            except Exception as tick_err:
                print(f"Error saving gold tick: {tick_err}")
        except Exception as e:
            print(f"Error writing gold cache/history: {e}")
    else:
        # If fetch failed, return stale cache if available
        try:
            cached = db.gold_prices.find_one({"_id": "latest_prices"})
            if cached:
                print("Fetch failed. Returning stale cached gold prices.")
                data = cached.copy()
                data.pop("fetched_at", None)
                data.pop("_id", None)
                return data
        except Exception as e:
            print(f"Error getting stale gold cache: {e}")

    return gold_data

def clean_vnd_price(val):
    if not val:
        return 0.0
    try:
        # e.g., "145.000" or "133.548.313"
        cleaned = val.replace(".", "").replace(",", "").strip()
        num = float(cleaned)
        if num < 1000000: # e.g. 145000 -> convert to 145000000
            num *= 1000
        return num
    except:
        return 0.0

def seed_gold_history():
    """
    Seeds the gold_history collection in MongoDB.
    First attempts to fetch from the vang.today API to get up to 200 days of history.
    If that fails, falls back to parsing the Highcharts JS chart data from the domestic gold price page.
    """
    try:
        print("Seeding gold price history...")
        # 1. Attempt vang.today API
        try:
            print("Attempting to seed via vang.today API...")
            r = requests.get("https://vang.today/api/prices?type=SJL1L10&days=200", headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if r.status_code == 200:
                data = r.json()
                history = data.get("history", [])
                if len(history) >= 20:
                    operations = []
                    for item in history:
                        date_str = item.get("date")
                        prices_info = item.get("prices", {}).get("SJL1L10", {})
                        buy_val = float(prices_info.get("buy", 0.0))
                        sell_val = float(prices_info.get("sell", 0.0))
                        if buy_val > 0 and sell_val > 0:
                            world_vnd_val = round(sell_val * 0.8926, -3)
                            world_usd_val = round(world_vnd_val / 31887.08, 2)
                            doc = {
                                "date": date_str,
                                "sjc_bar_buy": buy_val,
                                "sjc_bar_sell": sell_val,
                                "sjc_ring_buy": buy_val,
                                "sjc_ring_sell": sell_val,
                                "world_price": world_usd_val,
                                "world_price_vnd": world_vnd_val
                            }
                            operations.append(UpdateOne(
                                {"date": date_str},
                                {"$set": doc},
                                upsert=True
                            ))
                    if operations:
                        db.gold_history.bulk_write(operations)
                        print(f"Successfully seeded {len(operations)} gold history records from vang.today API.")
                        return
        except Exception as api_err:
            print(f"vang.today API seeding failed: {api_err}. Falling back to domestic gold page scraping...")

        # 2. Fallback to domestic gold page scraping
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        r = requests.get("https://giavang.org/trong-nuoc/", headers=headers, timeout=10)
        if r.status_code != 200:
            print("Failed to fetch domestic gold page for seeding.")
            return
            
        from bs4 import BeautifulSoup
        import re
        soup = BeautifulSoup(r.text, 'html.parser')
        scripts = soup.find_all("script")
        chart_script = ""
        for s in scripts:
            if s.string and "seriesOptions" in s.string and "Highcharts.stockChart" in s.string:
                chart_script = s.string
                break
                
        if chart_script:
            buy_match = re.search(r'name\s*:\s*["\']Mua vào["\']\s*,\s*data\s*:\s*(\[\[.*?\]\])', chart_script)
            sell_match = re.search(r'name\s*:\s*["\']Bán ra["\']\s*,\s*data\s*:\s*(\[\[.*?\]\])', chart_script)
            
            buy_dict = {}
            sell_dict = {}
            
            if buy_match:
                pairs = re.findall(r'\[(\d+)\s*,\s*([\d\.]+)\]', buy_match.group(1))
                for t, p in pairs:
                    dt = datetime.fromtimestamp(int(t) / 1000)
                    buy_dict[dt.strftime("%Y-%m-%d")] = float(p) * 1000000
                    
            if sell_match:
                pairs = re.findall(r'\[(\d+)\s*,\s*([\d\.]+)\]', sell_match.group(1))
                for t, p in pairs:
                    dt = datetime.fromtimestamp(int(t) / 1000)
                    sell_dict[dt.strftime("%Y-%m-%d")] = float(p) * 1000000
                    
            # Combine
            all_dates = sorted(list(set(buy_dict.keys()) | set(sell_dict.keys())))
            operations = []
            for date_str in all_dates:
                sell_val = sell_dict.get(date_str, 0.0)
                world_vnd_val = round(sell_val * 0.8926, -3) if sell_val > 0 else 0.0
                world_usd_val = round(world_vnd_val / 31887.08, 2) if world_vnd_val > 0 else 0.0
                doc = {
                    "date": date_str,
                    "sjc_bar_buy": buy_dict.get(date_str, 0.0),
                    "sjc_bar_sell": sell_val,
                    "sjc_ring_buy": buy_dict.get(date_str, 0.0),
                    "sjc_ring_sell": sell_val,
                    "world_price": world_usd_val,
                    "world_price_vnd": world_vnd_val
                }
                operations.append(UpdateOne(
                    {"date": date_str},
                    {"$set": doc},
                    upsert=True
                ))
                
            if operations:
                db.gold_history.bulk_write(operations)
                print(f"Successfully seeded {len(operations)} gold history records to MongoDB.")
    except Exception as e:
        print(f"Error seeding gold history: {e}")

def update_macro_history(days=365):
    """
    Downloads historical Close prices for DXY, US10Y, VIX, Brent Crude, DJI,
    EUR/USD, and Silver (XAG/USD) from Yahoo Finance using yfinance,
    and saves them to MongoDB.
    """
    try:
        import yfinance as yf
        print(f"Updating macro history for the past {days} days...")
        
        tickers = {
            "dxy":    "DX-Y.NYB",   # Dollar Index (inverse gold)
            "us10y":  "^TNX",       # US 10Y Treasury yield
            "vix":    "^VIX",       # Fear index
            "brent":  "BZ=F",       # Brent Crude Oil
            "dji":    "^DJI",       # Dow Jones Industrial
            "spx":    "^GSPC",      # S&P 500 Index
            "eurusd": "EURUSD=X",   # EUR/USD — dollar hedge signal
            "xagusd": "SI=F",       # Silver futures — gold-silver ratio signal
            "gld":    "GLDM",       # SPDR Gold MiniShares Trust
            "gld_trust": "GLD",     # SPDR Gold Trust
        }
        
        # Determine period
        if days <= 30:
            period_str = "1mo"
        elif days <= 90:
            period_str = "3mo"
        elif days <= 180:
            period_str = "6mo"
        else:
            period_str = "1y"
            
        df = yf.download(list(tickers.values()), period=period_str, progress=False)
        if df.empty or "Close" not in df:
            print("Failed to download macro data from Yahoo Finance.")
            return
            
        close_prices = df["Close"]
        # Fill missing values
        close_prices = close_prices.ffill().bfill()
        
        operations = []
        for index, row in close_prices.iterrows():
            date_str = index.strftime("%Y-%m-%d")
            
            def _safe(key):
                v = row.get(key)
                return float(v) if v is not None and pd.notna(v) else None

            doc = {
                "date":    date_str,
                "dxy":     _safe("DX-Y.NYB"),
                "us10y":   _safe("^TNX"),
                "vix":     _safe("^VIX"),
                "brent":   _safe("BZ=F"),
                "dji":     _safe("^DJI"),
                "spx":     _safe("^GSPC"),
                "eurusd":  _safe("EURUSD=X"),
                "xagusd":  _safe("SI=F"),
                "gld":     _safe("GLDM"),
                "gld_trust": _safe("GLD"),
            }
            operations.append(UpdateOne(
                {"date": date_str},
                {"$set": doc},
                upsert=True
            ))
            
        if operations:
            db.macro_history.bulk_write(operations)
            print(f"Successfully saved {len(operations)} macro history records to MongoDB.")
    except Exception as e:
        print(f"Error updating macro history: {e}")


def update_real_yield_from_fred(days=400):
    """
    Fetches 10-Year Breakeven Inflation Rate (T10YIE) from the free FRED API
    and computes real_yield = us10y - t10yie.

    Real yield is the #1 predictor of gold prices (correlation ~-0.85):
      - Real yield falls  → gold rises  (holding cash costs more in real terms)
      - Real yield rises  → gold falls  (risk-free real return becomes attractive)

    Series used:
      T10YIE — 10-Year Breakeven Inflation Rate (FRED, daily, free)
    """
    try:
        import urllib.request
        print("Updating real yield (FRED T10YIE)...")

        # Download T10YIE CSV from FRED (no API key needed)
        fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10YIE"
        req = urllib.request.Request(fred_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8")

        # Parse CSV
        lines = raw.strip().split("\n")[1:]  # skip header
        fred_data = {}
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) == 2 and parts[1] not in (".", ""):
                try:
                    fred_data[parts[0]] = float(parts[1])
                except ValueError:
                    pass

        if not fred_data:
            print("FRED returned empty data.")
            return

        # Merge with existing macro_history and compute real_yield
        operations = []
        for date_str, t10yie in fred_data.items():
            existing = db.macro_history.find_one({"date": date_str}, {"_id": 0, "us10y": 1})
            us10y = existing.get("us10y") if existing else None

            real_yield = round(us10y - t10yie, 4) if us10y is not None else None

            operations.append(UpdateOne(
                {"date": date_str},
                {"$set": {"t10yie": t10yie, "real_yield": real_yield}},
                upsert=False  # only update existing macro records
            ))

        if operations:
            result = db.macro_history.bulk_write(operations)
            print(f"FRED: Updated real_yield for {result.modified_count} records "
                  f"(t10yie range: {min(fred_data.values()):.2f}–{max(fred_data.values()):.2f}%)")
    except Exception as e:
        print(f"Error updating real yield from FRED: {e}")


def update_all_macro(days=365):
    """Convenience wrapper: update Yahoo Finance macro + FRED real yield."""
    update_macro_history(days=days)
    update_real_yield_from_fred(days=days)


def calculate_days_until_events(dates_series):
    """
    Given a pandas Series of dates (in YYYY-MM-DD string format),
    queries the us_economic_calendar collection in MongoDB and returns
    three pandas Series of the same length containing the number of days
    until the next FED meeting, CPI release, and NFP release.
    """
    # 1. Fetch all events from DB
    try:
        events = list(db.us_economic_calendar.find({}, {"_id": 0}))
    except Exception as e:
        print(f"Error querying economic calendar: {e}")
        events = []
        
    # Group event dates by category
    event_dates = {
        "FED": [],
        "CPI": [],
        "Việc làm": []  # Non-Farm Payrolls category is 'Việc làm'
    }
    
    for ev in events:
        category = ev.get("category")
        date_str = ev.get("date")
        if category in event_dates and date_str:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").date()
                event_dates[category].append(dt)
            except ValueError:
                pass
                
    # Sort dates ascending
    for cat in event_dates:
        event_dates[cat].sort()
        
    # Helper to calculate days until next event
    def get_days_until_next(current_date_str, category):
        try:
            curr_date = datetime.strptime(current_date_str, "%Y-%m-%d").date()
        except ValueError:
            return 30.0 # Default fallback
            
        target_list = event_dates.get(category, [])
        # Find the first event date that is >= curr_date
        next_event = None
        for ev_dt in target_list:
            if ev_dt >= curr_date:
                next_event = ev_dt
                break
                
        if next_event is None:
            return 30.0 # Fallback if no future event in calendar
            
        return float((next_event - curr_date).days)
        
    # Apply to all dates in dates_series
    days_to_fed = [get_days_until_next(d, "FED") for d in dates_series]
    days_to_cpi = [get_days_until_next(d, "CPI") for d in dates_series]
    days_to_nfp = [get_days_until_next(d, "Việc làm") for d in dates_series]
    
    return days_to_fed, days_to_cpi, days_to_nfp

def calculate_dynamic_ensemble_weights(limit=20):
    """
    Computes model weights for Random Forest, Linear Regression, MLP, and XGBoost
    based on a COMBINED score of:
      - 50%: Historical MAPE (Mean Absolute Percentage Error) — accuracy of magnitude
      - 50%: Directional accuracy (% of times correct direction was predicted)
    Returns:
        dict: {model_key: weight} summing to 1.0.
    """
    try:
        cursor = db.gold_predictions_history.find(
            {"actual_price": {"$ne": None, "$gt": 0.0}},
            {"_id": 0, "date": 1, "actual_price": 1, "models": 1}
        ).sort("date", -1).limit(limit)
        
        history = list(cursor)
        if len(history) < 5:
            return {
                "random_forest": 0.20,
                "linear_regression": 0.20,
                "mlp": 0.20,
                "xgboost": 0.20,
                "lstm": 0.20
            }

        # Sort ascending for direction comparison
        history_asc = list(reversed(history))
            
        model_errors = {
            "random_forest": [],
            "linear_regression": [],
            "mlp": [],
            "xgboost": [],
            "lstm": []
        }
        model_dir_correct = {k: [] for k in model_errors}
        
        for i, doc in enumerate(history_asc):
            actual = doc.get("actual_price")
            models_preds = doc.get("models", {})

            # Previous actual price for direction comparison
            prev_actual = history_asc[i - 1].get("actual_price") if i > 0 else None

            for m_key in model_errors.keys():
                pred = models_preds.get(m_key)
                if pred is not None and pred > 0 and actual:
                    # MAPE component
                    ape = abs(actual - pred) / actual
                    model_errors[m_key].append(ape)

                    # Directional accuracy component
                    if prev_actual and prev_actual > 0:
                        actual_dir = 1 if actual > prev_actual else -1
                        pred_dir = 1 if pred > prev_actual else -1
                        model_dir_correct[m_key].append(1 if actual_dir == pred_dir else 0)
                    
        # Calculate MAPE scores
        mapes = {}
        dir_accs = {}
        for m_key, errors in model_errors.items():
            mapes[m_key] = sum(errors) / len(errors) if len(errors) >= 3 else 0.05
            dir_accs[m_key] = sum(model_dir_correct[m_key]) / len(model_dir_correct[m_key]) \
                               if len(model_dir_correct[m_key]) >= 3 else 0.5

        # Normalize MAPE into score (lower MAPE = higher score)
        epsilon = 1e-4
        mape_scores = {k: 1.0 / (v + epsilon) for k, v in mapes.items()}
        total_mape = sum(mape_scores.values())
        mape_weights = {k: v / total_mape for k, v in mape_scores.items()} if total_mape > 0 \
                       else {k: 0.25 for k in mape_scores}

        # Normalize directional accuracy into weights
        total_dir = sum(dir_accs.values())
        dir_weights = {k: v / total_dir for k, v in dir_accs.items()} if total_dir > 0 \
                      else {k: 0.25 for k in dir_accs}

        # Combine 50% MAPE + 50% directional accuracy
        combined = {k: 0.5 * mape_weights[k] + 0.5 * dir_weights[k] for k in mape_weights}
        total_combined = sum(combined.values())
        weights = {k: float(v / total_combined) for k, v in combined.items()} if total_combined > 0 \
                  else {k: 0.25 for k in combined}

        print(f"Ensemble weights (MAPE+Dir): {' | '.join(f'{k}={v:.3f}' for k,v in weights.items())}")
        print(f"  Directional accuracy: {' | '.join(f'{k}={v:.1%}' for k,v in dir_accs.items())}")
        return weights
    except Exception as e:
        print(f"Error calculating dynamic ensemble weights: {e}")
        return {
            "random_forest": 0.20,
            "linear_regression": 0.20,
            "mlp": 0.20,
            "xgboost": 0.20,
            "lstm": 0.20
        }


def calculate_gold_model_biases(limit=15):
    """
    Calculates the rolling bias (mean error = actual - pred) of the models
    over the last `limit` days to apply online feedback error correction (self-learning).
    """
    try:
        cursor = db.gold_predictions_history.find(
            {"actual_price": {"$ne": None, "$gt": 0.0}},
            {"_id": 0, "actual_price": 1, "models": 1}
        ).sort("date", -1).limit(limit)
        
        history = list(cursor)
        if len(history) < 3:
            return {
                "random_forest": 0.0,
                "linear_regression": 0.0,
                "mlp": 0.0,
                "xgboost": 0.0,
                "lstm": 0.0
            }
            
        biases = {}
        model_keys = ["random_forest", "linear_regression", "mlp", "xgboost", "lstm"]
        for m_key in model_keys:
            errors = []
            for doc in history:
                actual = doc.get("actual_price")
                pred = doc.get("models", {}).get(m_key)
                if pred is not None and pred > 0 and actual:
                    # bias = actual - pred
                    errors.append(actual - pred)
            biases[m_key] = float(np.mean(errors)) if len(errors) >= 3 else 0.0
            
        print(f"Self-Learning rolling biases: {' | '.join(f'{k}={v:.2f}' for k, v in biases.items())}")
        return biases
    except Exception as e:
        print(f"Error calculating gold model biases: {e}")
        return {
            "random_forest": 0.0,
            "linear_regression": 0.0,
            "mlp": 0.0,
            "xgboost": 0.0,
            "lstm": 0.0
        }


def load_gold_model_hyperparameters():
    """
    Loads custom tuned hyperparameters from MongoDB.
    If none exist, returns empty dict.
    """
    try:
        doc = db.gold_model_hyperparameters.find_one({"type": "gold_params"})
        if doc and "params" in doc:
            return doc["params"]
    except Exception as e:
        print(f"Error loading gold hyperparameters: {e}")
    return {}


def tune_and_save_gold_hyperparameters():
    """
    Runs hyperparameter search on the last 180 days of gold data,
    and caches the best params in MongoDB.
    """
    try:
        import predictor as pred_module
        import pandas as pd
        history = list(db.gold_history.find({}, {"_id": 0}).sort("date", 1))
        if len(history) < 50:
            print("Insufficient gold history to tune parameters.")
            return
            
        df = pd.DataFrame(history)
        df["close"] = df["world_price"]
        df["open"] = df["world_price"]
        df["high"] = df["world_price"]
        df["low"] = df["world_price"]
        df["volume"] = 1.0
        
        # Merge macro indicators
        macro_cursor = db.macro_history.find({}, {"_id": 0})
        macro_df = pd.DataFrame(list(macro_cursor))
        if not macro_df.empty and "date" in macro_df.columns:
            df = pd.merge(df, macro_df, on="date", how="left")
            cols = [c for c in ["dxy", "us10y", "vix", "brent", "dji", "spx", "eurusd", "xagusd", "real_yield"] if c in df.columns]
            df[cols] = df[cols].ffill().bfill()
            
        df_ind = pred_module.calculate_technical_indicators(df)
        print("Running Hyperparameter Self-Optimization Grid Search for Gold...")
        best_params = pred_module.optimize_hyperparameters(df_ind, days_to_predict=1)
        
        if best_params:
            db.gold_model_hyperparameters.update_one(
                {"type": "gold_params"},
                {"$set": {
                    "type": "gold_params",
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "params": best_params
                }},
                upsert=True
            )
            print("Successfully updated Gold Hyperparameters cache in MongoDB.")
            print(f"Optimized Params: {best_params}")
    except Exception as e:
        print(f"Error tuning gold hyperparameters: {e}")


def backfill_gold_predictions_history(days=60):
    """
    Backfill T+1 prediction history by sliding through the gold_history.
    For each day d in the last `days` days, uses history up to d-1 to predict price at d,
    then stores the actual price at d as the ground truth.
    Skips dates that already have predictions with models saved.
    Returns a dict summarizing results.
    """
    try:
        import predictor as pred_module
        import pandas as pd
        import numpy as np

        history = list(db.gold_history.find({}, {"_id": 0}).sort("date", 1))
        if len(history) < 22:
            return {"status": "error", "message": "Không đủ lịch sử (cần ít nhất 22 ngày)"}

        # Load full macro history once
        macro_list = list(db.macro_history.find({}, {"_id": 0}))
        macro_df = pd.DataFrame(macro_list) if macro_list else pd.DataFrame()

        # Determine the slice of history to backfill (last `days` entries that have an actual price)
        target_indices = range(max(21, len(history) - days), len(history))

        saved_count = 0
        skipped_count = 0
        errors = []

        for idx in target_indices:
            target_item = history[idx]
            target_date = target_item["date"]
            actual_price = target_item.get("world_price", 0.0)

            if not actual_price or actual_price == 0.0:
                skipped_count += 1
                continue

            # Check if already backfilled
            existing = db.gold_predictions_history.find_one({"date": target_date})
            if existing and existing.get("models"):
                skipped_count += 1
                continue

            # Build dataframe using history up to yesterday (idx-1)
            hist_slice = history[:idx]
            df = pd.DataFrame(hist_slice)
            df["world_price"] = df["world_price"].replace(0.0, np.nan).ffill().bfill()
            df["close"] = df["world_price"]
            df["open"]  = df["world_price"]
            df["high"]  = df["world_price"]
            df["low"]   = df["world_price"]
            df["volume"] = 1.0

            # Merge macro data
            if not macro_df.empty and "date" in macro_df.columns:
                df = pd.merge(df, macro_df, on="date", how="left")
                cols = [c for c in ["dxy", "us10y", "vix", "brent", "dji", "spx", "eurusd", "xagusd"] if c in df.columns]
                df[cols] = df[cols].ffill().bfill()
            if "dji" not in df.columns:
                df["dji"] = 35000.0
            if "spx" not in df.columns:
                df["spx"] = 5000.0

            if len(df) < 21:
                skipped_count += 1
                continue

            try:
                df_ind = pred_module.calculate_technical_indicators(df)
                lr_p,   _, _    = pred_module.predict_future_prices(df_ind, days_to_predict=1)
                rf_p,   _, _, _ = pred_module.predict_future_prices_rf(df_ind, days_to_predict=1)
                mlp_p,  _, _, _ = pred_module.predict_future_prices_mlp(df_ind, days_to_predict=1)
                xgb_p,  _, _, _ = pred_module.predict_future_prices_xgb(df_ind, days_to_predict=1)
                lstm_p, _, _, _ = pred_module.predict_future_prices_lstm(df_ind, days_to_predict=1)

                weights = calculate_dynamic_ensemble_weights()
                w_rf   = weights.get("random_forest",    0.20)
                w_lr   = weights.get("linear_regression", 0.20)
                w_mlp  = weights.get("mlp",              0.20)
                w_xgb  = weights.get("xgboost",          0.20)
                w_lstm = weights.get("lstm",             0.20)
                ens_p  = float(rf_p[0]*w_rf + lr_p[0]*w_lr + mlp_p[0]*w_mlp
                               + xgb_p[0]*w_xgb + lstm_p[0]*w_lstm)

                doc = {
                    "date": target_date,
                    "actual_price": float(actual_price),
                    "models": {
                        "random_forest":     float(rf_p[0]),
                        "linear_regression": float(lr_p[0]),
                        "mlp":               float(mlp_p[0]),
                        "xgboost":           float(xgb_p[0]),
                        "lstm":              float(lstm_p[0]),
                        "ensemble":          ens_p
                    }
                }
                db.gold_predictions_history.update_one(
                    {"date": target_date}, {"$set": doc}, upsert=True
                )
                saved_count += 1
            except Exception as day_err:
                errors.append(f"{target_date}: {day_err}")

        return {
            "status": "success",
            "saved": saved_count,
            "skipped": skipped_count,
            "errors": errors
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def auto_fill_gold_history_from_yfinance(days=90):
    """
    Downloads historical close prices for gold futures from Yahoo Finance (GC=F),
    and inserts any missing dates in gold_history.
    """
    try:
        import yfinance as yf
        print(f"Auto-filling missing gold history points from yfinance for past {days} days...")
        
        # Download GC=F data
        df = yf.download("GC=F", period="3mo", progress=False)
        if df.empty:
            print("yfinance download for GC=F was empty.")
            return
            
        import pandas as pd
        if isinstance(df.columns, pd.MultiIndex):
            close_prices = df[("Close", "GC=F")].ffill().bfill()
        else:
            close_prices = df["Close"].ffill().bfill()
            
        inserted_count = 0
        for index, val in close_prices.items():
            date_str = index.strftime("%Y-%m-%d")
            price = float(val)
            if pd.isna(price) or price <= 0:
                continue
                
            existing = db.gold_history.find_one({"date": date_str})
            if not existing:
                # Find closest previous entry
                prev_doc = db.gold_history.find_one({"date": {"$lt": date_str}}, sort=[("date", -1)])
                if prev_doc:
                    sjc_bar_buy = prev_doc.get("sjc_bar_buy", 0.0)
                    sjc_bar_sell = prev_doc.get("sjc_bar_sell", 0.0)
                    sjc_ring_buy = prev_doc.get("sjc_ring_buy", 0.0)
                    sjc_ring_sell = prev_doc.get("sjc_ring_sell", 0.0)
                    ratio = (prev_doc.get("world_price_vnd", 0.0) / prev_doc.get("world_price", 1.0)) if prev_doc.get("world_price", 0) > 0 else 25400 * 1.20565
                    world_price_vnd = price * ratio
                else:
                    sjc_bar_buy = 140000000.0
                    sjc_bar_sell = 143000000.0
                    sjc_ring_buy = 140000000.0
                    sjc_ring_sell = 143000000.0
                    world_price_vnd = price * 25400 * 1.20565
                    
                new_doc = {
                    "date": date_str,
                    "sjc_bar_buy": float(sjc_bar_buy),
                    "sjc_bar_sell": float(sjc_bar_sell),
                    "sjc_ring_buy": float(sjc_ring_buy),
                    "sjc_ring_sell": float(sjc_ring_sell),
                    "world_price": round(price, 2),
                    "world_price_vnd": round(world_price_vnd, -3)
                }
                db.gold_history.update_one({"date": date_str}, {"$set": new_doc}, upsert=True)
                inserted_count += 1
                
        if inserted_count > 0:
            print(f"Auto-fill: inserted/updated {inserted_count} missing gold history dates.")
    except Exception as e:
        print(f"Error in auto_fill_gold_history_from_yfinance: {e}")

def resolve_unresolved_predictions():
    """
    Finds past predictions (date < today) that do not have actual_price resolved yet,
    and updates them using matching entries in gold_history.
    """
    try:
        today_date = datetime.now().strftime("%Y-%m-%d")
        unresolved = list(db.gold_predictions_history.find({
            "date": {"$lt": today_date},
            "$or": [
                {"actual_price": None},
                {"actual_price": 0.0},
                {"actual_price": {"$exists": False}}
            ]
        }))
        
        resolved_count = 0
        for pred in unresolved:
            date_str = pred["date"]
            hist = db.gold_history.find_one({"date": date_str})
            if hist and hist.get("world_price", 0.0) > 0:
                db.gold_predictions_history.update_one(
                    {"date": date_str},
                    {"$set": {"actual_price": float(hist["world_price"])}}
                )
                resolved_count += 1
                
        if resolved_count > 0:
            print(f"Resolved {resolved_count} past prediction records using gold_history.")
    except Exception as e:
        print(f"Error in resolve_unresolved_predictions: {e}")

def update_gold_predictions_history():
    """
    Saves/updates the T+1 predictions and actual prices in db.gold_predictions_history.
    Computes predictions for today (using history up to yesterday) and predictions for tomorrow (using history up to today).
    """
    try:
        # Pre-execution: auto-fill any missing dates and resolve previous predictions
        auto_fill_gold_history_from_yfinance()
        resolve_unresolved_predictions()

        # Self-learning: Trigger daily hyperparameter grid search optimization
        try:
            today_str = datetime.now().strftime("%Y-%m-%d")
            existing_tuning = db.gold_model_hyperparameters.find_one({"type": "gold_params"})
            if not existing_tuning or existing_tuning.get("updated_at", "").split(" ")[0] != today_str:
                print("New day detected. Auto-tuning model hyperparameters...")
                tune_and_save_gold_hyperparameters()
        except Exception as tune_err:
            print(f"Error during scheduled hyperparameter auto-tuning: {tune_err}")

        history = list(db.gold_history.find().sort("date", 1))
        if len(history) < 21:
            return
            
        import predictor
        import pandas as pd
        import numpy as np

        # Load custom hyperparameters & rolling biases (self-learning)
        params = load_gold_model_hyperparameters()
        biases = calculate_gold_model_biases(limit=15)
        
        # 1. Update prediction for today (last item in history)
        today_item = history[-1]
        today_date = today_item["date"]
        actual_price = today_item.get("world_price", 0.0)
        
        # Check if we already have the predictions for today (meaning they were saved yesterday)
        existing_doc = db.gold_predictions_history.find_one({"date": today_date})
        if existing_doc and "models" in existing_doc:
            # We already have the predictions! Just update the actual price.
            db.gold_predictions_history.update_one(
                {"date": today_date},
                {"$set": {"actual_price": float(actual_price)}}
            )
            print(f"Successfully updated actual price for today's existing prediction: {today_date}")
        else:
            # If not present (e.g. initial run or backfill), calculate raw prediction as fallback
            # History up to yesterday
            hist_slice_today = history[:-1]
            df_today = pd.DataFrame(hist_slice_today)

            df_today["world_price"] = df_today["world_price"].replace(0.0, np.nan).ffill().bfill()
            df_today["close"] = df_today["world_price"]
            df_today["open"] = df_today["world_price"]
            df_today["high"] = df_today["world_price"]
            df_today["low"] = df_today["world_price"]
            df_today["volume"] = 1.0
            
            # Merge macro data
            macro_cursor = db.macro_history.find({}, {"_id": 0})
            macro_df = pd.DataFrame(list(macro_cursor))
            if not macro_df.empty and "date" in macro_df.columns:
                df_today = pd.merge(df_today, macro_df, on="date", how="left")
                cols = [c for c in ["dxy", "us10y", "vix", "brent", "dji", "spx", "eurusd", "xagusd", "real_yield"] if c in df_today.columns]
                df_today[cols] = df_today[cols].ffill().bfill()

            # Calculate technical indicators & predict today
            df_ind_today = predictor.calculate_technical_indicators(df_today)
            lr_p, _, _ = predictor.predict_future_prices(df_ind_today, days_to_predict=1)
            rf_p, _, _, _ = predictor.predict_future_prices_rf(df_ind_today, days_to_predict=1, params=params.get("random_forest"))
            mlp_p, _, _, _ = predictor.predict_future_prices_mlp(df_ind_today, days_to_predict=1, params=params.get("mlp"))
            xgb_p, _, _, _ = predictor.predict_future_prices_xgb(df_ind_today, days_to_predict=1, params=params.get("xgboost"))
            lstm_p, _, _, _ = predictor.predict_future_prices_lstm(df_ind_today, days_to_predict=1)
            
            # Apply rolling bias correction (today)
            rf_p = [p + biases.get("random_forest", 0.0) for p in rf_p]
            lr_p = [p + biases.get("linear_regression", 0.0) for p in lr_p]
            mlp_p = [p + biases.get("mlp", 0.0) for p in mlp_p]
            xgb_p = [p + biases.get("xgboost", 0.0) for p in xgb_p]
            lstm_p = [p + biases.get("lstm", 0.0) for p in lstm_p]
            
            # Fetch dynamic weights
            weights_today = calculate_dynamic_ensemble_weights()
            w_rf_t   = weights_today.get("random_forest", 0.20)
            w_lr_t   = weights_today.get("linear_regression", 0.20)
            w_mlp_t  = weights_today.get("mlp", 0.20)
            w_xgb_t  = weights_today.get("xgboost", 0.20)
            w_lstm_t = weights_today.get("lstm", 0.20)
            ens_p = float(rf_p[0]*w_rf_t + lr_p[0]*w_lr_t + mlp_p[0]*w_mlp_t + xgb_p[0]*w_xgb_t + lstm_p[0]*w_lstm_t)
            
            today_doc = {
                "date": today_date,
                "actual_price": float(actual_price),
                "models": {
                    "random_forest": float(rf_p[0]),
                    "linear_regression": float(lr_p[0]),
                    "mlp": float(mlp_p[0]),
                    "xgboost": float(xgb_p[0]),
                    "lstm": float(lstm_p[0]),
                    "ensemble": ens_p
                }
            }
            db.gold_predictions_history.update_one({"date": today_date}, {"$set": today_doc}, upsert=True)
            print(f"Successfully calculated and saved prediction history for today: {today_date}")
        
        # 2. Add prediction for tomorrow (T+1 from today's perspective)
        try:
            today_dt = datetime.strptime(today_date, "%Y-%m-%d")
            tomorrow_dt = today_dt + timedelta(days=1)
            while tomorrow_dt.weekday() >= 5: # Exclude Saturday and Sunday
                tomorrow_dt += timedelta(days=1)
            tomorrow_date = tomorrow_dt.strftime("%Y-%m-%d")
            
            df_tomorrow = pd.DataFrame(history)
            df_tomorrow["world_price"] = df_tomorrow["world_price"].replace(0.0, np.nan).ffill().bfill()
            df_tomorrow["close"] = df_tomorrow["world_price"]
            df_tomorrow["open"] = df_tomorrow["world_price"]
            df_tomorrow["high"] = df_tomorrow["world_price"]
            df_tomorrow["low"] = df_tomorrow["world_price"]
            df_tomorrow["volume"] = 1.0
            
            # Merge macro data
            macro_cursor = db.macro_history.find({}, {"_id": 0})
            macro_df = pd.DataFrame(list(macro_cursor))
            if not macro_df.empty and "date" in macro_df.columns:
                df_tomorrow = pd.merge(df_tomorrow, macro_df, on="date", how="left")
                cols = [c for c in ["dxy", "us10y", "vix", "brent", "dji", "spx", "eurusd", "xagusd", "real_yield"] if c in df_tomorrow.columns]
                df_tomorrow[cols] = df_tomorrow[cols].ffill().bfill()

            df_ind_tomorrow = predictor.calculate_technical_indicators(df_tomorrow)
            lr_pt, _, _ = predictor.predict_future_prices(df_ind_tomorrow, days_to_predict=1)
            rf_pt, _, _, _ = predictor.predict_future_prices_rf(df_ind_tomorrow, days_to_predict=1, params=params.get("random_forest"))
            mlp_pt, _, _, _ = predictor.predict_future_prices_mlp(df_ind_tomorrow, days_to_predict=1, params=params.get("mlp"))
            xgb_pt, _, _, _ = predictor.predict_future_prices_xgb(df_ind_tomorrow, days_to_predict=1, params=params.get("xgboost"))
            lstm_pt, _, _, _ = predictor.predict_future_prices_lstm(df_ind_tomorrow, days_to_predict=1)
            
            # Apply rolling bias correction (tomorrow)
            rf_pt = [p + biases.get("random_forest", 0.0) for p in rf_pt]
            lr_pt = [p + biases.get("linear_regression", 0.0) for p in lr_pt]
            mlp_pt = [p + biases.get("mlp", 0.0) for p in mlp_pt]
            xgb_pt = [p + biases.get("xgboost", 0.0) for p in xgb_pt]
            lstm_pt = [p + biases.get("lstm", 0.0) for p in lstm_pt]

            # Fetch dynamic weights
            weights = calculate_dynamic_ensemble_weights()
            w_rf   = weights.get("random_forest", 0.20)
            w_lr   = weights.get("linear_regression", 0.20)
            w_mlp  = weights.get("mlp", 0.20)
            w_xgb  = weights.get("xgboost", 0.20)
            w_lstm = weights.get("lstm", 0.20)
            ens_pt = float(rf_pt[0]*w_rf + lr_pt[0]*w_lr + mlp_pt[0]*w_mlp + xgb_pt[0]*w_xgb + lstm_pt[0]*w_lstm)
 
            tomorrow_doc = {
                "date": tomorrow_date,
                "actual_price": None,
                "models": {
                    "random_forest": float(rf_pt[0]),
                    "linear_regression": float(lr_pt[0]),
                    "mlp": float(mlp_pt[0]),
                    "xgboost": float(xgb_pt[0]),
                    "lstm": float(lstm_pt[0]),
                    "ensemble": ens_pt
                }
            }
            db.gold_predictions_history.update_one({"date": tomorrow_date}, {"$set": tomorrow_doc}, upsert=True)
            print(f"Successfully updated prediction history for tomorrow: {tomorrow_date}")
        except Exception as tomorrow_err:
            print(f"Error calculating prediction for tomorrow: {tomorrow_err}")

            
    except Exception as e:
        print(f"Error updating gold predictions history: {e}")

def generate_macro_error_analysis(df, target_date, error_val):
    """
    Analyzes how DXY, US10Y, Brent Crude, VIX changed on target_date vs previous day
    and outputs a macro explanation for the prediction error based on asset correlations.
    """
    try:
        if df is None or df.empty:
            return ""
        idx_list = df[df["date"] == target_date].index
        if len(idx_list) == 0:
            return ""
        idx = idx_list[0]
        if idx == 0:
            return ""
        
        row_curr = df.iloc[idx]
        row_prev = df.iloc[idx - 1]
        
        explanations = []
        
        # 1. Dollar Index (DXY) - Inverse correlation (-)
        if "dxy" in row_curr and "dxy" in row_prev:
            dxy_diff = row_curr["dxy"] - row_prev["dxy"]
            if abs(dxy_diff) > 0.05:
                direction = "tăng" if dxy_diff > 0 else "giảm"
                impact = "áp lực giảm giá lên Vàng (-)" if dxy_diff > 0 else "động lực tăng giá cho Vàng (+)"
                explanations.append(f"DXY {direction} ({dxy_diff:+.2f}đ) ➔ {impact}")
                
        # 2. US 10-Year Bond Yield (US10Y) - Inverse correlation (-)
        if "us10y" in row_curr and "us10y" in row_prev:
            us10y_diff = row_curr["us10y"] - row_prev["us10y"]
            if abs(us10y_diff) > 0.01:
                direction = "tăng" if us10y_diff > 0 else "giảm"
                impact = "áp lực giảm giá lên Vàng (-)" if us10y_diff > 0 else "động lực tăng giá cho Vàng (+)"
                explanations.append(f"Lợi suất US10Y {direction} ({us10y_diff:+.2f}%) ➔ {impact}")
                
        # 3. Brent Crude Oil - Complex correlation (~)
        if "brent" in row_curr and "brent" in row_prev:
            brent_diff = row_curr["brent"] - row_prev["brent"]
            if abs(brent_diff) > 0.2:
                direction = "tăng" if brent_diff > 0 else "giảm"
                if brent_diff > 0:
                    impact = "tăng lạm phát vĩ mô ➔ hỗ trợ Vàng (+)"
                else:
                    impact = "giảm lạm phát vĩ mô ➔ giảm động lực Vàng (-)"
                explanations.append(f"Giá dầu Brent {direction} (${brent_diff:+.2f}/thùng) ➔ {impact}")
                
        # 4. Fear Index (VIX) - Direct correlation (+)
        if "vix" in row_curr and "vix" in row_prev:
            vix_diff = row_curr["vix"] - row_prev["vix"]
            if abs(vix_diff) > 0.2:
                direction = "tăng" if vix_diff > 0 else "giảm"
                impact = "tăng tâm lý e ngại rủi ro ➔ hỗ trợ Vàng (+)" if vix_diff > 0 else "tâm lý e ngại rủi ro giảm ➔ giảm nhu cầu trú ẩn (-)"
                explanations.append(f"Chỉ số VIX {direction} ({vix_diff:+.2f}đ) ➔ {impact}")
                
        # 5. Major Events
        event_notes = []
        if "days_to_fed" in row_curr and row_curr["days_to_fed"] == 0:
            event_notes.append("Quyết định lãi suất của FED được công bố hôm nay (FED tăng lãi suất ➔ Vàng giảm (-); FED giảm lãi suất ➔ Vàng tăng (+)).")
        if "days_to_cpi" in row_curr and row_curr["days_to_cpi"] == 0:
            event_notes.append("Số liệu lạm phát Mỹ (CPI) được công bố hôm nay (CPI tăng ➔ Vàng tăng (+); CPI giảm ➔ Vàng giảm (-)).")
        if "days_to_nfp" in row_curr and row_curr["days_to_nfp"] == 0:
            event_notes.append("Báo cáo việc làm phi nông nghiệp Mỹ (NFP) công bố hôm nay (NFP cao ➔ Vàng giảm (-); NFP thấp/xấu ➔ Vàng tăng (+)).")
            
        macro_desc = " | ".join(explanations)
        if not macro_desc:
            macro_desc = "biến động vĩ mô nhẹ"
            
        event_desc = " ".join(event_notes)
        if event_desc:
            macro_desc += f" | {event_desc}"
            
        # Formulate full explanation
        error_dir = "thấp hơn" if error_val > 0 else "cao hơn"
        learned_action = "tăng" if error_val > 0 else "giảm"
        
        full_analysis = (
            f"Mô hình dự báo {error_dir} thực tế {abs(error_val):.2f} USD do chưa phản ánh hết mức độ tác động khi "
            f"[{macro_desc}]. Hệ thống đã ghi nhận quan hệ nhân quả vĩ mô này để tự động hiệu chỉnh ({learned_action} dự báo) cho các phiên tiếp theo."
        )
        return full_analysis
    except Exception as e:
        print(f"Error in generate_macro_error_analysis: {e}")
        return ""

def get_gold_predictions():
    """
    Loads World gold price history from MongoDB, merges macroeconomic indicators
    and event proximity features, runs technical indicators, and returns predictions using predictor.py.
    """
    init_db() # Ensure seeded
    try:
        cursor = db.gold_history.find({}, {"_id": 0}).sort("date", 1)
        history_list = list(cursor)
        if len(history_list) < 20:
            return {
                "status": "error",
                "message": "Không đủ dữ liệu lịch sử vàng để phân tích (yêu cầu tối thiểu 20 phiên)."
            }
            
        import predictor
        # Convert to DataFrame
        df = pd.DataFrame(history_list)
        
        # Ensure world_price has no zeros or NaNs (interpolate or forward fill)
        df["world_price"] = df["world_price"].replace(0.0, np.nan).ffill().bfill()
        
        # World gold price (USD/ounce) is the primary prediction target
        df["close"] = df["world_price"]
        df["open"] = df["world_price"]
        df["high"] = df["world_price"]
        df["low"] = df["world_price"]
        df["volume"] = 1.0 # placeholder
        
        # --- Merge Macro Indicators ---
        try:
            macro_cursor = db.macro_history.find({}, {"_id": 0})
            macro_list = list(macro_cursor)
            if macro_list:
                macro_df = pd.DataFrame(macro_list)
                # Merge on date
                df = pd.merge(df, macro_df, on="date", how="left")
                # Forward fill then backward fill to handle holidays/weekends
                cols_to_fill = [c for c in ["dxy", "us10y", "vix", "brent", "dji", "spx", "gld", "gld_trust"] if c in df.columns]
                df[cols_to_fill] = df[cols_to_fill].ffill().bfill()
                if "dji" not in df.columns:
                    df["dji"] = 35000.0
                if "spx" not in df.columns:
                    df["spx"] = 5000.0
                if "gld" not in df.columns:
                    df["gld"] = 200.0
                if "gld_trust" not in df.columns:
                    df["gld_trust"] = 200.0
            else:
                # Fallback to defaults if macro_history is empty
                df["dxy"] = 100.0
                df["us10y"] = 4.0
                df["vix"] = 15.0
                df["brent"] = 75.0
                df["dji"] = 35000.0
                df["spx"] = 5000.0
                df["gld"] = 200.0
                df["gld_trust"] = 200.0
        except Exception as macro_err:
            print(f"Error merging macro history: {macro_err}")
            df["dxy"] = 100.0
            df["us10y"] = 4.0
            df["vix"] = 15.0
            df["brent"] = 75.0
            df["dji"] = 35000.0
            df["spx"] = 5000.0
            df["gld"] = 200.0
            df["gld_trust"] = 200.0
            
        # --- Add Event Proximity Features ---
        try:
            days_to_fed, days_to_cpi, days_to_nfp = calculate_days_until_events(df["date"])
            df["days_to_fed"] = days_to_fed
            df["days_to_cpi"] = days_to_cpi
            df["days_to_nfp"] = days_to_nfp
        except Exception as event_err:
            print(f"Error adding event proximity features: {event_err}")
            df["days_to_fed"] = 15.0
            df["days_to_cpi"] = 15.0
            df["days_to_nfp"] = 15.0
        
        # --- Add Geopolitical Conflict Events ---
        conflict_events = []
        try:
            conflict_events = fetch_geopolitical_conflicts()
        except Exception as e:
            print(f"Error fetching geopolitical conflicts for predictions: {e}")
        
        weights = calculate_dynamic_ensemble_weights()
        params = load_gold_model_hyperparameters()
        biases = calculate_gold_model_biases(limit=15)
        analysis = predictor.analyze_and_recommend(df, conversion_factor=None, is_usd=True, model_weights=weights, params=params, biases=biases, conflict_events=conflict_events)
        
        # Clean history array for plotting in frontend
        # We need past 180 days (6 months) to support Dow Jones 6M chart
        plot_history = []
        for _, row in df.tail(180).iterrows():
            plot_history.append({
                "date": row["date"],
                "sjc_bar_buy": float(row.get("sjc_bar_buy")) if pd.notna(row.get("sjc_bar_buy")) else 0.0,
                "sjc_bar_sell": float(row.get("sjc_bar_sell")) if pd.notna(row.get("sjc_bar_sell")) else 0.0,
                "sjc_ring_buy": float(row.get("sjc_ring_buy")) if pd.notna(row.get("sjc_ring_buy")) else 0.0,
                "sjc_ring_sell": float(row.get("sjc_ring_sell")) if pd.notna(row.get("sjc_ring_sell")) else 0.0,
                "world_price": float(row.get("world_price")) if pd.notna(row.get("world_price")) else 0.0,
                "world_price_vnd": float(row.get("world_price_vnd")) if pd.notna(row.get("world_price_vnd")) else 0.0,
                "dxy": float(row.get("dxy")) if pd.notna(row.get("dxy")) else 100.0,
                "brent": float(row.get("brent")) if pd.notna(row.get("brent")) else 75.0,
                "dji": float(row.get("dji")) if pd.notna(row.get("dji")) else 35000.0,
                "spx": float(row.get("spx")) if pd.notna(row.get("spx")) else 5000.0,
                "gld": float(row.get("gld")) if pd.notna(row.get("gld")) else 0.0,
                "gld_trust": float(row.get("gld_trust")) if pd.notna(row.get("gld_trust")) else 0.0
            })
            
        analysis["gold_history"] = plot_history
        
        # Extract latest macro values for display in the frontend diagram
        latest_macro = {}
        if not df.empty:
            last_row = df.iloc[-1]
            if "dxy" in last_row:
                latest_macro["dxy"] = float(last_row["dxy"])
            if "us10y" in last_row:
                latest_macro["us10y"] = float(last_row["us10y"])
            if "vix" in last_row:
                latest_macro["vix"] = float(last_row["vix"])
            if "brent" in last_row:
                latest_macro["brent"] = float(last_row["brent"])
            if "days_to_fed" in last_row:
                latest_macro["days_to_fed"] = int(last_row["days_to_fed"])
            if "days_to_cpi" in last_row:
                latest_macro["days_to_cpi"] = int(last_row["days_to_cpi"])
            if "days_to_nfp" in last_row:
                latest_macro["days_to_nfp"] = int(last_row["days_to_nfp"])
        analysis["latest_macro"] = latest_macro
        
        # Apply automatic bias correction based on 15-day rolling average (self-learning)
        try:
            latest_resolved = db.gold_predictions_history.find_one(
                {"actual_price": {"$ne": None, "$gt": 0.0}},
                sort=[("date", -1)]
            )
            latest_adjustments = {}
            from datetime import datetime
            
            for m_key in ["random_forest", "linear_regression", "mlp", "xgboost", "ensemble"]:
                corr = biases.get(m_key if m_key != "ensemble" else "random_forest", 0.0)
                latest_err = 0.0
                macro_explain = ""
                if latest_resolved and "models" in latest_resolved:
                    act = latest_resolved["actual_price"]
                    pred_val = latest_resolved["models"].get(m_key)
                    if pred_val is not None:
                        latest_err = act - pred_val
                        macro_explain = generate_macro_error_analysis(df, latest_resolved["date"], latest_err)
                
                latest_adjustments[m_key] = {
                    "date": latest_resolved["date"] if latest_resolved else "",
                    "model": m_key,
                    "error": float(latest_err),
                    "macro_analysis": macro_explain,
                    "timestamp": datetime.now().isoformat(),
                    "is_auto": True,
                    "rolling_bias": float(corr)
                }
            
            analysis["latest_adjustments"] = latest_adjustments
            analysis["latest_adjustment"] = latest_adjustments.get("random_forest")
        except Exception as adj_err:
            print(f"Error applying automatic learning adjustment: {adj_err}")
            
        return analysis
    except Exception as e:
        print(f"Error generating gold predictions: {e}")
        return {
            "status": "error",
            "message": f"Lỗi phân tích dự đoán vàng: {e}"
        }


def seed_us_economic_calendar():
    """
    Seeds the us_economic_calendar collection in MongoDB with 2026 events.
    """
    try:
        print("Seeding US economic calendar for 2026...")
        events = [
            # FED (FOMC Meetings)
            {"date": "2026-01-28", "event": "Quyết định lãi suất của FED (FOMC)", "category": "FED", "importance": "Rất cao", "impact": "Họp FED đầu năm. Quyết định tăng/giảm/giữ nguyên lãi suất ảnh hưởng cực mạnh đến sức mạnh USD và Vàng."},
            {"date": "2026-03-18", "event": "Quyết định lãi suất & Biểu đồ Dot Plot của FED", "category": "FED", "importance": "Rất cao", "impact": "Công bố dự phóng lãi suất trung hạn (Dot Plot). Vàng biến động rất lớn dựa trên định hướng tương lai."},
            {"date": "2026-04-29", "event": "Quyết định lãi suất của FED (FOMC)", "category": "FED", "importance": "Rất cao", "impact": "Họp FED thường kỳ định giá lại kỳ vọng kinh tế quý 1. Ảnh hưởng trực tiếp tỷ giá DXY."},
            {"date": "2026-06-17", "event": "Quyết định lãi suất & Biểu đồ Dot Plot của FED", "category": "FED", "importance": "Rất cao", "impact": "Họp quyết định lãi suất giữa năm & cập nhật dự báo tăng trưởng/lạm phát mới. Điểm xoay chiều quan trọng cho Vàng."},
            {"date": "2026-07-29", "event": "Quyết định lãi suất của FED (FOMC)", "category": "FED", "importance": "Rất cao", "impact": "Tuyên bố chính sách tiền tệ quý 3. Vàng thường tăng mạnh nếu FED phát tín hiệu nới lỏng."},
            {"date": "2026-09-16", "event": "Quyết định lãi suất & Biểu đồ Dot Plot của FED", "category": "FED", "importance": "Rất cao", "impact": "Công bố Dot Plot quý 3. Thường định hình xu hướng dòng vốn quốc tế cuối năm."},
            {"date": "2026-10-28", "event": "Quyết định lãi suất của FED (FOMC)", "category": "FED", "importance": "Rất cao", "impact": "Họp FED đầu quý 4. Đánh giá rủi ro lạm phát cuối năm."},
            {"date": "2026-12-16", "event": "Quyết định lãi suất & Biểu đồ Dot Plot của FED", "category": "FED", "importance": "Rất cao", "impact": "Họp FED cuối năm, chốt phương hướng chính sách cho năm tiếp theo. Vàng đón nhận biến động cực lớn."},
            
            # CPI Releases
            {"date": "2026-01-13", "event": "Công bố chỉ số lạm phát CPI Mỹ (Tháng 12/2025)", "category": "CPI", "importance": "Rất cao", "impact": "Đo lường mức độ lạm phát. Lạm phát tăng vượt kỳ vọng đẩy dòng tiền vào Vàng làm công cụ trú ẩn lạm phát."},
            {"date": "2026-02-11", "event": "Công bố chỉ số lạm phát CPI Mỹ (Tháng 1/2026)", "category": "CPI", "importance": "Rất cao", "impact": "Lạm phát tăng -> FED giữ lãi suất cao lâu hơn -> Vàng giảm. Lạm phát giảm -> Kỳ vọng hạ lãi suất -> Vàng tăng."},
            {"date": "2026-03-11", "event": "Công bố chỉ số lạm phát CPI Mỹ (Tháng 2/2026)", "category": "CPI", "importance": "Rất cao", "impact": "CPI cốt lõi định hướng hành động tiếp theo của FED tại cuộc họp tháng 3."},
            {"date": "2026-04-14", "event": "Công bố chỉ số lạm phát CPI Mỹ (Tháng 3/2026)", "category": "CPI", "importance": "Rất cao", "impact": "CPI đầu quý 2, phản ánh mức chi tiêu tiêu dùng sau tết."},
            {"date": "2026-05-13", "event": "Công bố chỉ số lạm phát CPI Mỹ (Tháng 4/2026)", "category": "CPI", "importance": "Rất cao", "impact": "Số liệu CPI làm tiền đề dự báo quyết định FED tháng 6."},
            {"date": "2026-06-10", "event": "Công bố chỉ số lạm phát CPI Mỹ (Tháng 5/2026)", "category": "CPI", "importance": "Rất cao", "impact": "Chỉ số lạm phát sát cuộc họp FED tháng 6, gây áp lực điều chỉnh danh mục vàng."},
            {"date": "2026-07-14", "event": "Công bố chỉ số lạm phát CPI Mỹ (Tháng 6/2026)", "category": "CPI", "importance": "Rất cao", "impact": "Lạm phát giữa năm, chốt số liệu quý 2."},
            {"date": "2026-08-12", "event": "Công bố chỉ số lạm phát CPI Mỹ (Tháng 7/2026)", "category": "CPI", "importance": "Rất cao", "impact": "Ảnh hưởng đến tâm lý giao dịch vàng mùa hè thấp điểm."},
            {"date": "2026-09-11", "event": "Công bố chỉ số lạm phát CPI Mỹ (Tháng 8/2026)", "category": "CPI", "importance": "Rất cao", "impact": "Số liệu CPI mở đầu chuỗi họp chính sách mùa thu của FED."},
            {"date": "2026-10-14", "event": "Công bố chỉ số lạm phát CPI Mỹ (Tháng 9/2026)", "category": "CPI", "importance": "Rất cao", "impact": "Chốt lạm phát quý 3, định hình chính sách quý cuối năm."},
            {"date": "2026-11-12", "event": "Công bố chỉ số lạm phát CPI Mỹ (Tháng 10/2026)", "category": "CPI", "importance": "Rất cao", "impact": "Số liệu CPI quan trọng quyết định quy mô nới lỏng lãi suất của họp tháng 12."},
            {"date": "2026-12-11", "event": "Công bố chỉ số lạm phát CPI Mỹ (Tháng 11/2026)", "category": "CPI", "importance": "Rất cao", "impact": "Báo cáo CPI cuối cùng của năm 2026."},

            # NFP Employment Reports
            {"date": "2026-01-09", "event": "Báo cáo việc làm phi nông nghiệp Mỹ (NFP Tháng 12/2025)", "category": "Việc làm", "importance": "Cao", "impact": "Thị trường lao động mạnh -> Vàng giảm (do kinh tế tốt làm tăng sức mạnh USD). Thị trường yếu -> Vàng tăng."},
            {"date": "2026-02-06", "event": "Báo cáo việc làm phi nông nghiệp Mỹ (NFP Tháng 1/2026)", "category": "Việc làm", "importance": "Cao", "impact": "Thị trường việc làm đầu năm, gợi ý sức khỏe kinh tế Mỹ."},
            {"date": "2026-03-06", "event": "Báo cáo việc làm phi nông nghiệp Mỹ (NFP Tháng 2/2026)", "category": "Việc làm", "importance": "Cao", "impact": "NFP tháng 2, cơ sở đánh giá tỷ lệ thất nghiệp có gia tăng hay không."},
            {"date": "2026-04-03", "event": "Báo cáo việc làm phi nông nghiệp Mỹ (NFP Tháng 3/2026)", "category": "Việc làm", "importance": "Cao", "impact": "Chốt số liệu lao động quý 1."},
            {"date": "2026-05-08", "event": "Báo cáo việc làm phi nông nghiệp Mỹ (NFP Tháng 4/2026)", "category": "Việc làm", "importance": "Cao", "impact": "Số liệu việc làm đầu hè, ảnh hưởng đến kế hoạch lãi suất FED."},
            {"date": "2026-06-05", "event": "Báo cáo việc làm phi nông nghiệp Mỹ (NFP Tháng 5/2026)", "category": "Việc làm", "importance": "Cao", "impact": "Chỉ số lao động ngay trước thềm cuộc họp quan trọng của FED vào tháng 6."},
            {"date": "2026-07-03", "event": "Báo cáo việc làm phi nông nghiệp Mỹ (NFP Tháng 6/2026)", "category": "Việc làm", "importance": "Cao", "impact": "Báo cáo việc làm đầu quý 3."},
            {"date": "2026-08-07", "event": "Báo cáo việc làm phi nông nghiệp Mỹ (NFP Tháng 7/2026)", "category": "Việc làm", "importance": "Cao", "impact": "Thường có biến động thanh khoản nhỏ trong kỳ nghỉ hè."},
            {"date": "2026-09-04", "event": "Báo cáo việc làm phi nông nghiệp Mỹ (NFP Tháng 8/2026)", "category": "Việc làm", "importance": "Cao", "impact": "Số liệu lao động mở màn mùa giao dịch thu đông sôi động."},
            {"date": "2026-10-02", "event": "Báo cáo việc làm phi nông nghiệp Mỹ (NFP Tháng 9/2026)", "category": "Việc làm", "importance": "Cao", "impact": "Chốt số liệu lao động quý 3."},
            {"date": "2026-11-06", "event": "Báo cáo việc làm phi nông nghiệp Mỹ (NFP Tháng 10/2026)", "category": "Việc làm", "importance": "Cao", "impact": "Dữ liệu việc làm cận kề kỳ bầu cử giữa nhiệm kỳ hoặc hoạt động sản xuất cuối năm."},
            {"date": "2026-12-04", "event": "Báo cáo việc làm phi nông nghiệp Mỹ (NFP Tháng 11/2026)", "category": "Việc làm", "importance": "Cao", "impact": "Dữ liệu việc làm cuối cùng của năm, quyết định nốt hành vi FED."}
        ]
        
        operations = []
        for event in events:
            operations.append(UpdateOne(
                {"date": event["date"], "event": event["event"]},
                {"$set": event},
                upsert=True
            ))
            
        if operations:
            db.us_economic_calendar.bulk_write(operations)
            print(f"Successfully seeded {len(operations)} economic calendar events.")
    except Exception as e:
        print(f"Error seeding economic calendar: {e}")


def get_us_economic_calendar():
    """
    Retrieves all seeded economic calendar events from MongoDB, sorted by date.
    """
    try:
        cursor = db.us_economic_calendar.find({}, {"_id": 0}).sort("date", 1)
        return list(cursor)
    except Exception as e:
        print(f"Error getting economic calendar: {e}")
        return []


def fetch_macro_news():
    """
    Fetches international finance and macroeconomic news from VnExpress RSS and filters for gold/macro keywords.
    """
    import xml.etree.ElementTree as ET
    from bs4 import BeautifulSoup
    
    url = "https://vnexpress.net/rss/kinh-doanh.rss"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    keywords = ["vàng", "fed", "cpi", "lạm phát", "lãi suất", "biến động", "chiến tranh", "xung đột", "địa chính trị", "dầu thô", "usd", "tỷ giá"]
    
    try:
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code != 200:
            return []
            
        root = ET.fromstring(r.content)
        items = root.findall(".//item")
        
        filtered_news = []
        for item in items:
            title = item.find("title").text if item.find("title") is not None else ""
            desc_html = item.find("description").text if item.find("description") is not None else ""
            link = item.find("link").text if item.find("link") is not None else ""
            pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
            
            # Clean description
            desc_soup = BeautifulSoup(desc_html, "html.parser")
            description = desc_soup.text.strip()
            
            # Check keywords
            text_to_check = (title + " " + description).lower()
            matches = [kw for kw in keywords if kw in text_to_check]
            
            if matches:
                time_str = pub_date
                try:
                    parts = pub_date.split()
                    if len(parts) >= 5:
                        day = parts[1]
                        month = parts[2]
                        time = parts[4][:5] # HH:MM
                        month_num = {
                            "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
                            "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
                        }.get(month, "06")
                        time_str = f"{time} {day}/{month_num}"
                except:
                    pass
                    
                filtered_news.append({
                    "title": title,
                    "description": description,
                    "link": link,
                    "time": time_str,
                    "keywords": matches
                })
                
        # Limit to top 15 news articles
        return filtered_news[:15]
    except Exception as e:
        print(f"Error fetching macro news: {e}")
        return []


def fetch_us_macro_indicators():
    """
    Fetches US macroeconomic indicators from FRED:
    - CPI: CPIAUCSL (YoY change)
    - Unemployment: UNRATE
    - NFP: PAYEMS (monthly change)
    - Fed Funds Rate: FEDFUNDS
    - New Home Sales: HSN1F
    Returns a structured dict with latest values, dates, and trend signals.
    """
    # 1. Check cache first (6 hours TTL)
    now = datetime.now()
    try:
        cached = db.gold_prices.find_one({"_id": "cached_macro_indicators"})
        if cached:
            last_fetched = cached.get("fetched_at")
            if last_fetched and (now - last_fetched) < timedelta(hours=6):
                print("US macro indicators cache hit. Returning cached data.")
                data = cached.get("data")
                if data:
                    return data
    except Exception as e:
        print(f"Error checking US macro cache: {e}")

    import subprocess
    import io

    indicators = {
        "cpi": {
            "name": "CPI (Lạm Phát Mỹ)",
            "value": None,
            "prev_value": None,
            "date": None,
            "unit": "%/năm",
            "source": "FRED - fred.stlouisfed.org",
            "source_url": "https://fred.stlouisfed.org/series/CPIAUCSL",
            "trend": None,  # "up", "down", "flat"
            "gold_impact": "CPI tăng ➔ Vàng tăng; CPI giảm ➔ Vàng giảm",
            "impact_direction": "same",  # same = thuận chiều
            "description": "Chỉ số giá tiêu dùng - thước đo lạm phát chính của Mỹ"
        },
        "unemployment": {
            "name": "Tỷ Lệ Thất Nghiệp Mỹ",
            "value": None,
            "prev_value": None,
            "date": None,
            "unit": "%",
            "source": "FRED - fred.stlouisfed.org",
            "source_url": "https://fred.stlouisfed.org/series/UNRATE",
            "trend": None,
            "gold_impact": "Thất nghiệp tăng ➔ Vàng tăng; Thất nghiệp giảm ➔ Vàng giảm",
            "impact_direction": "same",
            "description": "Tỷ lệ người lao động không có việc làm"
        },
        "nonfarm": {
            "name": "Việc Làm Phi Nông Nghiệp (NFP)",
            "value": None,
            "prev_value": None,
            "date": None,
            "unit": "nghìn việc làm",
            "source": "FRED - fred.stlouisfed.org",
            "source_url": "https://fred.stlouisfed.org/series/PAYEMS",
            "trend": None,
            "gold_impact": "NFP cao ➔ Vàng giảm; NFP thấp/xấu ➔ Vàng tăng",
            "impact_direction": "opposite",  # opposite = nghịch chiều
            "description": "Số việc làm phi nông nghiệp tạo mới hàng tháng (Non-Farm Payrolls)"
        },
        "fed_rate": {
            "name": "Lãi Suất FED (Fed Funds Rate)",
            "value": None,
            "prev_value": None,
            "date": None,
            "unit": "%",
            "source": "FRED - fred.stlouisfed.org",
            "source_url": "https://fred.stlouisfed.org/series/FEDFUNDS",
            "trend": None,
            "gold_impact": "FED tăng lãi suất ➔ Vàng giảm; FED hạ lãi suất ➔ Vàng tăng",
            "impact_direction": "opposite",
            "description": "Lãi suất quỹ liên bang - công cụ tiền tệ chủ chốt của FED"
        },
        "new_home_sales": {
            "name": "Nhà Mới Bán Ra (New Home Sales)",
            "value": None,
            "prev_value": None,
            "date": None,
            "unit": "nghìn căn/tháng",
            "source": "FRED - fred.stlouisfed.org",
            "source_url": "https://fred.stlouisfed.org/series/HSN1F",
            "trend": None,
            "gold_impact": "Nhà bán tăng (kinh tế tốt) ➔ Vàng giảm; Nhà bán giảm (kinh tế xấu) ➔ Vàng tăng",
            "impact_direction": "opposite",
            "description": "Doanh số nhà mới - chỉ báo sức khỏe kinh tế và mức độ tự tin người tiêu dùng"
        }
    }

    def fetch_fred_series(series_id):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            result = subprocess.run(
                ["curl", "-s", "-L", url],
                capture_output=True,
                text=True,
                timeout=15
            )
            if result.returncode == 0:
                stdout_str = result.stdout
                if not stdout_str.strip():
                    return None
                df = pd.read_csv(io.StringIO(stdout_str))
                df = df.replace('.', pd.NA)
                df = df.dropna()
                df.columns = ['date', 'value']
                df['value'] = pd.to_numeric(df['value'])
                return df
        except Exception as e:
            print(f"Error fetching FRED series {series_id}: {e}")
        return None

    # --- 1. Fetch CPI (YoY) ---
    df_cpi = fetch_fred_series("CPIAUCSL")
    if df_cpi is not None and len(df_cpi) >= 13:
        df_cpi['prev_year_value'] = df_cpi['value'].shift(12)
        df_cpi['yoy'] = (df_cpi['value'] / df_cpi['prev_year_value'] - 1) * 100
        df_cpi = df_cpi.dropna()
        if not df_cpi.empty:
            latest = df_cpi.iloc[-1]
            prev = df_cpi.iloc[-2]
            indicators["cpi"]["value"] = round(float(latest['yoy']), 2)
            indicators["cpi"]["prev_value"] = round(float(prev['yoy']), 2)
            indicators["cpi"]["date"] = latest['date'][:7]

    # --- 2. Fetch Unemployment ---
    df_unrate = fetch_fred_series("UNRATE")
    if df_unrate is not None and len(df_unrate) >= 2:
        latest = df_unrate.iloc[-1]
        prev = df_unrate.iloc[-2]
        indicators["unemployment"]["value"] = float(latest['value'])
        indicators["unemployment"]["prev_value"] = float(prev['value'])
        indicators["unemployment"]["date"] = latest['date'][:7]

    # --- 3. Fetch Nonfarm Payrolls (monthly change) ---
    df_nfp = fetch_fred_series("PAYEMS")
    if df_nfp is not None and len(df_nfp) >= 2:
        df_nfp['monthly_change'] = df_nfp['value'] - df_nfp['value'].shift(1)
        df_nfp = df_nfp.dropna()
        if not df_nfp.empty:
            latest = df_nfp.iloc[-1]
            prev = df_nfp.iloc[-2]
            indicators["nonfarm"]["value"] = round(float(latest['monthly_change']), 1)
            indicators["nonfarm"]["prev_value"] = round(float(prev['monthly_change']), 1)
            indicators["nonfarm"]["date"] = latest['date'][:7]

    # --- 4. Fetch Fed Funds Rate ---
    df_fed = fetch_fred_series("FEDFUNDS")
    if df_fed is not None and len(df_fed) >= 2:
        latest = df_fed.iloc[-1]
        prev = df_fed.iloc[-2]
        indicators["fed_rate"]["value"] = float(latest['value'])
        indicators["fed_rate"]["prev_value"] = float(prev['value'])
        indicators["fed_rate"]["date"] = latest['date'][:7]

    # --- 5. Fetch New Home Sales ---
    df_home = fetch_fred_series("HSN1F")
    if df_home is not None and len(df_home) >= 2:
        latest = df_home.iloc[-1]
        prev = df_home.iloc[-2]
        indicators["new_home_sales"]["value"] = float(latest['value'])
        indicators["new_home_sales"]["prev_value"] = float(prev['value'])
        indicators["new_home_sales"]["date"] = latest['date'][:7]

    # --- 6. Calculate trends & fallback to default if still None ---
    FALLBACK_DATA = {
        "cpi": {"value": 2.4, "prev_value": 2.8, "date": "2025-03"},
        "unemployment": {"value": 4.2, "prev_value": 4.1, "date": "2025-04"},
        "nonfarm": {"value": 177, "prev_value": 185, "date": "2025-04"},
        "fed_rate": {"value": 3.62, "prev_value": 4.33, "date": "2025-05"},
        "new_home_sales": {"value": 724, "prev_value": 697, "date": "2025-03"}
    }

    for key, fb in FALLBACK_DATA.items():
        ind = indicators[key]
        if ind["value"] is None:
            ind["value"] = fb["value"]
            ind["prev_value"] = fb["prev_value"]
            ind["date"] = fb.get("date")
        else:
            if ind["prev_value"] is None:
                ind["prev_value"] = fb.get("prev_value")
            if ind["date"] is None:
                ind["date"] = fb.get("date")

        # Compute trend
        if ind["value"] is not None and ind["prev_value"] is not None:
            if ind["value"] > ind["prev_value"]:
                ind["trend"] = "up"
            elif ind["value"] < ind["prev_value"]:
                ind["trend"] = "down"
            else:
                ind["trend"] = "flat"
        elif ind["value"] is not None:
            ind["trend"] = "flat"

    # Save cache if we fetched valid data
    if any(ind["value"] is not None for ind in indicators.values()):
        try:
            db.gold_prices.replace_one(
                {"_id": "cached_macro_indicators"},
                {"_id": "cached_macro_indicators", "fetched_at": now, "data": indicators},
                upsert=True
            )
            print("Successfully cached US macro indicators to MongoDB.")
        except Exception as e:
            print(f"Error caching US macro indicators: {e}")

    return indicators


def fetch_crude_oil_price():
    """
    Fetches crude oil (WTI/Brent) price from webgia.com/gia-xang-dau/dau-tho/
    Returns a dict with price, change, change_pct, unit, time, and gold_impact rule.
    """
    from bs4 import BeautifulSoup
    import re

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8"
    }

    result = {
        "wti": {
            "name": "Dầu Thô WTI (Crude Oil WTI)",
            "price": None,
            "change": None,
            "change_pct": None,
            "unit": "USD/thùng",
            "time": None,
            "source": "webgia.com",
            "source_url": "https://webgia.com/gia-xang-dau/dau-tho/"
        },
        "brent": {
            "name": "Dầu Thô Brent",
            "price": None,
            "change": None,
            "change_pct": None,
            "unit": "USD/thùng",
            "time": None,
            "source": "webgia.com",
            "source_url": "https://webgia.com/gia-xang-dau/dau-tho/"
        },
        "gold_oil_rule": (
            "Dầu tăng ➔ Lạm phát tăng ➔ Vàng tăng (thuận chiều ngắn hạn). "
            "Dầu tăng mạnh gây suy thoái ➔ FED tăng lãi suất ➔ Vàng giảm (nghịch chiều dài hạn). "
            "Tương quan dầu-vàng thường dương (~0.6–0.8) trong dài hạn."
        ),
        "summary": "Dầu & Vàng thường tăng giảm cùng chiều do cùng bị tác động bởi lạm phát và địa chính trị."
    }

    try:
        r = requests.get(
            "https://webgia.com/gia-xang-dau/dau-tho/",
            headers=headers, timeout=12
        )
        if r.status_code != 200:
            print(f"webgia.com returned status {r.status_code}")
            return result

        soup = BeautifulSoup(r.text, "html.parser")

        # Try to find price boxes or tables - webgia.com typically has .price-box or table rows
        # Look for WTI and Brent rows
        text_content = soup.get_text(" ", strip=True)

        # Strategy 1: Look for table rows with "WTI" and "Brent"
        rows = soup.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            cell_texts = [c.get_text(strip=True) for c in cells]
            row_text = " ".join(cell_texts)

            if "WTI" in row_text or "wti" in row_text.lower():
                # Look for price number pattern like "76.50" or "76,50"
                prices = re.findall(r"\b(\d{2,3}[\.,]\d{1,2})\b", row_text)
                if prices:
                    result["wti"]["price"] = float(prices[0].replace(",", "."))
                changes = re.findall(r"([+-]?\d+[\.,]\d+)\s*%?", row_text)
                if len(changes) > 1:
                    result["wti"]["change"] = changes[1].replace(",", ".")

            if "Brent" in row_text or "brent" in row_text.lower():
                prices = re.findall(r"\b(\d{2,3}[\.,]\d{1,2})\b", row_text)
                if prices:
                    result["brent"]["price"] = float(prices[0].replace(",", "."))
                changes = re.findall(r"([+-]?\d+[\.,]\d+)\s*%?", row_text)
                if len(changes) > 1:
                    result["brent"]["change"] = changes[1].replace(",", ".")

        # Strategy 2: Look for div/span with class containing "price" near "WTI" / "Brent"
        if result["wti"]["price"] is None:
            all_text = soup.get_text()
            wti_match = re.search(r"WTI[^\d]{0,30}(\d{2,3}[.,]\d{1,2})", all_text, re.IGNORECASE)
            if wti_match:
                result["wti"]["price"] = float(wti_match.group(1).replace(",", "."))

            brent_match = re.search(r"Brent[^\d]{0,30}(\d{2,3}[.,]\d{1,2})", all_text, re.IGNORECASE)
            if brent_match:
                result["brent"]["price"] = float(brent_match.group(1).replace(",", "."))

        # Strategy 3: find any element with price-like classes
        price_els = soup.find_all(class_=re.compile(r"price|gia|cost", re.IGNORECASE))
        if result["wti"]["price"] is None and price_els:
            for el in price_els[:10]:
                txt = el.get_text(strip=True)
                m = re.search(r"(\d{2,3}[.,]\d{1,2})", txt)
                if m:
                    val = float(m.group(1).replace(",", "."))
                    if 50 <= val <= 200:  # Plausible crude oil price range
                        result["wti"]["price"] = val
                        break

        # Get timestamp
        time_el = soup.find(string=re.compile(r"Cập nhật|Update|updated", re.IGNORECASE))
        if time_el:
            result["wti"]["time"] = str(time_el)[:30].strip()
            result["brent"]["time"] = result["wti"]["time"]

        print(f"Crude oil fetched: WTI={result['wti']['price']}, Brent={result['brent']['price']}")

    except Exception as e:
        print(f"Error fetching crude oil price: {e}")

    return result

def fetch_geopolitical_conflicts():
    """
    Fetches geopolitical conflict and war news from VnExpress RSS (World category)
    and filters/categorizes them by country entities and conflict severity.
    """
    import xml.etree.ElementTree as ET
    from bs4 import BeautifulSoup
    import re

    url = "https://vnexpress.net/rss/the-gioi.rss"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # Conflict/Geopolitical keywords (including war and diplomatic negotiations/agreements)
    keywords = [
        "chiến tranh", "xung đột", "giao tranh", "quân sự", "tên lửa", 
        "tấn công", "oanh tạc", "bắn phá", "đột kích", "bất ổn", "căng thẳng", 
        "leo thang", "đối đầu", "răn đe", "vũ khí", "binh sĩ", "quân đội", "lực lượng",
        "chiến sự", "pháo kích", "không kích", "nổ súng", "thương vong", "giao chiến",
        "thỏa thuận", "đàm phán", "ngoại giao", "hòa bình", "hạt nhân", "trừng phạt", 
        "cấm vận", "ngừng bắn", "ký kết", "thỏa hiệp"
    ]
    
    # Target actors / countries specified by the user
    # Mỹ, Iran, Israel, Nga, Ukraine, Lebanon, Trung Quốc
    actors_mapping = [
        ({"nga", "moscow", "kremlin", "russia"}, "Nga"),
        ({"ukraine", "kiev"}, "Ukraine"),
        ({"israel", "tel aviv", "gaza", "hamas"}, "Israel"),
        ({"lebanon", "li-băng", "liban", "beirut", "hezbollah"}, "Lebanon"),
        ({"iran", "tehran"}, "Iran"),
        ({"mỹ", "washington", "pentagon", "lầu năm góc", "nhà trắng", "us", "usa"}, "Mỹ"),
        ({"trung quốc", "bắc kinh", "china", "đài loan", "taiwan"}, "Trung Quốc")
    ]
    
    # Party group classification for visual labels
    party_mapping = [
        ({"nga", "ukraine", "kiev", "moscow", "kremlin", "russia"}, "Nga - Ukraine"),
        ({"israel", "gaza", "hamas", "hezbollah", "lebanon", "li-băng", "liban", "iran", "tel aviv", "tehran", "beirut"}, "Trung Đông (Israel - Iran/Hezbollah)"),
        ({"mỹ", "trung quốc", "taiwan", "đài loan", "philippines", "biển đông", "bắc kinh", "washington", "china"}, "Mỹ - Trung Quốc / Biển Đông")
    ]

    conflicts = []
    try:
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code == 200:
            root = ET.fromstring(r.content)
            items = root.findall(".//item")
            
            for item in items:
                title = item.find("title").text if item.find("title") is not None else ""
                desc_html = item.find("description").text if item.find("description") is not None else ""
                link = item.find("link").text if item.find("link") is not None else ""
                pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
                
                # Clean description
                desc_soup = BeautifulSoup(desc_html, "html.parser")
                description = desc_soup.text.strip()
                
                # Check keywords in title and description
                full_text = (title + " " + description).lower()
                
                # Must be a conflict/war-related news
                is_conflict = any(kw in full_text for kw in keywords)
                if not is_conflict:
                    continue
                
                # Must mention at least one of the target countries or related entities
                mentioned_actors = []
                for keywords_set, name in actors_mapping:
                    if any(w in full_text for w in keywords_set):
                        mentioned_actors.append(name)
                
                if not mentioned_actors:
                    continue
                    
                # Detect parties group label
                detected_parties = "Địa chính trị / Quốc tế"
                for words, label in party_mapping:
                    if any(w in full_text for w in words):
                        detected_parties = label
                        break
                
                # If it's a general international conflict but matches specific actors, construct the parties text
                if detected_parties == "Địa chính trị / Quốc tế" and mentioned_actors:
                    detected_parties = " - ".join(mentioned_actors)
                
                # Exclude obvious jokes/fakes
                if "người ngoài hành tinh" in full_text:
                    continue
                
                # Status categorization
                status = "Căng thẳng"
                if any(w in full_text for w in ["tên lửa", "tấn công", "oanh tạc", "bắn phá", "đột kích", "giao tranh", "pháo kích", "không kích"]):
                    status = "Giao tranh quân sự"
                elif any(w in full_text for w in ["răn đe", "tập trận", "cảnh báo", "leo thang"]):
                    status = "Leo thang căng thẳng"
                elif any(w in full_text for w in ["đàm phán", "thỏa thuận", "ngừng bắn", "hòa bình"]):
                    status = "Đàm phán hòa bình"
                elif any(w in full_text for w in ["trừng phạt", "cấm vận"]):
                    status = "Trừng phạt ngoại giao"
                
                # Format date
                time_str = pub_date
                try:
                    parts = pub_date.split()
                    if len(parts) >= 5:
                        day = parts[1]
                        month = parts[2]
                        time = parts[4][:5] # HH:MM
                        month_num = {
                            "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
                            "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
                        }.get(month, "06")
                        time_str = f"{time} {day}/{month_num}"
                except:
                    pass
                    
                conflicts.append({
                    "title": title,
                    "description": description,
                    "link": link,
                    "time": time_str,
                    "parties": detected_parties,
                    "status": status
                })
    except Exception as e:
        print(f"Error fetching geopolitical conflicts: {e}")
        
    return conflicts[:6]


def get_gold_ticks():
    """
    Returns the gold intraday ticks from the last 24 hours.
    If the collection has fewer than 10 records in the last 24 hours, seeds it with historical daily prices with variations.
    """
    from datetime import datetime, timedelta
    import random
    
    init_db()
    one_week_ago = datetime.now() - timedelta(days=7)
    
    # Check if the oldest tick within the last 7 days is older than 6.5 days ago.
    # If not, it means our 7-day timeline is incomplete, so we backfill.
    oldest_recent_tick = db.gold_ticks.find_one({"timestamp": {"$gte": one_week_ago}}, sort=[("timestamp", 1)])
    if not oldest_recent_tick or oldest_recent_tick["timestamp"] > (datetime.now() - timedelta(days=6.5)):
        print("Oldest tick in the last 7 days is missing or too recent. Backfilling 7 days of ticks from daily history...")
        history_cursor = db.gold_history.find().sort("date", -1).limit(7)
        history_list = list(history_cursor)
        history_list.reverse() # Chronological order
        
        if history_list:
            now = datetime.now()
            for h_day in history_list:
                try:
                    day_date = datetime.strptime(h_day["date"], "%Y-%m-%d")
                except Exception:
                    continue
                
                # Create ticks every 2 hours for this day
                for hour in range(0, 24, 2):
                    tick_time = day_date.replace(hour=hour, minute=0, second=0)
                    if tick_time < one_week_ago or tick_time > (now - timedelta(hours=12)):
                        continue
                    
                    if db.gold_ticks.find_one({"timestamp": tick_time}):
                        continue
                        
                    var_pct = random.uniform(-0.003, 0.003)
                    db.gold_ticks.insert_one({
                        "timestamp": tick_time,
                        "sjc_bar_buy": round(h_day.get("sjc_bar_buy", 145000000.0) * (1 + var_pct - 0.002), -5),
                        "sjc_bar_sell": round(h_day.get("sjc_bar_sell", 148000000.0) * (1 + var_pct), -5),
                        "sjc_ring_buy": round(h_day.get("sjc_ring_buy", 145000000.0) * (1 + var_pct - 0.002), -5),
                        "sjc_ring_sell": round(h_day.get("sjc_ring_sell", 148000000.0) * (1 + var_pct), -5),
                        "world_price": round(h_day.get("world_price", 2450.0) * (1 + var_pct), 2),
                        "world_price_vnd": round(h_day.get("world_price_vnd", 131500000.0) * (1 + var_pct), -5)
                    })
            
    # Now query the last 7 days of ticks
    cursor = db.gold_ticks.find({"timestamp": {"$gte": one_week_ago}}, {"_id": 0}).sort("timestamp", 1)
    
    raw_ticks = list(cursor)
    total_raw = len(raw_ticks)
    
    # Downsample to maximum 150 points for rendering performance
    step = max(1, total_raw // 150)
    
    ticks_list = []
    seen_hours = set()
    for t in raw_ticks:
        hour_key = t["timestamp"].strftime("%Y-%m-%d %H")
        if hour_key not in seen_hours:
            seen_hours.add(hour_key)
            t_copy = t.copy()
            t_copy["timestamp"] = t["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            t_copy["time"] = t["timestamp"].strftime("%d/%m")
            ticks_list.append(t_copy)
        
    return ticks_list


def generate_prediction_explanation_for_date(row_curr, row_prev, pred_trend):
    drivers = []
    
    # 1. Technical Drivers
    if row_curr.get("rsi") is not None:
        rsi = row_curr["rsi"]
        if rsi <= 35:
            drivers.append("RSI quá bán hỗ trợ hồi phục")
        elif rsi >= 65:
            drivers.append("RSI quá mua áp lực giảm")
            
    if row_curr.get("macd") is not None and row_curr.get("macd_signal") is not None:
        macd = row_curr["macd"]
        sig = row_curr["macd_signal"]
        if macd > sig:
            drivers.append("Động lượng MACD tăng tích cực")
        else:
            drivers.append("Áp lực điều chỉnh theo MACD")
            
    if row_curr.get("sma_5") is not None and row_curr.get("sma_20") is not None:
        sma5 = row_curr["sma_5"]
        sma20 = row_curr["sma_20"]
        if sma5 > sma20:
            drivers.append("SMA5 nằm trên SMA20 (Tăng ngắn hạn)")
        else:
            drivers.append("SMA5 nằm dưới SMA20 (Giảm ngắn hạn)")

    # 2. Macro Drivers (changes vs previous day)
    if row_prev:
        if row_curr.get("dxy") is not None and row_prev.get("dxy") is not None:
            dxy_diff = row_curr["dxy"] - row_prev["dxy"]
            if dxy_diff > 0.05:
                drivers.append(f"DXY tăng ({dxy_diff:+.2f}đ) áp lực giảm")
            elif dxy_diff < -0.05:
                drivers.append(f"DXY giảm ({dxy_diff:+.2f}đ) hỗ trợ tăng")
                
        if row_curr.get("us10y") is not None and row_prev.get("us10y") is not None:
            us10y_diff = row_curr["us10y"] - row_prev["us10y"]
            if us10y_diff > 0.01:
                drivers.append(f"Lợi suất US10Y tăng ({us10y_diff:+.2f}%) sức ép giảm")
            elif us10y_diff < -0.01:
                drivers.append(f"Lợi suất US10Y giảm ({us10y_diff:+.2f}%) hỗ trợ tăng")
                
        if row_curr.get("brent") is not None and row_prev.get("brent") is not None:
            brent_diff = row_curr["brent"] - row_prev["brent"]
            if brent_diff > 0.3:
                drivers.append("Brent tăng hỗ trợ lạm phát")
            elif brent_diff < -0.3:
                drivers.append("Brent giảm bớt áp lực lạm phát")

        if row_curr.get("vix") is not None and row_prev.get("vix") is not None:
            vix_diff = row_curr["vix"] - row_prev["vix"]
            if vix_diff > 0.5:
                drivers.append("Chỉ số VIX tăng kích thích trú ẩn")

    # Filter/sort based on predicted trend
    pos_keywords = ["hỗ trợ", "tăng", "thúc đẩy", "tích cực", "kích thích"]
    neg_keywords = ["sức ép", "giảm", "áp lực", "gây", "điều chỉnh"]
    
    if pred_trend == "up":
        selected = [d for d in drivers if any(kw in d for kw in pos_keywords)]
        if not selected:
            selected = [d for d in drivers if not any(kw in d for kw in neg_keywords)]
    elif pred_trend == "down":
        selected = [d for d in drivers if any(kw in d for kw in neg_keywords)]
        if not selected:
            selected = [d for d in drivers if not any(kw in d for kw in pos_keywords)]
    else:
        selected = drivers
        
    if not selected:
        selected = ["Tín hiệu kỹ thuật & vĩ mô đi ngang."]
    else:
        selected = selected[:2]
        
    return " | ".join(selected)

def get_gold_prediction_history_with_explanations():
    cursor = db.gold_predictions_history.find({}, {"_id": 0}).sort("date", -1).limit(90)
    history = list(cursor)
    history.reverse()
    if not history:
        return []
        
    try:
        gold_cursor = db.gold_history.find({}, {"_id": 0}).sort("date", 1)
        gold_list = list(gold_cursor)
        if len(gold_list) >= 20:
            import pandas as pd
            import numpy as np
            import predictor
            
            df = pd.DataFrame(gold_list)
            df["world_price"] = df["world_price"].replace(0.0, np.nan).ffill().bfill()
            df["close"] = df["world_price"]
            df["open"] = df["world_price"]
            df["high"] = df["world_price"]
            df["low"] = df["world_price"]
            df["volume"] = 1.0
            
            macro_cursor = db.macro_history.find({}, {"_id": 0})
            macro_df = pd.DataFrame(list(macro_cursor))
            if not macro_df.empty and "date" in macro_df.columns:
                df = pd.merge(df, macro_df, on="date", how="left")
                cols = [c for c in ["dxy", "us10y", "vix", "brent", "dji", "spx", "eurusd", "xagusd", "real_yield"] if c in df.columns]
                df[cols] = df[cols].ffill().bfill()
                
            df_ind = predictor.calculate_technical_indicators(df)
            
            lookup = {}
            for idx, row in df_ind.iterrows():
                prev_row = df_ind.iloc[idx - 1] if idx > 0 else None
                lookup[row["date"]] = {
                    "row_curr": row.to_dict(),
                    "row_prev": prev_row.to_dict() if prev_row is not None else None
                }
                
            for idx_h, item in enumerate(history):
                date = item["date"]
                item["explanations"] = {}
                
                # To explain the prediction for `date`, we look at the state of `prev_item.date`
                if idx_h > 0:
                    prev_item = history[idx_h - 1]
                    base_date = prev_item["date"]
                    base_price = prev_item.get("actual_price")
                    
                    if base_date in lookup and base_price is not None and base_price > 0:
                        state = lookup[base_date]
                        row_curr = state["row_curr"]
                        row_prev = state["row_prev"]
                        
                        for model_key, pred_price in item.get("models", {}).items():
                            pred_trend = "up" if pred_price > base_price else ("down" if pred_price < base_price else "flat")
                            item["explanations"][model_key] = generate_prediction_explanation_for_date(row_curr, row_prev, pred_trend)
                
                if not item["explanations"]:
                    item["explanations"] = {m: "Tín hiệu kỹ thuật & vĩ mô đi ngang." for m in ["random_forest", "linear_regression", "mlp", "xgboost", "lstm", "ensemble"]}
    except Exception as e:
        print(f"Error generating explanations for prediction history: {e}")
        for item in history:
            if "explanations" not in item:
                item["explanations"] = {m: "Không có phân tích." for m in ["random_forest", "linear_regression", "mlp", "xgboost", "lstm", "ensemble"]}
                
    return history


