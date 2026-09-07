import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import re
import threading
from pymongo import MongoClient, UpdateOne
from zoneinfo import ZoneInfo

# Initialize MongoDB client
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client["stock_analytics"]
_calendar_results_lock = threading.Lock()


def _prepare_gold_history_frame(rows):
    """Return chronological trading-day Gold observations for model training."""
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if frame.empty or "date" not in frame.columns or "world_price" not in frame.columns:
        return frame
    parsed_dates = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.assign(_parsed_date=parsed_dates)
    frame["world_price"] = pd.to_numeric(frame["world_price"], errors="coerce")
    frame = frame.dropna(subset=["_parsed_date", "world_price"])
    frame = frame[frame["world_price"] > 0]
    frame = frame[frame["_parsed_date"].dt.weekday < 5]
    frame = frame.sort_values("_parsed_date").drop_duplicates(subset=["date"], keep="last")
    return frame.drop(columns=["_parsed_date"]).reset_index(drop=True)

def init_db():
    """
    Initializes the MongoDB indexes and seeds the gold data used by the app.
    """
    try:
        # Create unique indexes for the gold and macro collections.
        db.gold_history.create_index([("date", 1)], unique=True)
        db.us_economic_calendar.create_index([("date", 1), ("event", 1)], unique=True)
        db.macro_history.create_index([("date", 1)], unique=True)
        db.background_activity_history.create_index([("started_at", -1)])
        db.background_activity_history.create_index([("category", 1), ("started_at", -1)])
        db.gold_backtest_results.create_index([("year", 1), ("month", 1)], unique=True)

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
    market_signals = collect_gold_market_signals(news)
    calendar = get_us_economic_calendar()
    macro_indicators = fetch_us_macro_indicators()
    try:
        macro_snapshot = list(
            db.macro_history.find({}, {"_id": 0})
            .sort("date", -1)
            .limit(2)
        )
    except Exception as macro_snapshot_error:
        print(f"Error loading macro snapshot for FED outlook: {macro_snapshot_error}")
        macro_snapshot = []
    fed_policy_outlook = build_fed_policy_outlook(
        macro_indicators=macro_indicators,
        macro_history=macro_snapshot,
        calendar=calendar,
        market_signals=market_signals,
    )
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
        "market_signals": market_signals,
        "calendar": calendar,
        "macro_indicators": macro_indicators,
        "fed_policy_outlook": fed_policy_outlook,
        "crude_oil": crude_oil,
        "conflict_events": conflict_events
    }

    # A transient RSS/FRED/commodity-source failure must not erase the last
    # usable news, calendar, macro or price section from the shared cache.
    # Keep the fresh value when it exists and retain the previous value only
    # for an empty section.
    try:
        previous_cache = db.gold_prices.find_one({"_id": "latest_prices"})
        if previous_cache:
            for key in ("news", "market_signals", "calendar", "conflict_events", "fed_policy_outlook"):
                if not gold_data.get(key) and previous_cache.get(key):
                    gold_data[key] = previous_cache[key]
            for parent, children in {
                "world": ("price", "converted_ounce", "converted_cay"),
                "domestic": ("sjc_bar", "sjc_ring", "brands"),
            }.items():
                previous_parent = previous_cache.get(parent, {})
                current_parent = gold_data.get(parent, {})
                for child in children:
                    if not current_parent.get(child) and previous_parent.get(child):
                        current_parent[child] = previous_parent[child]
    except Exception as cache_merge_err:
        print(f"Error retaining stale dashboard sections: {cache_merge_err}")

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
            
            # 3. Save to intraday ticks immediately. Full model training and
            # history backfill are intentionally not run in this price crawler:
            # they used to block the worker before the tick insert completed,
            # leaving the 24-hour chart stale for many hours.
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
        close_prices = close_prices.ffill()
        
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


PCE_RELEASES_2026 = [
    ("2026-01-22", "2025-11"),
    ("2026-02-20", "2025-12"),
    ("2026-03-13", "2026-01"),
    ("2026-04-09", "2026-02"),
    ("2026-04-30", "2026-03"),
    ("2026-05-28", "2026-04"),
    ("2026-06-25", "2026-05"),
    ("2026-07-30", "2026-06"),
    ("2026-08-26", "2026-07"),
    ("2026-09-30", "2026-08"),
    ("2026-10-29", "2026-09"),
    ("2026-11-25", "2026-10"),
    ("2026-12-23", "2026-11"),
]


# Policy events are kept separately from the FOMC rate-decision dates.  A
# speech can change the expected rate path even when no target range changes.
# The event is sourced from the Federal Reserve's public event/speech page and
# is intentionally represented as a FED event so the existing calendar and
# `days_to_fed` feature can consume it.
FED_POLICY_EVENTS_2026 = [
    {
        "date": "2026-08-28",
        "event": "Phát biểu Chủ tịch FED Kevin Warsh tại Jackson Hole",
        "category": "FED",
        "event_type": "FED_SPEECH",
        "importance": "Rất cao",
        "impact": (
            "Phát biểu về lạm phát và định hướng lãi suất có thể làm thay đổi "
            "kỳ vọng lợi suất/DXY, từ đó tác động trực tiếp đến vàng."
        ),
        "policy_tone": "hawkish",
        "policy_tone_label": "Hawkish · gây áp lực giảm vàng ngắn hạn",
        "source": "Federal Reserve",
        "source_url": "https://www.federalreserve.gov/newsevents/speech/warsh20260828a.htm",
    },
]


FED_FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
BEA_RELEASE_SCHEDULE_URL = "https://www.bea.gov/news/schedule"
BLS_RELEASE_SCHEDULE_URL = "https://www.bls.gov/schedule/{year}/"
BLS_ICS_SCHEDULE_URL = "https://www.bls.gov/schedule/news_release/bls.ics"
CALENDAR_SCHEDULE_TTL = timedelta(hours=6)


def _calendar_source_headers():
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }


def _month_number(month_name):
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    return months.get(str(month_name or "").strip().lower())


def _parse_calendar_date(value, year=None):
    """Parse common English official-calendar date labels into YYYY-MM-DD."""
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return None
    match = re.search(
        r"(?P<month>January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+(?P<day>\d{1,2})"
        r"(?:,?\s+(?P<year>\d{4}))?",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    parsed_year = int(match.group("year") or year or datetime.now().year)
    month = _month_number(match.group("month"))
    day = int(match.group("day"))
    try:
        return datetime(parsed_year, month, day).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def _parse_reference_period(title):
    """Return (month, year) from official release titles such as 'August 2026'."""
    match = re.search(
        r"(?:for|,|\()\s*(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+(\d{4})",
        str(title or ""),
        re.IGNORECASE,
    )
    if not match:
        return None
    month = _month_number(match.group(1))
    return (month, int(match.group(2))) if month else None


def _crawl_fomc_meeting_events(years=None):
    """Crawl the official FOMC meeting calendar (including future years)."""
    from bs4 import BeautifulSoup

    requested_years = {int(year) for year in (years or [])}
    response = requests.get(
        FED_FOMC_CALENDAR_URL,
        headers=_calendar_source_headers(),
        timeout=20,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    events = []
    for panel in soup.select("#article .panel.panel-default"):
        heading = panel.select_one(".panel-heading h4")
        year_match = re.search(r"(20\d{2})", heading.get_text(" ", strip=True) if heading else "")
        if not year_match:
            continue
        year = int(year_match.group(1))
        if requested_years and year not in requested_years:
            continue

        for row in panel.select(".fomc-meeting"):
            month_node = row.select_one(".fomc-meeting__month")
            date_node = row.select_one(".fomc-meeting__date")
            if not month_node or not date_node:
                continue
            month = _month_number(month_node.get_text(" ", strip=True))
            raw_range = date_node.get_text(" ", strip=True)
            range_match = re.search(r"(\d{1,2})\s*-\s*(\d{1,2})", raw_range)
            if not month or not range_match:
                continue
            start_day, end_day = map(int, range_match.groups())
            try:
                meeting_start = datetime(year, month, start_day)
                meeting_end = datetime(year, month, end_day)
            except ValueError:
                continue

            statement_link = row.select_one(
                'a[href*="/newsevents/pressreleases/monetary"]'
            )
            press_conference_link = row.select_one(
                'a[href*="/monetarypolicy/fomcpresconf"]'
            )
            source_url = FED_FOMC_CALENDAR_URL
            statement_url = ""
            if statement_link and statement_link.get("href"):
                statement_url = requests.compat.urljoin(FED_FOMC_CALENDAR_URL, statement_link["href"])
                source_url = statement_url

            decision_date = meeting_end.strftime("%Y-%m-%d")
            start_date = meeting_start.strftime("%Y-%m-%d")
            end_date = meeting_end.strftime("%Y-%m-%d")
            projection = "*" in raw_range
            events.append({
                "date": decision_date,
                "event": (
                    f"Cuộc họp FOMC & quyết định lãi suất FED "
                    f"({start_day:02d}–{end_day:02d}/{month:02d}/{year})"
                ),
                "category": "FED",
                "event_type": "FOMC_MEETING",
                "event_key": f"FOMC:{decision_date}",
                "importance": "Rất cao",
                "impact": (
                    "Cuộc họp FOMC, tuyên bố chính sách và cuộc họp báo có thể "
                    "làm thay đổi kỳ vọng lãi suất, USD/lợi suất và giá vàng."
                ),
                "meeting_start_date": start_date,
                "meeting_end_date": end_date,
                "fomc_projection": projection,
                "source": "Federal Reserve · FOMC",
                "source_url": source_url,
                "statement_url": statement_url,
                "press_conference_url": (
                    requests.compat.urljoin(FED_FOMC_CALENDAR_URL, press_conference_link["href"])
                    if press_conference_link and press_conference_link.get("href")
                    else ""
                ),
                "schedule_crawled_at": datetime.now().isoformat(timespec="seconds"),
            })
    return events


def _crawl_bea_release_events(years=None):
    """Crawl BEA releases, especially PCE, GDP, income and trade."""
    from bs4 import BeautifulSoup

    requested_years = {int(year) for year in (years or [])}
    response = requests.get(
        BEA_RELEASE_SCHEDULE_URL,
        headers=_calendar_source_headers(),
        timeout=20,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    events = []
    for table in soup.select("table"):
        header = table.select_one("th")
        year_match = re.search(r"(20\d{2})", header.get_text(" ", strip=True) if header else "")
        year = int(year_match.group(1)) if year_match else datetime.now().year
        if requested_years and year not in requested_years:
            continue
        for row in table.select("tr"):
            date_node = row.select_one(".release-date")
            title_node = row.select_one(".release-title")
            if not date_node or not title_node:
                continue
            title = re.sub(r"\s+", " ", title_node.get_text(" ", strip=True))
            date = _parse_calendar_date(date_node.get_text(" ", strip=True), year=year)
            if not date or not title:
                continue
            title_lower = title.lower()
            if "personal income and outlays" in title_lower:
                category = "PCE"
                period = _parse_reference_period(title)
                if period:
                    month, reference_year = period
                    event_name = f"Công bố PCE & Core PCE Mỹ (Tháng {month:02d}/{reference_year})"
                else:
                    event_name = f"BEA · {title}"
                event_type = "BEA_PCE_RELEASE"
                impact = (
                    "PCE/Core PCE cao hơn dự báo → kỳ vọng lãi suất cao lâu hơn → "
                    "USD/lợi suất tăng → Vàng chịu áp lực giảm ngắn hạn."
                )
            elif "gdp" in title_lower or "corporate profits" in title_lower:
                category, event_type = "GDP", "BEA_GDP_RELEASE"
                event_name = f"Công bố GDP Mỹ · {title}"
                impact = "GDP tốt hơn kỳ vọng có thể củng cố USD/lợi suất và gây áp lực lên vàng."
            elif "trade" in title_lower or "international transactions" in title_lower:
                category, event_type = "Thương mại", "BEA_TRADE_RELEASE"
                event_name = f"Cán cân thương mại Mỹ · {title}"
                impact = "Dữ liệu thương mại làm thay đổi đánh giá tăng trưởng và nhu cầu USD ngắn hạn."
            else:
                continue
            time_node = date_node.select_one("small")
            release_time = time_node.get_text(" ", strip=True) if time_node else ""
            events.append({
                "date": date,
                "event": event_name,
                "category": category,
                "event_type": event_type,
                "event_key": f"{event_type}:{date}:{title}",
                "importance": "Rất cao" if category == "PCE" else "Cao",
                "impact": impact,
                "release_time": release_time,
                "source_timezone": "America/New_York",
                "source": "U.S. Bureau of Economic Analysis (BEA)",
                "source_url": BEA_RELEASE_SCHEDULE_URL,
                "source_title": title,
                "schedule_crawled_at": datetime.now().isoformat(timespec="seconds"),
            })
    return events


def _ics_unescape(value):
    return (
        str(value or "")
        .replace("\\n", " ")
        .replace("\\N", " ")
        .replace("\\,", ",")
        .replace("\\;", ";")
        .replace("\\\\", "\\")
    )


def _parse_ics_datetime(value):
    """Parse an ICS DTSTART into (YYYY-MM-DD, HH:MM AM/PM)."""
    raw = str(value or "").strip()
    match = re.search(r"(20\d{2})(\d{2})(\d{2})(?:T(\d{2})(\d{2}))?", raw)
    if not match:
        return None, ""
    year, month, day = (int(match.group(i)) for i in (1, 2, 3))
    try:
        date = datetime(year, month, day).strftime("%Y-%m-%d")
    except ValueError:
        return None, ""
    hour, minute = match.group(4), match.group(5)
    release_time = ""
    if hour is not None and minute is not None:
        release_time = datetime(2000, 1, 1, int(hour), int(minute)).strftime("%I:%M %p")
    return date, release_time


def _build_bls_calendar_event(date, title, release_time="", source_url=None):
    """Normalize one BLS schedule row into the calendar schema."""
    title = re.sub(r"\s+", " ", _ics_unescape(title).strip())
    if not date or not title:
        return None
    title_lower = title.lower()
    if "employment situation" in title_lower:
        category, event_type = "Việc làm", "BLS_NFP_RELEASE"
        period = _parse_reference_period(title)
        event_name = (
            f"Báo cáo việc làm phi nông nghiệp Mỹ (NFP Tháng {period[0]:02d}/{period[1]})"
            if period else f"BLS · {title}"
        )
        impact = "NFP mạnh thường củng cố USD/lợi suất và gây áp lực lên vàng; NFP yếu có tác động ngược lại."
    elif "consumer price index" in title_lower:
        category, event_type = "CPI", "BLS_CPI_RELEASE"
        period = _parse_reference_period(title)
        event_name = (
            f"Công bố chỉ số lạm phát CPI Mỹ (Tháng {period[0]:02d}/{period[1]})"
            if period else f"BLS · {title}"
        )
        impact = "CPI cao hơn kỳ vọng có thể làm FED giữ lãi suất cao lâu hơn và gây áp lực lên vàng."
    elif "producer price index" in title_lower:
        category, event_type = "PPI", "BLS_PPI_RELEASE"
        event_name = f"Công bố PPI Mỹ · {title}"
        impact = "PPI là tín hiệu sớm của áp lực giá, có thể làm thay đổi kỳ vọng chính sách FED."
    elif "job openings and labor turnover" in title_lower:
        category, event_type = "Việc làm", "BLS_JOLTS_RELEASE"
        event_name = f"Công bố JOLTS Mỹ · {title}"
        impact = "JOLTS phản ánh nhu cầu lao động và có thể làm thay đổi kỳ vọng lãi suất FED."
    elif "employment cost index" in title_lower or "employer costs for employee compensation" in title_lower:
        category, event_type = "Việc làm", "BLS_ECI_RELEASE"
        event_name = f"Công bố Employment Cost Index Mỹ · {title}"
        impact = "Chi phí lao động dai dẳng có thể giữ áp lực lạm phát và lợi suất ở mức cao."
    else:
        return None
    return {
        "date": date,
        "event": event_name,
        "category": category,
        "event_type": event_type,
        "event_key": f"{event_type}:{date}:{title}",
        "importance": "Rất cao" if category in ("CPI", "Việc làm") else "Cao",
        "impact": impact,
        "release_time": release_time,
        "source_timezone": "America/New_York",
        "source": "U.S. Bureau of Labor Statistics (BLS)",
        "source_url": source_url or BLS_RELEASE_SCHEDULE_URL.format(year=date[:4]),
        "source_title": title,
        "schedule_crawled_at": datetime.now().isoformat(timespec="seconds"),
    }


def _crawl_bls_ics_release_events(years=None):
    """Crawl BLS' official iCalendar feed for key labor/inflation releases."""
    requested_years = {int(year) for year in (years or [])}
    response = requests.get(
        BLS_ICS_SCHEDULE_URL,
        headers={**_calendar_source_headers(), "Accept": "text/calendar,*/*;q=0.8"},
        timeout=20,
    )
    response.raise_for_status()
    # RFC 5545 allows folded lines; continuation lines begin with a space/tab.
    lines = []
    for line in response.text.replace("\r\n", "\n").split("\n"):
        if line.startswith((" ", "\t")) and lines:
            lines[-1] += line[1:]
        else:
            lines.append(line)
    events = []
    current = None
    for line in lines:
        if line == "BEGIN:VEVENT":
            current = {}
            continue
        if line == "END:VEVENT":
            if current:
                date, release_time = _parse_ics_datetime(current.get("DTSTART"))
                if date and (not requested_years or int(date[:4]) in requested_years):
                    event = _build_bls_calendar_event(
                        date,
                        current.get("SUMMARY", ""),
                        release_time=release_time,
                        source_url=BLS_ICS_SCHEDULE_URL,
                    )
                    if event:
                        events.append(event)
            current = None
            continue
        if current is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        current[key.split(";", 1)[0].upper()] = _ics_unescape(value)
    return events


def _crawl_bls_release_events(years=None):
    """Crawl BLS key releases, trying official ICS before HTML schedules."""
    requested_years = sorted({int(year) for year in (years or [])}) or [datetime.now().year]
    try:
        ics_events = _crawl_bls_ics_release_events(requested_years)
        if ics_events:
            return ics_events
    except Exception as ics_error:
        print(f"BLS iCalendar crawl failed, trying HTML schedule: {ics_error}")

    from bs4 import BeautifulSoup

    events = []
    for year in requested_years:
        response = requests.get(
            BLS_RELEASE_SCHEDULE_URL.format(year=year),
            headers=_calendar_source_headers(),
            timeout=20,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        for row in soup.select("table tr"):
            cells = row.find_all(["th", "td"])
            if len(cells) < 3:
                continue
            date = _parse_calendar_date(cells[0].get_text(" ", strip=True), year=year)
            title = cells[-1].get_text(" ", strip=True)
            event = _build_bls_calendar_event(
                date,
                title,
                release_time=cells[1].get_text(" ", strip=True),
                source_url=BLS_RELEASE_SCHEDULE_URL.format(year=year),
            )
            if event:
                events.append(event)
    return events


def _upsert_crawled_calendar_events(events):
    """Merge crawled rows into the existing calendar without deleting seeded data."""
    updated = 0
    for event in events:
        date = event.get("date")
        category = event.get("category")
        event_type = event.get("event_type")
        if not date or not category or not event_type:
            continue

        # Give legacy seeded rows a type so the first crawl updates them in
        # place instead of creating a duplicate row for the same release.
        db.us_economic_calendar.update_many(
            {
                "date": date,
                "category": category,
                "event_type": {"$exists": False},
            },
            {"$set": {"event_type": event_type}},
        )
        result = db.us_economic_calendar.update_one(
            {"date": date, "category": category, "event_type": event_type},
            {"$set": event},
            upsert=True,
        )
        if result.modified_count or result.upserted_id:
            updated += 1
    return updated


def crawl_official_us_economic_calendar(force=False):
    """Refresh official schedules for FED, BLS and BEA with safe fallbacks."""
    now = datetime.now()
    try:
        state = db.calendar_schedule_sync_state.find_one({"_id": "official_schedule"}) or {}
        last_success = state.get("last_success_at")
        if not force and isinstance(last_success, datetime) and now - last_success < CALENDAR_SCHEDULE_TTL:
            return {
                "status": "skipped",
                "updated": 0,
                "sources": state.get("sources", []),
                "source_errors": state.get("source_errors", []),
                "last_success_at": last_success.isoformat(timespec="seconds"),
            }

        years = [now.year, now.year + 1]
        crawled_events = []
        source_errors = []
        source_counts = {}
        for source_name, crawler_fn in (
            ("Federal Reserve FOMC", _crawl_fomc_meeting_events),
            ("BEA", _crawl_bea_release_events),
            ("BLS", _crawl_bls_release_events),
        ):
            try:
                source_events = crawler_fn(years=years)
                crawled_events.extend(source_events)
                source_counts[source_name] = len(source_events)
            except Exception as error:
                source_counts[source_name] = 0
                source_errors.append(f"{source_name}: {error}")
                print(f"Economic calendar crawl failed for {source_name}: {error}")

        updated = _upsert_crawled_calendar_events(crawled_events)
        sync_doc = {
            "last_attempt_at": now,
            "last_success_at": now,
            "sources": source_counts,
            "source_errors": source_errors,
            "updated": updated,
        }
        db.calendar_schedule_sync_state.update_one(
            {"_id": "official_schedule"},
            {"$set": sync_doc},
            upsert=True,
        )
        return {
            "status": "partial" if source_errors else "success",
            "updated": updated,
            "sources": source_counts,
            "source_errors": source_errors,
            "last_success_at": now.isoformat(timespec="seconds"),
        }
    except Exception as error:
        print(f"Error crawling official US economic calendar: {error}")
        return {"status": "error", "updated": 0, "source_errors": [str(error)]}


def ensure_pce_calendar_events():
    """Upsert the official BEA Personal Income and Outlays release dates."""
    operations = []
    for release_date, reference_month in PCE_RELEASES_2026:
        year, month = reference_month.split("-")
        event = {
            "date": release_date,
            "event": f"Công bố PCE & Core PCE Mỹ (Tháng {month}/{year})",
            "category": "PCE",
            "importance": "Rất cao",
            "impact": (
                "PCE/Core PCE cao hơn dự báo → kỳ vọng lãi suất cao lâu hơn → "
                "USD/lợi suất tăng → Vàng chịu áp lực giảm ngắn hạn."
            ),
            "source": "U.S. Bureau of Economic Analysis (BEA)",
            "source_url": "https://www.bea.gov/news/schedule",
        }
        operations.append(UpdateOne(
            {"date": release_date, "event": event["event"]},
            {"$set": event},
            upsert=True,
        ))
    if operations:
        db.us_economic_calendar.bulk_write(operations)


def update_pce_history_from_fred(days=365):
    """Attach headline/core PCE to each macro row only after its release date.

    FRED indexes observations by the reference month, not the date on which the
    market learned the value.  Mapping observations to the official BEA
    release date prevents future PCE values from leaking into backtests.
    """
    try:
        ensure_pce_calendar_events()
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=max(int(days), 365) + 500)).strftime("%Y-%m-%d")
        headline = _fetch_fred_series("PCEPI", start_date, end_date)
        core = _fetch_fred_series("PCEPILFE", start_date, end_date)
        if not headline or not core:
            print("FRED returned empty PCE series.")
            return 0

        pce = pd.DataFrame({
            "pce_headline_index": pd.Series(headline),
            "pce_core_index": pd.Series(core),
        })
        pce.index = pd.to_datetime(pce.index, errors="coerce")
        pce = pce.dropna().sort_index()
        pce["pce_headline_mom"] = pce["pce_headline_index"].pct_change(1) * 100.0
        pce["pce_core_mom"] = pce["pce_core_index"].pct_change(1) * 100.0
        pce["pce_headline_yoy"] = pce["pce_headline_index"].pct_change(12) * 100.0
        pce["pce_core_yoy"] = pce["pce_core_index"].pct_change(12) * 100.0

        official_release_map = {
            reference_month: release_date
            for release_date, reference_month in PCE_RELEASES_2026
        }
        released_rows = []
        for period_date, row in pce.iterrows():
            reference_month = period_date.strftime("%Y-%m")
            release_date = official_release_map.get(reference_month)
            if not release_date:
                # Conservative fallback for historical months not covered by
                # the explicit calendar: expose the value at the second month
                # end, after the normal late-next-month release window.
                release_date = (period_date + pd.offsets.MonthEnd(2)).strftime("%Y-%m-%d")
            released_rows.append({
                "release_date": release_date,
                "pce_reference_month": reference_month,
                "pce_headline_yoy": row.get("pce_headline_yoy"),
                "pce_core_yoy": row.get("pce_core_yoy"),
                "pce_headline_mom": row.get("pce_headline_mom"),
                "pce_core_mom": row.get("pce_core_mom"),
            })

        releases = pd.DataFrame(released_rows)
        for col in ("pce_headline_yoy", "pce_core_yoy", "pce_headline_mom", "pce_core_mom"):
            releases[col] = pd.to_numeric(releases[col], errors="coerce")
        releases["_release_dt"] = pd.to_datetime(releases["release_date"], errors="coerce")
        releases = releases.dropna(subset=["_release_dt"]).sort_values("_release_dt")

        macro_dates = list(db.macro_history.find({}, {"_id": 0, "date": 1}).sort("date", 1))
        if not macro_dates:
            return 0
        macro_frame = pd.DataFrame(macro_dates)
        macro_frame["_date_dt"] = pd.to_datetime(macro_frame["date"], errors="coerce")
        macro_frame = macro_frame.dropna(subset=["_date_dt"]).sort_values("_date_dt")
        merged = pd.merge_asof(
            macro_frame,
            releases,
            left_on="_date_dt",
            right_on="_release_dt",
            direction="backward",
        )

        operations = []
        pce_fields = ("pce_headline_yoy", "pce_core_yoy", "pce_headline_mom", "pce_core_mom")
        for _, row in merged.iterrows():
            values = {}
            for field in pce_fields:
                value = row.get(field)
                if value is not None and pd.notna(value):
                    values[field] = float(value)
            if not values:
                continue
            values["pce_release_date"] = str(row.get("release_date"))
            values["pce_reference_month"] = str(row.get("pce_reference_month"))
            operations.append(UpdateOne(
                {"date": str(row["date"])},
                {"$set": values},
                upsert=False,
            ))

        if operations:
            result = db.macro_history.bulk_write(operations)
            print(f"FRED/BEA: Updated PCE features on {result.modified_count} macro rows.")
        return len(operations)
    except Exception as error:
        print(f"Error updating PCE history: {error}")
        return 0


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
    """Update Yahoo Finance series plus FRED real-yield and PCE features."""
    update_macro_history(days=days)
    update_real_yield_from_fred(days=days)
    update_pce_history_from_fred(days=days)


def calculate_days_until_events(dates_series):
    """
    Given a pandas Series of dates (in YYYY-MM-DD string format),
    queries the us_economic_calendar collection in MongoDB and returns
    four arrays containing the days until the next FED, CPI, NFP, and PCE
    release. PCE dates come from the official BEA schedule.
    """
    # 1. Ensure the PCE schedule exists, then fetch all events from DB.
    try:
        ensure_pce_calendar_events()
        ensure_fed_policy_events()
        crawl_official_us_economic_calendar()
    except Exception as pce_calendar_error:
        print(f"Error ensuring macro policy calendar events: {pce_calendar_error}")
    try:
        events = list(db.us_economic_calendar.find({}, {"_id": 0}))
    except Exception as e:
        print(f"Error querying economic calendar: {e}")
        events = []
        
    # Group event dates by category
    event_dates = {
        "FED": [],
        "CPI": [],
        "Việc làm": [],  # Non-Farm Payrolls category is 'Việc làm'
        "PCE": [],
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
    days_to_pce = [get_days_until_next(d, "PCE") for d in dates_series]
    
    return days_to_fed, days_to_cpi, days_to_nfp, days_to_pce

def calculate_dynamic_ensemble_weights(limit=20, return_diagnostics=False, model_version=None):
    """
    Computes adaptive weights for six base models: RF, Linear Regression, MLP,
    XGBoost, LSTM and CNN 1D.
    based on a COMBINED score of:
      - 50%: Historical MAPE (Mean Absolute Percentage Error) — accuracy of magnitude
      - 50%: Directional accuracy (% of times correct direction was predicted)
    Returns:
        dict: {model_key: weight} summing to 1.0.
    """
    try:
        query = {"actual_price": {"$ne": None, "$gt": 0.0}}
        if model_version:
            query["model_version"] = model_version
        cursor = db.gold_predictions_history.find(
            query,
            {"_id": 0, "date": 1, "actual_price": 1, "forecast_base_price": 1, "models": 1, "model_version": 1}
        ).sort("date", -1).limit(limit)
        
        history = list(cursor)
        if len(history) < 5:
            fallback = {
                "random_forest": 1 / 6,
                "linear_regression": 1 / 6,
                "mlp": 1 / 6,
                "xgboost": 1 / 6,
                "lstm": 1 / 6,
                "cnn": 1 / 6,
            }
            return (fallback, {}) if return_diagnostics else fallback

        # Sort ascending for direction comparison
        history_asc = list(reversed(history))
            
        model_errors = {
            "random_forest": [],
            "linear_regression": [],
            "mlp": [],
            "xgboost": [],
            "lstm": [],
            "cnn": []
        }
        model_dir_correct = {k: [] for k in model_errors}
        persistence_errors = []
        
        for i, doc in enumerate(history_asc):
            actual = doc.get("actual_price")
            models_preds = doc.get("models", {})

            # Prefer the exact base price saved with that forecast. Older
            # records fall back to the previous resolved close.
            prev_actual = doc.get("forecast_base_price")
            if not prev_actual and i > 0:
                prev_actual = history_asc[i - 1].get("actual_price")

            if actual and prev_actual and prev_actual > 0:
                persistence_errors.append(abs(actual - prev_actual) / actual)

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
                       else {k: 1.0 / len(mape_scores) for k in mape_scores}

        # Normalize directional accuracy into weights
        total_dir = sum(dir_accs.values())
        dir_weights = {k: v / total_dir for k, v in dir_accs.items()} if total_dir > 0 \
                      else {k: 1.0 / len(dir_accs) for k in dir_accs}

        # Combine 50% MAPE + 50% directional accuracy
        combined = {k: 0.5 * mape_weights[k] + 0.5 * dir_weights[k] for k in mape_weights}
        total_combined = sum(combined.values())
        weights = {k: float(v / total_combined) for k, v in combined.items()} if total_combined > 0 \
                  else {k: 1.0 / len(combined) for k in combined}

        persistence_mape = (
            float(np.mean(persistence_errors)) if len(persistence_errors) >= 3 else None
        )
        diagnostics = {}
        for model_key in model_errors:
            sample_count = len(model_errors[model_key])
            mape_skill = (
                1.0 - mapes[model_key] / max(persistence_mape, 1e-10)
                if persistence_mape is not None else 0.0
            )
            direction_skill = max(0.0, min(1.0, (dir_accs[model_key] - 0.5) * 2.0))
            sample_factor = min(1.0, sample_count / 30.0)
            reliability = sample_factor * (
                0.60 * max(0.0, min(1.0, mape_skill))
                + 0.40 * direction_skill
            )
            diagnostics[model_key] = {
                "samples": sample_count,
                "mape": float(mapes[model_key]),
                "persistence_mape": persistence_mape,
                "mape_skill_vs_persistence": float(mape_skill),
                "directional_accuracy": float(dir_accs[model_key]),
                "reliability": float(reliability),
            }

        print(f"Ensemble weights (MAPE+Dir): {' | '.join(f'{k}={v:.3f}' for k,v in weights.items())}")
        print(f"  Directional accuracy: {' | '.join(f'{k}={v:.1%}' for k,v in dir_accs.items())}")
        return (weights, diagnostics) if return_diagnostics else weights
    except Exception as e:
        print(f"Error calculating dynamic ensemble weights: {e}")
        fallback = {
            "random_forest": 1 / 6,
            "linear_regression": 1 / 6,
            "mlp": 1 / 6,
            "xgboost": 1 / 6,
            "lstm": 1 / 6,
            "cnn": 1 / 6
        }
        return (fallback, {}) if return_diagnostics else fallback


def calculate_gold_model_biases(limit=15, model_version=None):
    """
    Calculates the rolling bias (mean error = actual - pred) of the models
    over the last `limit` days to apply online feedback error correction (self-learning).
    """
    try:
        query = {"actual_price": {"$ne": None, "$gt": 0.0}}
        if model_version:
            query["model_version"] = model_version
        cursor = db.gold_predictions_history.find(
            query,
            {"_id": 0, "actual_price": 1, "models": 1, "model_version": 1}
        ).sort("date", -1).limit(limit)
        
        history = list(cursor)
        if len(history) < 3:
            return {
                "random_forest": 0.0,
                "linear_regression": 0.0,
                "mlp": 0.0,
                "xgboost": 0.0,
                "lstm": 0.0,
                "cnn": 0.0
            }
            
        biases = {}
        model_keys = ["random_forest", "linear_regression", "mlp", "xgboost", "lstm", "cnn"]
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
            "lstm": 0.0,
            "cnn": 0.0
        }


def calculate_gold_model_comparison(days=30):
    """Compare resolved Gold forecasts over the most recent trading sessions.

    Metrics are calculated only from records with a known actual price. The
    primary ranking metric is MAPE, followed by RMSE and directional accuracy,
    so a model is not promoted merely because it guessed the direction while
    missing the price by a large amount.
    """
    model_keys = [
        "random_forest",
        "linear_regression",
        "mlp",
        "xgboost",
        "lstm",
        "cnn",
        "ensemble",
        "roundtable_vote",
    ]
    model_labels = {
        "random_forest": "Random Forest",
        "linear_regression": "Hồi quy tuyến tính",
        "mlp": "MLP",
        "xgboost": "XGBoost",
        "lstm": "LSTM",
        "cnn": "CNN 1D",
        "ensemble": "Ensemble",
    }
    try:
        days = max(1, min(int(days), 180))
        query = {"actual_price": {"$ne": None, "$gt": 0.0}}
        # One extra record provides the previous close needed for direction
        # scoring when the first evaluated forecast lacks forecast_base_price.
        cursor = db.gold_predictions_history.find(
            query,
            {
                "_id": 0,
                "date": 1,
                "actual_price": 1,
                "forecast_base_price": 1,
                "models": 1,
            },
        ).sort("date", -1).limit(days + 1)
        history = list(cursor)
        history.reverse()
        if len(history) > days:
            evaluation = history[-days:]
        else:
            evaluation = history

        accumulators = {
            key: {
                "absolute_errors": [],
                "squared_errors": [],
                "percentage_errors": [],
                "signed_errors": [],
                "direction_correct": [],
                "predictions": 0,
            }
            for key in model_keys
        }

        evaluation_start_index = len(history) - len(evaluation)
        for index, item in enumerate(evaluation):
            try:
                actual = float(item.get("actual_price"))
            except (TypeError, ValueError):
                continue
            if actual <= 0:
                continue

            base_price = item.get("forecast_base_price")
            try:
                base_price = float(base_price)
            except (TypeError, ValueError):
                base_price = 0.0
            if base_price <= 0:
                # Look first at the preceding resolved record, including the
                # extra context record fetched above.
                history_index = evaluation_start_index + index
                previous_item = history[history_index - 1] if history_index > 0 else None
                try:
                    base_price = float(previous_item.get("actual_price")) if previous_item else 0.0
                except (TypeError, ValueError, AttributeError):
                    base_price = 0.0

            for model_key in model_keys:
                try:
                    predicted = float((item.get("models") or {}).get(model_key))
                except (TypeError, ValueError):
                    continue
                if predicted <= 0:
                    continue
                error = predicted - actual
                metrics = accumulators[model_key]
                metrics["absolute_errors"].append(abs(error))
                metrics["squared_errors"].append(error ** 2)
                metrics["percentage_errors"].append(abs(error) / actual * 100.0)
                metrics["signed_errors"].append(error)
                metrics["predictions"] += 1
                if base_price > 0:
                    actual_direction = 1 if actual > base_price else (-1 if actual < base_price else 0)
                    predicted_direction = 1 if predicted > base_price else (-1 if predicted < base_price else 0)
                    metrics["direction_correct"].append(
                        1 if actual_direction == predicted_direction else 0
                    )

        stats = []
        for model_key in model_keys:
            metrics = accumulators[model_key]
            sample_count = metrics["predictions"]
            if sample_count == 0:
                continue
            direction_count = len(metrics["direction_correct"])
            stats.append({
                "model": model_key,
                "label": model_labels[model_key],
                "samples": sample_count,
                "mae": float(np.mean(metrics["absolute_errors"])),
                "rmse": float(np.sqrt(np.mean(metrics["squared_errors"]))),
                "mape": float(np.mean(metrics["percentage_errors"])),
                "directional_accuracy": (
                    float(np.mean(metrics["direction_correct"]) * 100.0)
                    if direction_count else None
                ),
                "direction_samples": direction_count,
                "mean_error": float(np.mean(metrics["signed_errors"])),
            })

        stats.sort(
            key=lambda row: (
                row["mape"],
                row["rmse"],
                -(row["directional_accuracy"] or 0.0),
            )
        )
        for rank, row in enumerate(stats, start=1):
            row["rank"] = rank
            row["is_best"] = rank == 1

        return {
            "status": "success",
            "window_sessions": days,
            "evaluated_sessions": len(evaluation),
            "resolved_sessions": len(history),
            "from_date": evaluation[0].get("date") if evaluation else None,
            "to_date": evaluation[-1].get("date") if evaluation else None,
            "ranking_metric": "mape_then_rmse_then_direction",
            "best_model": stats[0] if stats else None,
            "models": stats,
        }
    except Exception as error:
        print(f"Error calculating Gold model comparison: {error}")
        return {
            "status": "error",
            "message": str(error),
            "window_sessions": days,
            "evaluated_sessions": 0,
            "models": [],
            "best_model": None,
        }


def run_gold_month_backtest(year=None, month=9):
    """Run a leakage-safe one-step walk-forward backtest for one month.

    Each target session is predicted using only gold history strictly before
    that session. The function intentionally does not use current production
    bias corrections or dynamic weights because either could contain
    information from dates after the target. The returned ensemble is a plain
    average of the available six base-model forecasts.
    """
    model_keys = [
        "random_forest",
        "linear_regression",
        "mlp",
        "xgboost",
        "lstm",
        "cnn",
        "ensemble",
        "roundtable_vote",
    ]
    try:
        now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
        as_of_date = now.date()
        year = int(year if year is not None else now.year)
        month = int(month)
        if year < 2000 or year > 2100 or month < 1 or month > 12:
            raise ValueError("year phải trong khoảng 2000-2100 và month trong khoảng 1-12")

        import predictor as pred_module
        import pandas as pd

        history = _prepare_gold_history_frame(
            list(db.gold_history.find({}, {"_id": 0}).sort("date", 1))
        ).to_dict(orient="records")
        # A backtest may only score sessions whose actual close is known as of
        # today. This prevents seeded/cache rows or a data-source timezone
        # mismatch from making a current September run show prices through
        # September 30 before those sessions happen.
        history = [
            item for item in history
            if str(item.get("date", "")) <= as_of_date.strftime("%Y-%m-%d")
        ]
        month_prefix = f"{year:04d}-{month:02d}-"
        target_indices = [
            index for index, item in enumerate(history)
            if str(item.get("date", "")).startswith(month_prefix)
        ]
        autofill_result = None
        if not target_indices:
            # The live crawler keeps a compact recent history. Expand it on
            # demand for an older selected month before declaring the
            # backtest empty.
            requested_start = datetime(year, month, 1)
            fill_days = max(90, (now - requested_start).days + 35)
            autofill_result = auto_fill_gold_history_from_yfinance(days=fill_days)
            history = _prepare_gold_history_frame(
                list(db.gold_history.find({}, {"_id": 0}).sort("date", 1))
            ).to_dict(orient="records")
            history = [
                item for item in history
                if str(item.get("date", "")) <= as_of_date.strftime("%Y-%m-%d")
            ]
            target_indices = [
                index for index, item in enumerate(history)
                if str(item.get("date", "")).startswith(month_prefix)
            ]
        if not target_indices:
            available_dates = [str(item.get("date")) for item in history if item.get("date")]
            source_error = (autofill_result or {}).get("message") if isinstance(autofill_result, dict) else None
            message = (
                f"Không có dữ liệu giá vàng thực tế cho tháng {month:02d}/{year}."
            )
            if source_error:
                message += f" Không thể tải dữ liệu lịch sử: {source_error}"
            elif available_dates:
                message += (
                    f" Dữ liệu hiện có từ {available_dates[0]} đến {available_dates[-1]}."
                )
            else:
                message += " Nguồn dữ liệu lịch sử chưa trả về dữ liệu."
            return {
                "status": "no_data",
                "message": message,
                "model_version": pred_module.GOLD_MODEL_VERSION,
                "year": year,
                "month": month,
                "month_label": f"{month:02d}/{year}",
                "target_sessions": 0,
                "evaluated_sessions": 0,
                "as_of_date": as_of_date.strftime("%Y-%m-%d"),
                "available_from": available_dates[0] if available_dates else None,
                "available_to": available_dates[-1] if available_dates else None,
                "data_source": (autofill_result or {}).get("source") if isinstance(autofill_result, dict) else None,
                "models": [],
                "rows": [],
                "errors": [],
            }
        macro_list = list(db.macro_history.find({}, {"_id": 0}))
        macro_df = pd.DataFrame(macro_list) if macro_list else pd.DataFrame()
        rows = []
        errors = []
        base_model_keys = model_keys[:6]
        roundtable_performance = {key: [] for key in base_model_keys}

        for target_index in target_indices:
            target_item = history[target_index]
            target_date = target_item.get("date")
            actual_price = target_item.get("world_price")
            try:
                actual_price = float(actual_price)
            except (TypeError, ValueError):
                actual_price = 0.0
            if actual_price <= 0:
                continue

            hist_slice = history[:target_index]
            if len(hist_slice) < 21:
                errors.append(f"{target_date}: không đủ tối thiểu 21 phiên lịch sử")
                continue

            try:
                frame = pd.DataFrame(hist_slice)
                frame["world_price"] = frame["world_price"].replace(0.0, np.nan).ffill()
                frame["close"] = frame["world_price"]
                frame["open"] = frame["world_price"]
                frame["high"] = frame["world_price"]
                frame["low"] = frame["world_price"]
                frame["volume"] = 1.0
                if not macro_df.empty and "date" in macro_df.columns:
                    frame = pd.merge(frame, macro_df, on="date", how="left")
                    macro_columns = [
                        "dxy", "us10y", "vix", "brent", "dji", "spx",
                    "eurusd", "xagusd", "real_yield", "gld", "gld_trust",
                    "pce_headline_yoy", "pce_core_yoy", "pce_headline_mom", "pce_core_mom",
                    ]
                    available = [column for column in macro_columns if column in frame.columns]
                    if available:
                        frame[available] = frame[available].ffill()
                if "dji" not in frame.columns:
                    frame["dji"] = 35000.0
                if "spx" not in frame.columns:
                    frame["spx"] = 5000.0

                # Keep the calendar-aware event channels available to CNN
                # during walk-forward evaluation without using any released
                # value from after the target session.
                try:
                    days_to_fed, days_to_cpi, days_to_nfp, days_to_pce = calculate_days_until_events(frame["date"])
                    frame["days_to_fed"] = days_to_fed
                    frame["days_to_cpi"] = days_to_cpi
                    frame["days_to_nfp"] = days_to_nfp
                    frame["days_to_pce"] = days_to_pce
                except Exception:
                    frame["days_to_fed"] = 15.0
                    frame["days_to_cpi"] = 15.0
                    frame["days_to_nfp"] = 15.0
                    frame["days_to_pce"] = 15.0

                frame_ind = pred_module.calculate_technical_indicators(frame)
                predictions = {}
                model_errors = {}
                model_calls = {
                    "random_forest": lambda: pred_module.predict_future_prices_rf(frame_ind, days_to_predict=1),
                    "linear_regression": lambda: pred_module.predict_future_prices(frame_ind, days_to_predict=1),
                    "mlp": lambda: pred_module.predict_future_prices_mlp(frame_ind, days_to_predict=1),
                    "xgboost": lambda: pred_module.predict_future_prices_xgb(frame_ind, days_to_predict=1),
                    "lstm": lambda: pred_module.predict_future_prices_lstm(frame_ind, days_to_predict=1),
                    "cnn": lambda: pred_module.predict_future_prices_cnn(frame_ind, days_to_predict=1),
                }
                for model_key, model_call in model_calls.items():
                    try:
                        result = model_call()
                        prediction = float(result[0][0])
                        if prediction > 0 and np.isfinite(prediction):
                            predictions[model_key] = prediction
                            model_errors[model_key] = {
                                "absolute": abs(prediction - actual_price),
                                "percentage": abs(prediction - actual_price) / actual_price * 100.0,
                            }
                    except Exception as model_error:
                        errors.append(f"{target_date}/{model_key}: {model_error}")

                base_price = float(frame["close"].iloc[-1])
                base_predictions = [
                    predictions[key]
                    for key in model_keys[:-1]
                    if key in predictions
                ]
                if base_predictions:
                    predictions["ensemble"] = float(np.mean(base_predictions))
                    model_errors["ensemble"] = {
                        "absolute": abs(predictions["ensemble"] - actual_price),
                        "percentage": abs(predictions["ensemble"] - actual_price) / actual_price * 100.0,
                    }
                # Reproducible meeting decision. Weights may use only errors
                # from earlier target sessions, never this session's actual.
                available_members = [key for key in base_model_keys if key in predictions]
                raw_weights = {}
                for model_key in available_members:
                    prior_errors = roundtable_performance[model_key]
                    if len(prior_errors) >= 3:
                        inverse_error = 1.0 / max(float(np.mean(prior_errors)), 0.01)
                        raw_weights[model_key] = float(np.clip(inverse_error, 0.25, 4.0))
                    else:
                        raw_weights[model_key] = 1.0
                weight_total = sum(raw_weights.values()) or 1.0
                member_votes = []
                roundtable_return = 0.0
                vote_tally = {"up": 0.0, "down": 0.0, "sideways": 0.0}
                direction_threshold = base_price * 0.0002
                for model_key in available_members:
                    weight = raw_weights[model_key] / weight_total
                    model_price = predictions[model_key]
                    change = model_price - base_price
                    vote = "up" if change > direction_threshold else ("down" if change < -direction_threshold else "sideways")
                    vote_tally[vote] += weight
                    roundtable_return += weight * (model_price / base_price - 1.0)
                    prior_errors = roundtable_performance[model_key]
                    prior_text = (
                        f"MAPE quá khứ {float(np.mean(prior_errors)):.2f}%/{len(prior_errors)} phiên"
                        if prior_errors else "chưa có phiên trước trong cửa sổ"
                    )
                    member_votes.append({
                        "model": model_key,
                        "weight": float(weight),
                        "vote": vote,
                        "price": float(model_price),
                        "pct_change": float(change / base_price * 100.0),
                        "reason": (
                            f"Dự báo {model_price:.2f} USD ({change / base_price * 100.0:+.2f}%); "
                            f"trọng số dựa trên {prior_text}. Phản biện: các model dùng chung dữ liệu nên phiếu có tương quan."
                        ),
                    })
                roundtable_price = float(base_price * (1.0 + roundtable_return))
                if available_members:
                    predictions["roundtable_vote"] = roundtable_price
                    model_errors["roundtable_vote"] = {
                        "absolute": abs(roundtable_price - actual_price),
                        "percentage": abs(roundtable_price - actual_price) / actual_price * 100.0,
                    }
                decision_change = roundtable_price - base_price
                decision_vote = "up" if decision_change > direction_threshold else ("down" if decision_change < -direction_threshold else "sideways")
                actual_change = actual_price - base_price
                actual_vote = "up" if actual_change > direction_threshold else ("down" if actual_change < -direction_threshold else "sideways")
                roundtable_record = {
                    "formed_before_actual": True,
                    "weight_basis": "inverse prior-session MAPE; equal until 3 prior observations",
                    "member_votes": member_votes,
                    "weighted_tally": {key: float(value) for key, value in vote_tally.items()},
                    "decision": {
                        "price": roundtable_price,
                        "vote": decision_vote,
                        "pct_change": float(decision_change / base_price * 100.0),
                        "confidence": "medium" if max(vote_tally.values(), default=0.0) >= 0.75 else "low",
                    },
                    "evaluation": {
                        "actual_vote": actual_vote,
                        "direction_correct": decision_vote == actual_vote,
                        "absolute_error": abs(roundtable_price - actual_price),
                        "percentage_error": abs(roundtable_price - actual_price) / actual_price * 100.0,
                    },
                }
                rows.append({
                    "date": target_date,
                    "actual_price": actual_price,
                    "base_price": base_price,
                    "models": predictions,
                    "errors": model_errors,
                    "roundtable": roundtable_record,
                })
                for model_key in available_members:
                    roundtable_performance[model_key].append(
                        abs(predictions[model_key] - actual_price) / actual_price * 100.0
                    )
            except Exception as day_error:
                errors.append(f"{target_date}: {day_error}")

        stats = []
        for model_key in model_keys:
            absolute_errors = []
            squared_errors = []
            percentage_errors = []
            direction_results = []
            signed_errors = []
            for row in rows:
                prediction = row["models"].get(model_key)
                if prediction is None:
                    continue
                actual = row["actual_price"]
                error = prediction - actual
                absolute_errors.append(abs(error))
                squared_errors.append(error ** 2)
                percentage_errors.append(abs(error) / actual * 100.0)
                signed_errors.append(error)
                base_price = row.get("base_price", 0.0)
                if base_price > 0:
                    actual_direction = 1 if actual > base_price else (-1 if actual < base_price else 0)
                    predicted_direction = 1 if prediction > base_price else (-1 if prediction < base_price else 0)
                    direction_results.append(1 if actual_direction == predicted_direction else 0)
            if not absolute_errors:
                continue
            stats.append({
                "model": model_key,
                "label": {
                    "random_forest": "Random Forest",
                    "linear_regression": "Hồi quy tuyến tính",
                    "mlp": "MLP",
                    "xgboost": "XGBoost",
                    "lstm": "LSTM",
                    "cnn": "CNN 1D",
                    "ensemble": "Ensemble",
                    "roundtable_vote": "Quyết định hội nghị",
                }[model_key],
                "samples": len(absolute_errors),
                "mae": float(np.mean(absolute_errors)),
                "rmse": float(np.sqrt(np.mean(squared_errors))),
                "mape": float(np.mean(percentage_errors)),
                "directional_accuracy": float(np.mean(direction_results) * 100.0) if direction_results else None,
                "direction_samples": len(direction_results),
                "mean_error": float(np.mean(signed_errors)),
            })

        stats.sort(
            key=lambda row: (
                row["mape"],
                row["rmse"],
                -(row["directional_accuracy"] or 0.0),
            )
        )
        for rank, row in enumerate(stats, start=1):
            row["rank"] = rank
            row["is_best"] = rank == 1

        return {
            "status": "success",
            "model_version": pred_module.GOLD_MODEL_VERSION,
            "year": year,
            "month": month,
            "month_label": f"{month:02d}/{year}",
            "target_sessions": len(target_indices),
            "evaluated_sessions": len(rows),
            "as_of_date": as_of_date.strftime("%Y-%m-%d"),
            "from_date": rows[0]["date"] if rows else None,
            "to_date": rows[-1]["date"] if rows else None,
            "ranking_metric": "mape_then_rmse_then_direction",
            "roundtable_backtest_mode": "walk_forward_prior_error_weighted_vote",
            "roundtable_summary": next((item for item in stats if item["model"] == "roundtable_vote"), None),
            "best_model": stats[0] if stats else None,
            "models": stats,
            "rows": rows,
            "errors": errors[:30],
        }
    except Exception as error:
        print(f"Error running Gold month backtest: {error}")
        return {
            "status": "error",
            "message": str(error),
            "model_version": pred_module.GOLD_MODEL_VERSION if "pred_module" in locals() else None,
            "year": year,
            "month": month,
            "target_sessions": 0,
            "evaluated_sessions": 0,
            "as_of_date": as_of_date.strftime("%Y-%m-%d") if "as_of_date" in locals() else None,
            "models": [],
            "rows": [],
            "errors": [str(error)],
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
    Runs leakage-safe multi-horizon tuning and caches only OOS improvements.
    """
    try:
        import predictor as pred_module
        import pandas as pd
        history = _prepare_gold_history_frame(
            list(db.gold_history.find({}, {"_id": 0}).sort("date", 1))
        ).to_dict(orient="records")
        if len(history) < 50:
            print("Insufficient gold history to tune parameters.")
            return {"status": "insufficient_data", "samples": len(history)}
            
        df = _prepare_gold_history_frame(history)
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
            cols = [c for c in [
                "dxy", "us10y", "vix", "brent", "dji", "spx", "eurusd",
                "xagusd", "real_yield", "gld", "gld_trust",
                "pce_headline_yoy", "pce_core_yoy", "pce_headline_mom", "pce_core_mom",
            ] if c in df.columns]
            df[cols] = df[cols].ffill()

        try:
            days_to_fed, days_to_cpi, days_to_nfp, days_to_pce = calculate_days_until_events(df["date"])
            df["days_to_fed"] = days_to_fed
            df["days_to_cpi"] = days_to_cpi
            df["days_to_nfp"] = days_to_nfp
            df["days_to_pce"] = days_to_pce
        except Exception as event_error:
            print(f"Tuning event features unavailable: {event_error}")
            for column in ("days_to_fed", "days_to_cpi", "days_to_nfp", "days_to_pce"):
                df[column] = 15.0
            
        df_ind = pred_module.calculate_technical_indicators(df)
        existing = db.gold_model_hyperparameters.find_one({"type": "gold_params"}) or {}
        print("Running multi-horizon walk-forward optimization for Gold...")
        best_params, diagnostics = pred_module.optimize_hyperparameters(
            df_ind,
            days_to_predict=5,
            incumbent_params=existing.get("params") or {},
            return_diagnostics=True,
        )
        
        if best_params:
            db.gold_model_hyperparameters.update_one(
                {"type": "gold_params"},
                {"$set": {
                    "type": "gold_params",
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model_version": pred_module.GOLD_MODEL_VERSION,
                    "params": best_params,
                    "diagnostics": diagnostics,
                }},
                upsert=True
            )
            print("Successfully updated Gold Hyperparameters cache in MongoDB.")
            print(f"Optimized Params: {best_params}")
        return {"status": diagnostics.get("status", "success"), "params": best_params, "diagnostics": diagnostics}
    except Exception as e:
        print(f"Error tuning gold hyperparameters: {e}")
        return {"status": "error", "message": str(e)}


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

        history = _prepare_gold_history_frame(
            list(db.gold_history.find({}, {"_id": 0}).sort("date", 1))
        ).to_dict(orient="records")
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
            df["world_price"] = df["world_price"].replace(0.0, np.nan).ffill()
            df["close"] = df["world_price"]
            df["open"]  = df["world_price"]
            df["high"]  = df["world_price"]
            df["low"]   = df["world_price"]
            df["volume"] = 1.0

            # Merge macro data
            if not macro_df.empty and "date" in macro_df.columns:
                df = pd.merge(df, macro_df, on="date", how="left")
                cols = [c for c in ["dxy", "us10y", "vix", "brent", "dji", "spx", "eurusd", "xagusd", "real_yield", "gld", "gld_trust"] if c in df.columns]
                df[cols] = df[cols].ffill()
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
                cnn_p,  _, _, _ = pred_module.predict_future_prices_cnn(df_ind, days_to_predict=1)

                weights = calculate_dynamic_ensemble_weights()
                w_rf   = weights.get("random_forest",    1 / 6)
                w_lr   = weights.get("linear_regression", 1 / 6)
                w_mlp  = weights.get("mlp",              1 / 6)
                w_xgb  = weights.get("xgboost",          1 / 6)
                w_lstm = weights.get("lstm",             1 / 6)
                w_cnn  = weights.get("cnn",              1 / 6)
                ens_p  = float(rf_p[0]*w_rf + lr_p[0]*w_lr + mlp_p[0]*w_mlp
                               + xgb_p[0]*w_xgb + lstm_p[0]*w_lstm + cnn_p[0]*w_cnn)

                doc = {
                    "date": target_date,
                    "actual_price": float(actual_price),
                    "forecast_base_price": float(hist_slice[-1].get("world_price", 0.0)),
                    "forecast_source": "walk_forward_backfill",
                    "models": {
                        "random_forest":     float(rf_p[0]),
                        "linear_regression": float(lr_p[0]),
                        "mlp":               float(mlp_p[0]),
                        "xgboost":           float(xgb_p[0]),
                        "lstm":              float(lstm_p[0]),
                        "cnn":               float(cnn_p[0]),
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
        try:
            import yfinance as yf
        except Exception as import_error:
            message = f"thiếu thư viện yfinance ({import_error})"
            print(f"Auto-fill Gold history unavailable: {message}")
            return {"status": "error", "message": message, "source": "yfinance"}
        days = max(int(days), 90)
        if days <= 120:
            period = "3mo"
        elif days <= 365:
            period = "1y"
        elif days <= 730:
            period = "2y"
        else:
            period = "5y"
        print(
            f"Auto-filling missing gold history points from yfinance "
            f"for past {days} days ({period})..."
        )

        # Download GC=F data. The old implementation always requested only
        # three months, which made a September 2025 backtest return 0/0 when
        # the current date was September 2026.
        df = yf.download("GC=F", period=period, progress=False)
        if df.empty:
            print("yfinance download for GC=F was empty.")
            return {
                "status": "error",
                "message": "Yahoo Finance không trả về dữ liệu GC=F.",
                "source": "yfinance",
                "requested_days": days,
            }
            
        import pandas as pd
        if isinstance(df.columns, pd.MultiIndex):
            close_prices = df[("Close", "GC=F")].ffill()
        else:
            close_prices = df["Close"].ffill()
            
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
        return {
            "status": "success",
            "source": "yfinance",
            "requested_days": days,
            "downloaded_rows": int(len(close_prices)),
            "inserted": inserted_count,
        }
    except Exception as e:
        print(f"Error in auto_fill_gold_history_from_yfinance: {e}")
        return {"status": "error", "message": str(e), "source": "yfinance"}

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

def _next_gold_business_date(date_str):
    """Return the next weekday used for the Gold T+1 forecast record."""
    next_date = datetime.strptime(str(date_str), "%Y-%m-%d") + timedelta(days=1)
    while next_date.weekday() >= 5:
        next_date += timedelta(days=1)
    return next_date.strftime("%Y-%m-%d")


def save_gold_forecast_history(analysis, latest_date):
    """Save the exact live-analysis forecast used by the dashboard.

    Only the future forecast is replaced. If a record has already received an
    actual price, that ground truth is preserved for verification/self-learning.
    """
    forecast_date = _next_gold_business_date(latest_date)
    model_values = {}
    for model_key, model_data in (analysis.get("models") or {}).items():
        prices = ((model_data or {}).get("ml_prediction") or {}).get("predicted_prices") or []
        if not prices:
            continue
        try:
            value = float(prices[0])
            if np.isfinite(value) and value > 0:
                model_values[model_key] = value
        except (TypeError, ValueError):
            continue

    if not model_values:
        return None

    db.gold_predictions_history.update_one(
        {"date": forecast_date},
        {
            "$set": {
                "date": forecast_date,
                "models": model_values,
                "forecast_base_price": float(analysis.get("current_price", 0.0)),
                "forecast_source": "live_analysis",
                "model_version": analysis.get("model_version"),
                "forecast_direction": analysis.get("direction"),
                "forecast_confidence": (
                    (analysis.get("ensemble_diagnostics") or {}).get("confidence_label")
                ),
                "market_context": (
                    (analysis.get("ensemble_diagnostics") or {}).get("market_context")
                ),
                "forecast_updated_at": datetime.now(),
            },
            "$setOnInsert": {"actual_price": None},
        },
        upsert=True,
    )
    print(f"Successfully synchronized live Gold forecast history for {forecast_date}.")
    return forecast_date


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

        history = _prepare_gold_history_frame(
            list(db.gold_history.find({}, {"_id": 0}).sort("date", 1))
        ).to_dict(orient="records")
        if len(history) < 21:
            return
            
        import predictor
        import pandas as pd
        import numpy as np

        # Load custom hyperparameters & rolling biases (self-learning)
        params = load_gold_model_hyperparameters()
        biases = calculate_gold_model_biases(
            limit=15,
            model_version=predictor.GOLD_MODEL_VERSION,
        )
        
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

            df_today["world_price"] = df_today["world_price"].replace(0.0, np.nan).ffill()
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
                cols = [c for c in ["dxy", "us10y", "vix", "brent", "dji", "spx", "eurusd", "xagusd", "real_yield", "gld", "gld_trust"] if c in df_today.columns]
                df_today[cols] = df_today[cols].ffill()

            # Calculate technical indicators & predict today
            df_ind_today = predictor.calculate_technical_indicators(df_today)
            lr_p, _, _ = predictor.predict_future_prices(
                df_ind_today,
                days_to_predict=1,
                params=params.get("linear_regression"),
            )
            rf_p, _, _, _ = predictor.predict_future_prices_rf(df_ind_today, days_to_predict=1, params=params.get("random_forest"))
            mlp_p, _, _, _ = predictor.predict_future_prices_mlp(df_ind_today, days_to_predict=1, params=params.get("mlp"))
            xgb_p, _, _, _ = predictor.predict_future_prices_xgb(df_ind_today, days_to_predict=1, params=params.get("xgboost"))
            lstm_p, _, _, _ = predictor.predict_future_prices_lstm(df_ind_today, days_to_predict=1)
            cnn_p, _, _, _ = predictor.predict_future_prices_cnn(df_ind_today, days_to_predict=1)
            
            # Apply rolling bias correction (today)
            rf_p = [p + biases.get("random_forest", 0.0) for p in rf_p]
            lr_p = [p + biases.get("linear_regression", 0.0) for p in lr_p]
            mlp_p = [p + biases.get("mlp", 0.0) for p in mlp_p]
            xgb_p = [p + biases.get("xgboost", 0.0) for p in xgb_p]
            lstm_p = [p + biases.get("lstm", 0.0) for p in lstm_p]
            cnn_p = [p + biases.get("cnn", 0.0) for p in cnn_p]
            
            # Fetch dynamic weights
            weights_today = calculate_dynamic_ensemble_weights(
                model_version=predictor.GOLD_MODEL_VERSION,
            )
            w_rf_t   = weights_today.get("random_forest", 1 / 6)
            w_lr_t   = weights_today.get("linear_regression", 1 / 6)
            w_mlp_t  = weights_today.get("mlp", 1 / 6)
            w_xgb_t  = weights_today.get("xgboost", 1 / 6)
            w_lstm_t = weights_today.get("lstm", 1 / 6)
            w_cnn_t  = weights_today.get("cnn", 1 / 6)
            ens_p = float(rf_p[0]*w_rf_t + lr_p[0]*w_lr_t + mlp_p[0]*w_mlp_t + xgb_p[0]*w_xgb_t + lstm_p[0]*w_lstm_t + cnn_p[0]*w_cnn_t)
            
            today_doc = {
                "date": today_date,
                "actual_price": float(actual_price),
                "forecast_base_price": float(df_today["close"].iloc[-1]),
                "forecast_source": "history_fallback",
                "model_version": predictor.GOLD_MODEL_VERSION,
                "models": {
                    "random_forest": float(rf_p[0]),
                    "linear_regression": float(lr_p[0]),
                    "mlp": float(mlp_p[0]),
                    "xgboost": float(xgb_p[0]),
                    "lstm": float(lstm_p[0]),
                    "cnn": float(cnn_p[0]),
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
            df_tomorrow["world_price"] = df_tomorrow["world_price"].replace(0.0, np.nan).ffill()
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
                cols = [c for c in ["dxy", "us10y", "vix", "brent", "dji", "spx", "eurusd", "xagusd", "real_yield", "gld", "gld_trust"] if c in df_tomorrow.columns]
                df_tomorrow[cols] = df_tomorrow[cols].ffill()

            df_ind_tomorrow = predictor.calculate_technical_indicators(df_tomorrow)
            lr_pt, _, _ = predictor.predict_future_prices(
                df_ind_tomorrow,
                days_to_predict=1,
                params=params.get("linear_regression"),
            )
            rf_pt, _, _, _ = predictor.predict_future_prices_rf(df_ind_tomorrow, days_to_predict=1, params=params.get("random_forest"))
            mlp_pt, _, _, _ = predictor.predict_future_prices_mlp(df_ind_tomorrow, days_to_predict=1, params=params.get("mlp"))
            xgb_pt, _, _, _ = predictor.predict_future_prices_xgb(df_ind_tomorrow, days_to_predict=1, params=params.get("xgboost"))
            lstm_pt, _, _, _ = predictor.predict_future_prices_lstm(df_ind_tomorrow, days_to_predict=1)
            cnn_pt, _, _, _ = predictor.predict_future_prices_cnn(df_ind_tomorrow, days_to_predict=1)
            
            # Apply rolling bias correction (tomorrow)
            rf_pt = [p + biases.get("random_forest", 0.0) for p in rf_pt]
            lr_pt = [p + biases.get("linear_regression", 0.0) for p in lr_pt]
            mlp_pt = [p + biases.get("mlp", 0.0) for p in mlp_pt]
            xgb_pt = [p + biases.get("xgboost", 0.0) for p in xgb_pt]
            lstm_pt = [p + biases.get("lstm", 0.0) for p in lstm_pt]
            cnn_pt = [p + biases.get("cnn", 0.0) for p in cnn_pt]

            # Fetch dynamic weights
            weights = calculate_dynamic_ensemble_weights(
                model_version=predictor.GOLD_MODEL_VERSION,
            )
            w_rf   = weights.get("random_forest", 1 / 6)
            w_lr   = weights.get("linear_regression", 1 / 6)
            w_mlp  = weights.get("mlp", 1 / 6)
            w_xgb  = weights.get("xgboost", 1 / 6)
            w_lstm = weights.get("lstm", 1 / 6)
            w_cnn  = weights.get("cnn", 1 / 6)
            ens_pt = float(rf_pt[0]*w_rf + lr_pt[0]*w_lr + mlp_pt[0]*w_mlp + xgb_pt[0]*w_xgb + lstm_pt[0]*w_lstm + cnn_pt[0]*w_cnn)
 
            tomorrow_doc = {
                "date": tomorrow_date,
                "actual_price": None,
                "forecast_base_price": float(df_tomorrow["close"].iloc[-1]),
                "forecast_source": "scheduled_prediction",
                "model_version": predictor.GOLD_MODEL_VERSION,
                "models": {
                    "random_forest": float(rf_pt[0]),
                    "linear_regression": float(lr_pt[0]),
                    "mlp": float(mlp_pt[0]),
                    "xgboost": float(xgb_pt[0]),
                    "lstm": float(lstm_pt[0]),
                    "cnn": float(cnn_pt[0]),
                    "ensemble": ens_pt
                }
            }
            existing_tomorrow = db.gold_predictions_history.find_one(
                {"date": tomorrow_date},
                {"_id": 0, "forecast_source": 1},
            ) or {}
            if existing_tomorrow.get("forecast_source") == "live_analysis":
                print(
                    f"Preserved canonical live-analysis forecast for tomorrow: {tomorrow_date}"
                )
            else:
                db.gold_predictions_history.update_one(
                    {"date": tomorrow_date}, {"$set": tomorrow_doc}, upsert=True
                )
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

def get_gold_predictions(progress_callback=None):
    """
    Loads World gold price history from MongoDB, merges macroeconomic indicators
    and event proximity features, runs technical indicators, and returns predictions using predictor.py.
    """
    def report_progress(stage, progress):
        if callable(progress_callback):
            try:
                progress_callback(stage, progress)
            except Exception as progress_err:
                print(f"Gold prediction progress callback failed: {progress_err}")

    init_db() # Ensure seeded
    try:
        report_progress("Đọc lịch sử giá vàng từ MongoDB", 8)
        cursor = db.gold_history.find({}, {"_id": 0}).sort("date", 1)
        history_list = list(cursor)
        if len(history_list) < 20:
            return {
                "status": "error",
                "message": "Không đủ dữ liệu lịch sử vàng để phân tích (yêu cầu tối thiểu 20 phiên)."
            }
            
        import predictor
        # Convert to trading-day observations. Weekend carry-forward rows make
        # a 5-step forecast mean five calendar rows instead of five sessions.
        df = _prepare_gold_history_frame(history_list)
        if len(df) < 20:
            return {
                "status": "error",
                "message": "Không đủ dữ liệu phiên giao dịch vàng sau khi làm sạch.",
            }
        report_progress("Chuẩn bị dữ liệu giá và biến mục tiêu", 15)
        
        # Ensure world_price has no zeros or NaNs (interpolate or forward fill)
        df["world_price"] = df["world_price"].replace(0.0, np.nan).ffill()
        
        # World gold price (USD/ounce) is the primary prediction target
        df["close"] = df["world_price"]
        df["open"] = df["world_price"]
        df["high"] = df["world_price"]
        df["low"] = df["world_price"]
        df["volume"] = 1.0 # placeholder
        
        # --- Merge Macro Indicators ---
        macro_list = []
        try:
            report_progress("Đồng bộ DXY, lợi suất, PCE/Core PCE và dữ liệu thị trường", 22)
            latest_macro_doc = db.macro_history.find_one(
                {}, {"_id": 0, "pce_release_date": 1}, sort=[("date", -1)]
            ) or {}
            if not latest_macro_doc.get("pce_release_date"):
                update_pce_history_from_fred(days=365)
            macro_cursor = db.macro_history.find({}, {"_id": 0})
            macro_list = list(macro_cursor)
            if macro_list:
                macro_df = pd.DataFrame(macro_list)
                # Merge on date
                df = pd.merge(df, macro_df, on="date", how="left")
                # Forward-fill only: earlier rows must not see future macro values.
                cols_to_fill = [c for c in [
                    "dxy", "us10y", "vix", "brent", "dji", "spx", "eurusd",
                    "xagusd", "real_yield", "gld", "gld_trust",
                    "pce_headline_yoy", "pce_core_yoy", "pce_headline_mom", "pce_core_mom",
                ] if c in df.columns]
                df[cols_to_fill] = df[cols_to_fill].ffill()
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
            report_progress("Tính khoảng cách tới lịch FED, CPI, NFP và PCE", 31)
            days_to_fed, days_to_cpi, days_to_nfp, days_to_pce = calculate_days_until_events(df["date"])
            df["days_to_fed"] = days_to_fed
            df["days_to_cpi"] = days_to_cpi
            df["days_to_nfp"] = days_to_nfp
            df["days_to_pce"] = days_to_pce
        except Exception as event_err:
            print(f"Error adding event proximity features: {event_err}")
            df["days_to_fed"] = 15.0
            df["days_to_cpi"] = 15.0
            df["days_to_nfp"] = 15.0
            df["days_to_pce"] = 15.0

        # Load already-released market signals before inference. Previously
        # these were appended after prediction, so PCE/profit-taking context
        # could be shown in the UI without influencing the ensemble.
        market_signals = []
        latest_cache = {}
        try:
            latest_cache = db.gold_prices.find_one(
                {"_id": "latest_prices"},
                {"_id": 0, "market_signals": 1, "macro_indicators": 1, "calendar": 1},
            ) or {}
            market_signals = latest_cache.get("market_signals", []) or []
            latest_macro_values = {}
            for signal in market_signals:
                values = signal.get("macro_values") or {}
                if isinstance(values, dict):
                    latest_macro_values.update(values)
            for field in ("pce_headline_yoy", "pce_core_yoy", "pce_headline_mom", "pce_core_mom"):
                value = latest_macro_values.get(field)
                if value is not None:
                    df.loc[df.index[-1], field] = float(value)
        except Exception as signal_err:
            print(f"Error loading Gold market signals before inference: {signal_err}")

        # Build the policy layer from the same released inputs used by the
        # dashboard. This keeps the gold forecast and the UI explanation in
        # sync instead of showing a FED signal that never reached the model.
        try:
            calendar = latest_cache.get("calendar") or get_us_economic_calendar()
            macro_indicators = latest_cache.get("macro_indicators") or fetch_us_macro_indicators()
            fed_policy_outlook = build_fed_policy_outlook(
                macro_indicators=macro_indicators,
                macro_history=macro_list[-2:],
                calendar=calendar,
                market_signals=market_signals,
            )
        except Exception as policy_error:
            print(f"Error building FED policy outlook: {policy_error}")
            fed_policy_outlook = build_fed_policy_outlook(
                macro_history=macro_list[-2:],
                calendar=[],
                market_signals=market_signals,
            )
        
        # --- Add Geopolitical Conflict Events ---
        conflict_events = []
        try:
            report_progress("Đọc tín hiệu rủi ro địa chính trị", 38)
            conflict_events = fetch_geopolitical_conflicts()
        except Exception as e:
            print(f"Error fetching geopolitical conflicts for predictions: {e}")
        
        report_progress("Chuẩn bị huấn luyện và đánh giá các mô hình AI", 45)
        # Resolve yesterday's forecasts before deriving production priors.
        # Otherwise the adaptive ensemble can keep using stale accuracy weights
        # even though the matching actual Gold close is already available.
        resolve_unresolved_predictions()
        weights, production_metrics = calculate_dynamic_ensemble_weights(
            return_diagnostics=True,
            model_version=predictor.GOLD_MODEL_VERSION,
        )
        params = load_gold_model_hyperparameters()
        biases = calculate_gold_model_biases(
            limit=15,
            model_version=predictor.GOLD_MODEL_VERSION,
        )
        analysis = predictor.analyze_and_recommend(
            df,
            model_weights=weights,
            params=params,
            biases=biases,
            production_metrics=production_metrics,
            conflict_events=conflict_events,
            market_signals=market_signals,
            fed_policy_outlook=fed_policy_outlook,
            progress_callback=report_progress,
        )
        analysis.setdefault("market_signals", market_signals)
        analysis["fed_policy_outlook"] = fed_policy_outlook
        latest_date = str(df.iloc[-1]["date"])
        forecast_date = _next_gold_business_date(latest_date)
        analysis["forecast_date"] = forecast_date
        analysis["forecast_base_price"] = float(analysis.get("current_price", 0.0))
        analysis["forecast_generated_at"] = datetime.now().isoformat()

        # Give the Codex chair the observations behind the model reports,
        # not only the models' final votes. Values are deliberately compact,
        # chronological and JSON-safe so the final reasoning is auditable.
        advisor_history_columns = [
            "date", "world_price", "dxy", "us10y", "real_yield", "vix",
            "brent", "dji", "spx", "gld", "gld_trust",
            "pce_headline_yoy", "pce_core_yoy", "pce_headline_mom", "pce_core_mom",
            "days_to_fed", "days_to_cpi", "days_to_nfp", "days_to_pce",
        ]
        advisor_history = []
        for _, history_row in df.tail(30).iterrows():
            observation = {}
            for column in advisor_history_columns:
                if column not in history_row or not pd.notna(history_row[column]):
                    continue
                value = history_row[column]
                if isinstance(value, (np.integer, int)):
                    observation[column] = int(value)
                elif isinstance(value, (np.floating, float)):
                    observation[column] = round(float(value), 6)
                else:
                    observation[column] = str(value)
            advisor_history.append(observation)

        calendar_rows = list(calendar or [])
        today_key = datetime.now().strftime("%Y-%m-%d")
        past_calendar = [event for event in calendar_rows if str(event.get("date") or "") <= today_key]
        future_calendar = [event for event in calendar_rows if str(event.get("date") or "") > today_key]
        # Include the latest released events and the nearest upcoming events;
        # taking the first rows of an annual ascending calendar would mostly
        # send stale January data to the chair in September.
        advisor_calendar_source = past_calendar[-20:] + future_calendar[:10]
        advisor_calendar = []
        for event in advisor_calendar_source:
            if not isinstance(event, dict):
                continue
            compact_event = {
                key: event.get(key)
                for key in (
                    "date", "time", "event", "category", "status", "actual",
                    "forecast", "previous", "direction", "policy_tone",
                    "policy_tone_label", "gold_impact",
                )
                if event.get(key) is not None
            }
            if isinstance(event.get("result"), dict):
                compact_event["result"] = {
                    key: event["result"].get(key)
                    for key in (
                        "value", "unit", "previous", "direction", "release_period",
                        "source",
                    )
                    if event["result"].get(key) is not None
                }
            advisor_calendar.append(compact_event)
        advisor_conflicts = []
        for event in (conflict_events or [])[:15]:
            if not isinstance(event, dict):
                continue
            advisor_conflicts.append({
                key: event.get(key)
                for key in (
                    "title", "event", "date", "region", "status", "severity",
                    "gold_impact", "summary", "source",
                )
                if event.get(key) is not None
            })
        analysis["advisor_input_data"] = {
            "data_groups": [
                "30 phiên giá vàng và biến vĩ mô gần nhất",
                "PCE/Core PCE và khoảng cách tới FED, CPI, NFP, PCE",
                "lịch sự kiện kinh tế Mỹ kèm actual/forecast/previous",
                "tín hiệu thị trường đã công bố",
                "sự kiện địa chính trị và tác động dự kiến",
                "đặc trưng kỹ thuật, backtest OOS và phiếu từng mô hình",
                "phiếu suy luận hành động FED của từng model và nowcast chính sách FED",
            ],
            "recent_market_history": advisor_history,
            "economic_calendar": advisor_calendar,
            "market_signals": market_signals[:12],
            "geopolitical_events": advisor_conflicts,
            "macro_snapshot": macro_indicators if isinstance(macro_indicators, dict) else {},
        }

        # Codex is the qualitative chair after the deterministic vote. An API
        # error never invalidates the numeric forecast;
        # the UI transparently falls back to Ensemble and shows the reason.
        report_progress("Codex đang đọc báo cáo và biểu quyết của hội nghị", 95)
        try:
            from openai_advisor import generate_gold_roundtable_advice

            chatgpt_advisor = generate_gold_roundtable_advice(analysis)
        except Exception as advisor_error:
            chatgpt_advisor = {
                "status": "error",
                "provider": "Rùa AI Codex",
                "model": os.getenv("CODEX_MODEL", "") or "Codex mặc định",
                "generated_at": datetime.now().isoformat(),
                "message": str(advisor_error)[:300],
                "decisions": [],
            }
        analysis["chatgpt_advisor"] = chatgpt_advisor
        if isinstance(analysis.get("roundtable"), dict):
            analysis["roundtable"]["chatgpt_advisor"] = chatgpt_advisor

        # Persist the exact per-model T+1 values returned to the dashboard.
        # The scheduled history updater has a separate lightweight path; if it
        # writes first, the verification table could otherwise show a stale
        # ensemble value that disagrees with the live forecast cards.
        try:
            save_gold_forecast_history(analysis, latest_date)
        except Exception as history_err:
            print(f"Error saving live Gold forecast history: {history_err}")
        report_progress("Tổng hợp dự báo và điểm tin cậy của mô hình", 97)
        
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
                "gld_trust": float(row.get("gld_trust")) if pd.notna(row.get("gld_trust")) else 0.0,
                "pce_headline_yoy": float(row.get("pce_headline_yoy")) if pd.notna(row.get("pce_headline_yoy")) else None,
                "pce_core_yoy": float(row.get("pce_core_yoy")) if pd.notna(row.get("pce_core_yoy")) else None,
                "pce_headline_mom": float(row.get("pce_headline_mom")) if pd.notna(row.get("pce_headline_mom")) else None,
                "pce_core_mom": float(row.get("pce_core_mom")) if pd.notna(row.get("pce_core_mom")) else None,
            })
            
        analysis["gold_history"] = plot_history
        report_progress("Đóng gói biểu đồ lịch sử và biến động vĩ mô", 98)
        
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
            if "days_to_pce" in last_row:
                latest_macro["days_to_pce"] = int(last_row["days_to_pce"])
            for field in ("pce_headline_yoy", "pce_core_yoy", "pce_headline_mom", "pce_core_mom"):
                if field in last_row and pd.notna(last_row[field]):
                    latest_macro[field] = float(last_row[field])
        analysis["latest_macro"] = latest_macro
        
        # Apply automatic bias correction based on 15-day rolling average (self-learning)
        try:
            latest_resolved = db.gold_predictions_history.find_one(
                {"actual_price": {"$ne": None, "$gt": 0.0}},
                sort=[("date", -1)]
            )
            latest_adjustments = {}
            
            for m_key in ["random_forest", "linear_regression", "mlp", "xgboost", "lstm", "cnn", "ensemble"]:
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
            
        report_progress("Hoàn tất phân tích và sẵn sàng hiển thị", 99)
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

        # Add public FED speeches/Jackson Hole events to the same calendar.
        # These are not FOMC decisions, but they are policy-relevant events.
        events.extend(FED_POLICY_EVENTS_2026)
        
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


def ensure_fed_policy_events():
    """Upsert known FED speeches that can reprice the policy path."""
    operations = [
        UpdateOne(
            {"date": event["date"], "event": event["event"]},
            {"$set": event},
            upsert=True,
        )
        for event in FED_POLICY_EVENTS_2026
    ]
    if operations:
        db.us_economic_calendar.bulk_write(operations)


_CALENDAR_TIME_FIELDS = (
    "datetime",
    "release_datetime",
    "event_datetime",
    "release_time",
    "event_time",
    "time",
)


def _calendar_event_time_key(event):
    """Return a sortable HH:MM:SS key when an event carries a release time."""
    for field in _CALENDAR_TIME_FIELDS:
        value = event.get(field)
        if value in (None, ""):
            continue
        if isinstance(value, datetime):
            return value.strftime("%H:%M:%S")
        match = re.search(r"(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?", str(value))
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            second = int(match.group(3) or 0)
            if 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59:
                return f"{hour:02d}:{minute:02d}:{second:02d}"
    # Events without a published time stay at the end of their date instead
    # of being interleaved ahead of events that have an exact release time.
    return "23:59:59"


def _sort_calendar_events(events, descending=False):
    """Sort economic events chronologically by date, then release time."""
    sorted_events = sorted(
        events,
        key=lambda event: (
            str(event.get("date") or "9999-12-31"),
            _calendar_event_time_key(event),
            str(event.get("event") or ""),
        ),
        reverse=descending,
    )
    return sorted_events


def _fetch_bls_series(series_id, start_year, end_year):
    """Fetch one BLS public series without requiring an API key."""
    url = (
        f"https://api.bls.gov/publicAPI/v2/timeseries/data/{series_id}"
        f"?startyear={start_year}&endyear={end_year}"
    )
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=12,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS không trả dữ liệu cho {series_id}")

    series = payload.get("Results", {}).get("series", [])
    if not series:
        return {}

    values = {}
    for row in series[0].get("data", []):
        period = row.get("period", "")
        if not period.startswith("M") or not period[1:].isdigit():
            continue
        try:
            values[(int(row["year"]), int(period[1:]))] = {
                "value": float(row["value"]),
                "footnotes": [
                    footnote.get("code")
                    for footnote in row.get("footnotes", [])
                    if footnote.get("code")
                ],
            }
        except (KeyError, TypeError, ValueError):
            continue
    return values


def _fetch_fred_series(series_id, start_date, end_date):
    """Fetch a daily FRED CSV series; FRED graph CSV needs no API key."""
    import csv
    import io
    import urllib.error
    import urllib.request

    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        f"&cosd={start_date}&coed={end_date}"
    )
    csv_text = None
    last_error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=20) as response:
                csv_text = response.read().decode("utf-8")
            break
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt == 2:
                raise
    if csv_text is None:
        raise RuntimeError(f"Không thể tải chuỗi FRED {series_id}: {last_error}")

    values = {}
    for row in csv.DictReader(io.StringIO(csv_text)):
        date_str = row.get("observation_date")
        raw_value = row.get(series_id)
        if not date_str or raw_value in (None, "", "."):
            continue
        try:
            values[date_str] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return values


def _parse_fed_rate_token(token):
    """Convert Fed notation such as 3-1/2 or 3.50 to a float."""
    token = token.strip().replace("‑", "-").replace("–", "-")
    if "/" in token and "-" in token:
        whole, fraction = token.split("-", 1)
        numerator, denominator = fraction.split("/", 1)
        return float(whole) + (float(numerator) / float(denominator))
    return float(token)


def _fetch_fed_decision_result(event):
    """Read the actual FOMC action/range from the official Fed release."""
    event_date = event.get("date", "")
    url = f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{event_date.replace('-', '')}a.htm"
    try:
        from bs4 import BeautifulSoup

        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8,
        )
        response.raise_for_status()
        text = BeautifulSoup(response.text, "html.parser").get_text(" ", strip=True)
        text = text.replace("‑", "-").replace("–", "-").replace("â", "-")

        range_match = re.search(
            r"target range for the federal funds rate\s+(?:at|to)\s+"
            r"([0-9]+(?:-[0-9]+/[0-9]+|\.[0-9]+)?)\s+to\s+"
            r"([0-9]+(?:-[0-9]+/[0-9]+|\.[0-9]+)?)\s+percent",
            text,
            re.IGNORECASE,
        )
        if not range_match:
            return None

        action_match = re.search(
            r"\b(maintain|raise|lower|keep)\b[^.]{0,160}?target range for the federal funds rate",
            text,
            re.IGNORECASE,
        )
        action_map = {"maintain": "Giữ nguyên", "keep": "Giữ nguyên", "raise": "Tăng", "lower": "Giảm"}
        action = action_map.get(action_match.group(1).lower(), "Đã công bố") if action_match else "Đã công bố"
        lower = _parse_fed_rate_token(range_match.group(1))
        upper = _parse_fed_rate_token(range_match.group(2))
        return {
            "value": action,
            "unit": f"{_format_rate(lower)}–{_format_rate(upper)}%",
            "direction": action,
            "release_period": event_date,
            "source": "Federal Reserve",
            "source_url": url,
        }
    except Exception as error:
        print(f"Error fetching FOMC result for {event_date}: {error}")
        return None


def _event_period(event_name):
    """Read the release month/year embedded in CPI/NFP event names."""
    match = re.search(r"\(?(?:Tháng|tháng)\s+(\d{1,2})/(\d{4})\)?", event_name or "")
    if not match:
        return None
    # Return year/month so callers can use the same order as BLS keys.
    return int(match.group(2)), int(match.group(1))


def _month_key(year, month):
    return year * 12 + month


def _previous_month(year, month):
    return (year - 1, 12) if month == 1 else (year, month - 1)


def _format_rate(value):
    return f"{value:.2f}".replace(".", ",")


def _direction_from_difference(value, previous, tolerance=1e-9):
    if value > previous + tolerance:
        return "Tăng"
    if value < previous - tolerance:
        return "Giảm"
    return "Đi ngang"


def _nearest_fred_value(values, date_obj):
    """Return the latest daily observation on or before date_obj."""
    candidates = [
        (datetime.strptime(date_str, "%Y-%m-%d").date(), value)
        for date_str, value in values.items()
    ]
    candidates = [item for item in candidates if item[0] <= date_obj]
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _calendar_result_for_event(event, cpi_series, nfp_series, fed_upper, fed_lower):
    """Build one auditable actual result from the public source series."""
    event_date = datetime.strptime(event["date"], "%Y-%m-%d").date()
    if event_date > datetime.now().date():
        return None

    category = event.get("category")
    if event.get("event_type") == "FED_SPEECH":
        # A speech has no target-range result.  Preserve the sourced tone so
        # the UI and the policy layer can distinguish it from an FOMC hold.
        tone = str(event.get("policy_tone", "neutral")).lower()
        direction = {
            "hawkish": "Hawkish",
            "dovish": "Dovish",
        }.get(tone, "Trung lập")
        return {
            "value": "Đã phát biểu",
            "unit": event.get("policy_tone_label", "Tín hiệu chính sách"),
            "direction": direction,
            "release_period": event["date"],
            "source": event.get("source", "Federal Reserve"),
            "source_url": event.get("source_url", ""),
        }

    if category == "CPI":
        period = _event_period(event.get("event", ""))
        if not period:
            return None
        year, month = period
        current = cpi_series.get((year, month))
        previous_year, previous_month = _previous_month(year, month)
        previous = cpi_series.get((previous_year, previous_month))
        year_ago = cpi_series.get((year - 1, month))
        if not current or not previous or not year_ago:
            return None

        yoy = (current["value"] / year_ago["value"] - 1) * 100
        previous_yoy = (previous["value"] / cpi_series[(year - 1, previous_month)]["value"] - 1) * 100 \
            if (year - 1, previous_month) in cpi_series else None
        return {
            "value": f"{yoy:.1f}%",
            "unit": "YoY",
            "previous": f"{previous_yoy:.1f}%" if previous_yoy is not None else "",
            "direction": _direction_from_difference(yoy, previous_yoy) if previous_yoy is not None else "",
            "release_period": f"{year:04d}-{month:02d}",
            "source": "U.S. Bureau of Labor Statistics",
            "source_url": "https://www.bls.gov/cpi/",
        }

    if category == "Việc làm":
        period = _event_period(event.get("event", ""))
        if not period:
            return None
        year, month = period
        current = nfp_series.get((year, month))
        previous_year, previous_month = _previous_month(year, month)
        previous = nfp_series.get((previous_year, previous_month))
        previous_previous_year, previous_previous_month = _previous_month(previous_year, previous_month)
        previous_previous = nfp_series.get((previous_previous_year, previous_previous_month))
        if not current or not previous or not previous_previous:
            return None

        change = current["value"] - previous["value"]
        previous_change = previous["value"] - previous_previous["value"]
        is_preliminary = "P" in current.get("footnotes", [])
        return {
            "value": f"{change:+.0f} nghìn",
            "unit": "việc làm",
            "previous": f"{previous_change:+.0f} nghìn",
            "direction": _direction_from_difference(change, previous_change),
            "status": "Sơ bộ" if is_preliminary else "Đã điều chỉnh",
            "release_period": f"{year:04d}-{month:02d}",
            "source": "U.S. Bureau of Labor Statistics",
            "source_url": "https://www.bls.gov/news.release/empsit.toc.htm",
        }

    if category == "PCE":
        period = _event_period(event.get("event", ""))
        if not period:
            return None
        year, month = period
        reference_month = f"{year:04d}-{month:02d}"
        pce_doc = db.macro_history.find_one(
            {
                "pce_reference_month": reference_month,
                "pce_release_date": {"$lte": event["date"]},
                "pce_headline_yoy": {"$ne": None},
            },
            {"_id": 0},
            sort=[("pce_release_date", -1), ("date", -1)],
        )
        if not pce_doc:
            return None

        headline_yoy = pce_doc.get("pce_headline_yoy")
        core_yoy = pce_doc.get("pce_core_yoy")
        if headline_yoy is None and core_yoy is None:
            return None

        previous_year, previous_month = _previous_month(year, month)
        previous_doc = db.macro_history.find_one(
            {
                "pce_reference_month": f"{previous_year:04d}-{previous_month:02d}",
                "pce_release_date": {"$lte": event["date"]},
            },
            {"_id": 0},
            sort=[("pce_release_date", -1), ("date", -1)],
        )
        previous_core = previous_doc.get("pce_core_yoy") if previous_doc else None
        direction = (
            _direction_from_difference(float(core_yoy), float(previous_core))
            if core_yoy is not None and previous_core is not None
            else ""
        )
        value = f"{float(headline_yoy):.2f}%" if headline_yoy is not None else "--"
        unit = "YoY"
        if core_yoy is not None:
            unit += f" · Core PCE {float(core_yoy):.2f}%"
        previous = ""
        if previous_doc:
            previous_headline = previous_doc.get("pce_headline_yoy")
            previous_core_value = previous_doc.get("pce_core_yoy")
            previous_parts = []
            if previous_headline is not None:
                previous_parts.append(f"{float(previous_headline):.2f}%")
            if previous_core_value is not None:
                previous_parts.append(f"Core {float(previous_core_value):.2f}%")
            previous = " · ".join(previous_parts)
        return {
            "value": value,
            "unit": unit,
            "previous": previous,
            "direction": direction,
            "release_period": reference_month,
            "source": "U.S. Bureau of Economic Analysis / FRED",
            "source_url": event.get("source_url", "https://fred.stlouisfed.org/series/PCEPI"),
        }

    if category == "FED":
        # Prefer the official release because it contains the actual action
        # (raise/lower/hold), not just the resulting target range.
        official_result = _fetch_fed_decision_result(event)
        if official_result:
            return official_result

        upper = _nearest_fred_value(fed_upper, event_date)
        lower = _nearest_fred_value(fed_lower, event_date)
        previous_date = event_date - timedelta(days=1)
        previous_upper = _nearest_fred_value(fed_upper, previous_date)
        previous_lower = _nearest_fred_value(fed_lower, previous_date)
        if None in (upper, lower, previous_upper, previous_lower):
            return None

        if upper > previous_upper:
            action = "Tăng"
        elif upper < previous_upper:
            action = "Giảm"
        else:
            action = "Giữ nguyên"
        return {
            "value": action,
            "unit": f"{_format_rate(lower)}–{_format_rate(upper)}%",
            "direction": action,
            "previous": f"{_format_rate(previous_lower)}–{_format_rate(previous_upper)}%",
            "release_period": event["date"],
            "source": "Federal Reserve (FRED)",
            "source_url": "https://fred.stlouisfed.org/series/DFEDTARU",
        }

    return None


def refresh_economic_calendar_results(force=False):
    """Synchronize released CPI/NFP/PCE/FOMC values into the calendar.

    The calendar is seeded with dates and rules, but those are not actual
    releases.  This function only writes values returned by BLS/FRED and
    never invents a result for a future or unavailable event.
    """
    now = datetime.now()
    with _calendar_results_lock:
        try:
            schedule_sync = crawl_official_us_economic_calendar(force=force)
            if not force:
                state = db.calendar_sync_state.find_one({"_id": "economic_results"})
                last_attempt = state.get("last_attempt_at") if state else None
                if isinstance(last_attempt, datetime) and now - last_attempt < timedelta(minutes=30):
                    return {
                        "status": "skipped",
                        "updated": 0,
                        "message": "Kết quả đã được kiểm tra trong 30 phút gần đây.",
                        "schedule": schedule_sync,
                    }

            events = list(db.us_economic_calendar.find({}, {"_id": 0}))
            past_events = [
                event for event in events
                if event.get("date") and event["date"] <= now.strftime("%Y-%m-%d")
            ]
            if not past_events:
                return {
                    "status": "success",
                    "updated": 0,
                    "message": "Chưa có sự kiện đã công bố.",
                    "schedule": schedule_sync,
                }

            if any(event.get("category") == "PCE" for event in past_events):
                # PCE values are stored in macro_history with their official
                # release date. Refresh them before resolving the calendar
                # event so the UI does not remain stuck at "Chưa cập nhật".
                update_pce_history_from_fred(days=730)

            periods = [
                _event_period(event.get("event", ""))
                for event in past_events
                if event.get("category") in ("CPI", "Việc làm")
            ]
            periods = [period for period in periods if period]
            min_year = min([period[0] for period in periods] + [now.year - 1])
            max_year = max([period[0] for period in periods] + [now.year])
            source_errors = []
            try:
                cpi_series = _fetch_bls_series("CUSR0000SA0", min_year - 1, max_year)
            except Exception as error:
                source_errors.append(f"BLS CPI: {error}")
                cpi_series = {}
            try:
                nfp_series = _fetch_bls_series("CES0000000001", min_year - 1, max_year)
            except Exception as error:
                source_errors.append(f"BLS NFP: {error}")
                nfp_series = {}

            # FOMC results are read from the official Fed release per event.
            # Leave the FRED maps empty here so a slow FRED endpoint cannot
            # block CPI/NFP or make the calendar request appear stuck.
            fed_upper = {}
            fed_lower = {}

            operations = []
            updated = 0
            sync_timestamp = now.isoformat(timespec="seconds")
            for event in past_events:
                result = _calendar_result_for_event(
                    event,
                    cpi_series,
                    nfp_series,
                    fed_upper,
                    fed_lower,
                )
                if result is None:
                    continue
                operations.append(UpdateOne(
                    {"date": event["date"], "event": event["event"]},
                    {
                        "$set": {
                            "result": result,
                            "result_source": result["source"],
                            "result_updated_at": sync_timestamp,
                        }
                    },
                    upsert=False,
                ))
                updated += 1

            if operations:
                db.us_economic_calendar.bulk_write(operations)

            sync_state = {
                "last_attempt_at": now,
                "last_success_at": now if updated else None,
                "updated": updated,
            }
            if source_errors:
                sync_state["source_errors"] = source_errors
            db.calendar_sync_state.update_one(
                {"_id": "economic_results"},
                {
                    "$set": sync_state,
                    "$unset": {"error": ""},
                },
                upsert=True,
            )

            # Keep the short-lived dashboard cache in sync so a normal reload
            # shows the new result without waiting for the next price crawl.
            calendar = _sort_calendar_events(
                list(db.us_economic_calendar.find({}, {"_id": 0}))
            )
            db.gold_prices.update_one(
                {"_id": "latest_prices"},
                {"$set": {"calendar": calendar}},
            )
            return {
                "status": "partial" if source_errors else "success",
                "updated": updated,
                "checked_at": sync_timestamp,
                "source_errors": source_errors,
                "schedule": schedule_sync,
                "calendar": calendar,
            }
        except Exception as error:
            print(f"Error refreshing economic calendar results: {error}")
            try:
                db.calendar_sync_state.update_one(
                    {"_id": "economic_results"},
                    {"$set": {"last_attempt_at": now, "error": str(error)}},
                    upsert=True,
                )
            except Exception:
                pass
            return {"status": "error", "updated": 0, "message": str(error)}


def get_us_economic_calendar():
    """
    Retrieves all seeded economic calendar events from MongoDB, sorted by date
    and release time when the source provides one.
    """
    try:
        # Existing deployments already have a populated collection, so this
        # must run even when the original seed step was skipped.
        ensure_fed_policy_events()
        crawl_official_us_economic_calendar()
        # Refresh released values on the first dashboard request and then use
        # the 30-minute DB throttle to keep page loads fast.
        refresh_economic_calendar_results()
        events = _sort_calendar_events(
            list(db.us_economic_calendar.find({}, {"_id": 0}))
        )
        # Keep one stable API shape; upcoming events intentionally remain null
        # instead of showing guessed values.
        for event in events:
            event.setdefault("result", None)
        return events
    except Exception as e:
        print(f"Error getting economic calendar: {e}")
        return []


FEATURED_GOLD_ARTICLES = [
    {
        "url": "https://thanhnien.vn/gia-vang-bac-giam-sau-luc-ban-chot-loi-185260826203133635.htm",
        "source": "Thanh Niên",
    }
]


def analyze_gold_news_impact(title, text):
    """Extract deterministic, auditable gold-impact signals from news text.

    This is deliberately a rule/context layer, not a fabricated probability.
    It records the time horizon because profit-taking can be bearish in the
    short term while USD weakness or geopolitical risk remains supportive over
    a medium/long horizon.
    """
    import re

    full_text = f"{title} {text}".lower()
    drivers = []
    rules = []
    facts = []
    macro_values = {}
    short_term_bias = "neutral"
    medium_term_bias = "neutral"

    if any(term in full_text for term in ("chốt lời", "bán chốt lời", "hiện thực hóa lợi nhuận")):
        short_term_bias = "bearish"
        drivers.append("Lực bán chốt lời sau giai đoạn tăng mạnh")
        rules.append("Giá tăng nóng/quá mua → nhà đầu tư khóa lợi nhuận → áp lực giảm và điều chỉnh ngắn hạn")

    if any(term in full_text for term in ("dữ liệu kinh tế khả quan", "đơn đặt hàng hàng hóa lâu bền", "gdp quý 2", "pce cốt lõi")):
        short_term_bias = "bearish"
        drivers.append("Dữ liệu kinh tế Mỹ tương đối tích cực làm giảm kỳ vọng nới lỏng tiền tệ")
        rules.append("Dữ liệu Mỹ tốt → kỳ vọng lãi suất/lợi suất tăng → chi phí cơ hội nắm giữ vàng tăng → vàng chịu áp lực")

    if "đơn đặt hàng hàng hóa lâu bền" in full_text:
        facts.append("Đơn hàng lâu bền tháng 7: +1,1%, cao hơn dự báo +0,5%")
    if "gdp" in full_text and "tăng trưởng 1,5%" in full_text:
        facts.append("GDP quý 2 tăng trưởng 1,5%")
    if "pce cốt lõi" in full_text:
        facts.append("PCE cốt lõi tháng 7 tăng 0,2%")
        pce_core_mom_match = re.search(r"pce cốt lõi[^%]{0,100}?([0-9]+[,.][0-9]+)%", full_text)
        if pce_core_mom_match:
            macro_values["pce_core_mom"] = float(pce_core_mom_match.group(1).replace(",", "."))
    if "trong 12 tháng qua" in full_text and "3,7%" in full_text:
        facts.append("Lạm phát 12 tháng ở mức 3,7%, cao hơn dự báo 3,6%")
    pce_release_match = re.search(
        r"(?:pce|lạm phát)[^.]{0,260}?([0-9]+[,.][0-9]+)%[^.]{0,160}?"
        r"(?:dự báo|ước tính)[^%]{0,80}?([0-9]+[,.][0-9]+)%",
        full_text,
    )
    if "pce" in full_text and pce_release_match:
        actual_yoy = float(pce_release_match.group(1).replace(",", "."))
        forecast_yoy = float(pce_release_match.group(2).replace(",", "."))
        macro_values["pce_headline_yoy"] = actual_yoy
        macro_values["pce_forecast_yoy"] = forecast_yoy
        macro_values["pce_surprise_yoy"] = float(actual_yoy - forecast_yoy)

    support_match = re.search(r"hỗ trợ[^0-9]{0,40}(4[.,]?[0-9]{3})", full_text)
    if support_match:
        support_level = support_match.group(1).replace('.', ',')
        rules.append(f"Giữ trên vùng hỗ trợ ${support_level}/ounce → thiên về điều chỉnh kỹ thuật; thủng hỗ trợ → rủi ro bán tiếp diễn")
    else:
        support_level = None

    if any(term in full_text for term in ("usd yếu", "niềm tin vào usd suy yếu", "địa chính trị", "trú ẩn")):
        medium_term_bias = "supportive"
        drivers.append("USD suy yếu hoặc rủi ro địa chính trị vẫn tạo nhu cầu trú ẩn")
        rules.append("USD yếu/rủi ro địa chính trị tăng → dòng tiền phòng thủ tăng → hỗ trợ vàng trung–dài hạn")

    if not drivers and not facts and not rules:
        return None

    if short_term_bias == "bearish" and medium_term_bias == "supportive":
        label = "Giảm ngắn hạn · Hỗ trợ trung–dài hạn"
    elif short_term_bias == "bearish":
        label = "Áp lực giảm ngắn hạn"
    elif medium_term_bias == "supportive":
        label = "Hỗ trợ tăng trung–dài hạn"
    else:
        label = "Tác động hỗn hợp"

    return {
        "label": label,
        "short_term_bias": short_term_bias,
        "medium_term_bias": medium_term_bias,
        "drivers": drivers,
        "facts": facts,
        "rules": rules,
        "support_level": support_level,
        "macro_values": macro_values,
    }


def fetch_featured_gold_news():
    """Fetch curated gold articles and attach structured impact context."""
    import re
    from bs4 import BeautifulSoup

    featured = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    for article in FEATURED_GOLD_ARTICLES:
        try:
            response = requests.get(article["url"], headers=headers, timeout=10)
            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            title_meta = soup.select_one('meta[property="og:title"]')
            description_meta = soup.select_one('meta[property="og:description"]')
            content_node = soup.select_one("div.detail-content, div.detail-cmain, article")
            title = (title_meta.get("content") if title_meta else None) or (
                soup.select_one("h1").get_text(" ", strip=True) if soup.select_one("h1") else ""
            )
            description = (description_meta.get("content") if description_meta else "") or ""
            article_text = content_node.get_text(" ", strip=True) if content_node else description
            if not title or not article_text:
                continue

            published_match = re.search(r"\b(\d{1,2}:\d{2})\s+(\d{2}/\d{2}/\d{4})\b", soup.get_text(" ", strip=True))
            published = f"{published_match.group(1)} {published_match.group(2)}" if published_match else ""
            text_for_keywords = f"{title} {description} {article_text}".lower()
            keyword_terms = [
                "vàng", "bạc", "chốt lời", "lãi suất", "fed", "gdp", "pce",
                "lạm phát", "usd", "hỗ trợ", "trú ẩn"
            ]
            keywords = [term for term in keyword_terms if term in text_for_keywords]
            impact = analyze_gold_news_impact(title, article_text)
            featured.append({
                "title": title.strip(),
                "description": description.strip()[:500],
                "link": article["url"],
                "time": published,
                "keywords": keywords,
                "source": article["source"],
                "featured": True,
                "impact_analysis": impact,
            })
        except Exception as error:
            print(f"Error fetching featured gold article: {error}")
    return featured


def collect_gold_market_signals(news):
    """Return compact structured signals for the dashboard/model context."""
    signals = []
    for item in news or []:
        impact = item.get("impact_analysis")
        if not impact:
            continue
        signals.append({
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "source": item.get("source", ""),
            "time": item.get("time", ""),
            **impact,
        })
    return signals[:8]


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
            return fetch_featured_gold_news()
            
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
                    
                news_item = {
                    "title": title,
                    "description": description,
                    "link": link,
                    "time": time_str,
                    "keywords": matches,
                    "source": "VnExpress",
                    "impact_analysis": analyze_gold_news_impact(title, description),
                }
                filtered_news.append(news_item)

        # Put the curated article first so the current market explanation is
        # visible even when the RSS feed is noisy or temporarily unavailable.
        featured_news = fetch_featured_gold_news()
        seen_links = set()
        combined_news = []
        for item in featured_news + filtered_news:
            if item.get("link") and item["link"] in seen_links:
                continue
            if item.get("link"):
                seen_links.add(item["link"])
            combined_news.append(item)

        # Limit to top 15 news articles
        return combined_news[:15]
    except Exception as e:
        print(f"Error fetching macro news: {e}")
        return fetch_featured_gold_news()


def fetch_us_macro_indicators():
    """
    Fetches US macroeconomic indicators from FRED:
    - CPI: CPIAUCSL (YoY change)
    - PCE/Core PCE: PCEPI/PCEPILFE (YoY change)
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
                # Older cache documents predate the explicit PCE fields.  A
                # cache without them must be refreshed once after deployment.
                if data and data.get("pce") and data.get("core_pce"):
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
        "pce": {
            "name": "PCE (Lạm phát ưa thích của FED)",
            "value": None,
            "prev_value": None,
            "date": None,
            "unit": "%/năm",
            "source": "FRED - fred.stlouisfed.org",
            "source_url": "https://fred.stlouisfed.org/series/PCEPI",
            "trend": None,
            "gold_impact": "PCE cao → FED cứng rắn hơn → vàng chịu áp lực ngắn hạn",
            "impact_direction": "opposite",
            "description": "Thước đo lạm phát được FED ưu tiên theo dõi"
        },
        "core_pce": {
            "name": "Core PCE (PCE lõi)",
            "value": None,
            "prev_value": None,
            "date": None,
            "unit": "%/năm",
            "source": "FRED - fred.stlouisfed.org",
            "source_url": "https://fred.stlouisfed.org/series/PCEPILFE",
            "trend": None,
            "gold_impact": "Core PCE cao → kỳ vọng giữ lãi suất cao lâu hơn → vàng giảm",
            "impact_direction": "opposite",
            "description": "Lạm phát PCE loại trừ nhóm thực phẩm và năng lượng"
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

    # --- 1b. Fetch PCE/Core PCE (YoY) ---
    for indicator_key, series_id in (("pce", "PCEPI"), ("core_pce", "PCEPILFE")):
        df_pce = fetch_fred_series(series_id)
        if df_pce is not None and len(df_pce) >= 13:
            df_pce["prev_year_value"] = df_pce["value"].shift(12)
            df_pce["yoy"] = (df_pce["value"] / df_pce["prev_year_value"] - 1) * 100
            df_pce = df_pce.dropna()
            if not df_pce.empty:
                latest = df_pce.iloc[-1]
                prev = df_pce.iloc[-2]
                indicators[indicator_key]["value"] = round(float(latest["yoy"]), 2)
                indicators[indicator_key]["prev_value"] = round(float(prev["yoy"]), 2)
                indicators[indicator_key]["date"] = latest["date"][:7]

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


def build_fed_policy_outlook(
    macro_indicators=None,
    macro_history=None,
    calendar=None,
    market_signals=None,
):
    """Build an auditable FED-policy nowcast from released macro inputs.

    This is intentionally a transparent policy layer, not a claim that the
    FED can be predicted with certainty.  It converts inflation, labour,
    market and recent FED-speech signals into action probabilities that the
    gold model can consume and the UI can explain.
    """
    indicators = macro_indicators if isinstance(macro_indicators, dict) else {}

    def safe_float(value):
        try:
            number = float(value)
            return number if np.isfinite(number) else None
        except (TypeError, ValueError):
            return None

    def clamp(value, lower=-1.0, upper=1.0):
        return float(np.clip(float(value), lower, upper))

    def indicator_value(key, field="value"):
        item = indicators.get(key) or {}
        if not isinstance(item, dict):
            return None
        return safe_float(item.get(field))

    if isinstance(macro_history, pd.DataFrame):
        history_rows = macro_history.to_dict("records")
    elif isinstance(macro_history, dict):
        history_rows = [macro_history]
    else:
        history_rows = list(macro_history or [])
    history_rows = [row for row in history_rows if isinstance(row, dict)]
    history_rows.sort(key=lambda row: str(row.get("date", "")))
    latest_row = history_rows[-1] if history_rows else {}
    previous_row = history_rows[-2] if len(history_rows) > 1 else {}

    def history_value(key, row=latest_row):
        return safe_float(row.get(key))

    def history_change(key):
        latest = history_value(key, latest_row)
        previous = history_value(key, previous_row)
        return latest - previous if latest is not None and previous is not None else None

    def choose(indicator_key, history_key=None, field="value"):
        value = indicator_value(indicator_key, field)
        if value is not None:
            return value
        return history_value(history_key or indicator_key)

    cpi_yoy = choose("cpi", "cpi_yoy")
    pce_yoy = choose("pce", "pce_headline_yoy")
    core_pce_yoy = choose("core_pce", "pce_core_yoy")
    core_pce_mom = history_value("pce_core_mom")
    nfp_change = choose("nonfarm", "nfp_change")
    unemployment = choose("unemployment", "unemployment")
    fed_rate = choose("fed_rate", "fed_rate")
    dxy = history_value("dxy")
    us10y = history_value("us10y")
    real_yield = history_value("real_yield")
    vix = history_value("vix")
    brent = history_value("brent")

    # Positive score means more restrictive policy pressure.  Missing values
    # are omitted rather than replaced by a fabricated macro reading.
    components = []

    def add_component(name, score, weight):
        if score is not None:
            components.append({
                "name": name,
                "score": clamp(score),
                "weight": float(weight),
            })

    inflation_scores = []
    if core_pce_yoy is not None:
        core_gap = clamp((core_pce_yoy - 2.0) / 1.2)
        add_component("Core PCE so với mục tiêu 2%", core_gap, 0.30)
        inflation_scores.append(core_gap)
    if pce_yoy is not None:
        pce_gap = clamp((pce_yoy - 2.0) / 1.5)
        add_component("PCE so với mục tiêu 2%", pce_gap, 0.15)
        inflation_scores.append(pce_gap)
    if cpi_yoy is not None:
        cpi_gap = clamp((cpi_yoy - 2.0) / 1.5)
        add_component("CPI so với mục tiêu 2%", cpi_gap, 0.10)
        inflation_scores.append(cpi_gap)
    if core_pce_mom is not None:
        monthly_pressure = clamp((core_pce_mom - (2.0 / 12.0)) / 0.15)
        add_component("Động lượng Core PCE tháng", monthly_pressure, 0.10)
        inflation_scores.append(monthly_pressure)

    inflation_score = float(np.mean(inflation_scores)) if inflation_scores else 0.0

    if nfp_change is not None:
        add_component("NFP thay đổi hàng tháng", clamp((nfp_change - 150.0) / 180.0), 0.08)
    if unemployment is not None:
        add_component("Tỷ lệ thất nghiệp", clamp((4.5 - unemployment) / 1.0), 0.08)

    # Market moves confirm whether the restrictive/easing pressure is already
    # being priced. These are confirmation inputs, not a replacement for data.
    real_yield_change = history_change("real_yield")
    us10y_change = history_change("us10y")
    dxy_change = history_change("dxy")
    market_scores = []
    if real_yield_change is not None:
        market_scores.append(clamp(real_yield_change / 0.15))
    if us10y_change is not None:
        market_scores.append(clamp(us10y_change / 0.20))
    if dxy_change is not None and dxy is not None:
        market_scores.append(clamp((dxy_change / max(abs(dxy), 1e-6)) / 0.005))
    if market_scores:
        add_component("Lợi suất thực/US10Y/DXY biến động", float(np.mean(market_scores)), 0.12)

    today = datetime.now().date()
    past_fed_events = []
    upcoming_fed_events = []
    for event in calendar or []:
        if not isinstance(event, dict) or event.get("category") != "FED":
            continue
        try:
            event_date = datetime.strptime(str(event.get("date")), "%Y-%m-%d").date()
        except (TypeError, ValueError):
            continue
        if event_date <= today:
            past_fed_events.append((event_date, event))
        else:
            upcoming_fed_events.append((event_date, event))
    past_fed_events.sort(key=lambda item: item[0], reverse=True)
    upcoming_fed_events.sort(key=lambda item: item[0])

    latest_fed_event = past_fed_events[0][1] if past_fed_events else None
    latest_fed_date = past_fed_events[0][0] if past_fed_events else None
    speech_tone = str((latest_fed_event or {}).get("policy_tone", "neutral")).lower()
    speech_score = {"hawkish": 1.0, "dovish": -1.0}.get(speech_tone, 0.0)
    speech_is_recent = bool(latest_fed_date and (today - latest_fed_date).days <= 14)
    if speech_is_recent and speech_score:
        add_component("Phát biểu FED gần nhất", speech_score, 0.17)

    total_weight = sum(item["weight"] for item in components)
    policy_score = float(sum(item["score"] * item["weight"] for item in components))
    policy_score = clamp(policy_score)
    coverage = float(min(1.0, total_weight))

    # These are scenario probabilities, not official FED guidance. They are
    # deliberately conservative: a strong score still leaves a large hold
    # probability because the next meeting outcome is uncertain.
    hike_probability = 0.07 + 0.25 * max(policy_score, 0.0)
    cut_probability = 0.07 + 0.25 * max(-policy_score, 0.0)
    hold_probability = max(0.05, 1.0 - hike_probability - cut_probability)
    probability_total = hike_probability + hold_probability + cut_probability
    probabilities = {
        "hike": float(hike_probability / probability_total),
        "hold": float(hold_probability / probability_total),
        "cut": float(cut_probability / probability_total),
    }
    action_key = max(probabilities, key=probabilities.get)
    action_labels = {"hike": "Tăng lãi suất", "hold": "Giữ nguyên", "cut": "Giảm lãi suất"}
    bias_label = "Hawkish" if policy_score >= 0.12 else "Dovish" if policy_score <= -0.12 else "Trung lập"
    confidence_pct = float(np.clip((35.0 + 45.0 * abs(policy_score)) * coverage, 10.0, 90.0))

    # Gold usually reacts negatively to restrictive policy pressure in the
    # short run. Inflation can support gold over a longer horizon, so expose
    # both effects instead of collapsing them into a simplistic rule.
    gold_score = clamp(-0.75 * policy_score + 0.18 * max(inflation_score, 0.0))
    gold_label = "Tăng" if gold_score >= 0.12 else "Giảm" if gold_score <= -0.12 else "Trung lập"

    drivers = []
    if core_pce_yoy is not None:
        drivers.append(f"Core PCE {core_pce_yoy:.2f}% so với mục tiêu 2%")
    if cpi_yoy is not None:
        drivers.append(f"CPI {cpi_yoy:.2f}%")
    if nfp_change is not None and unemployment is not None:
        drivers.append(f"NFP {nfp_change:+.0f} nghìn, thất nghiệp {unemployment:.1f}%")
    if speech_is_recent and latest_fed_event:
        drivers.append(
            f"{latest_fed_event.get('event', 'Phát biểu FED')}: "
            f"{latest_fed_event.get('policy_tone_label', speech_tone)}"
        )
    if real_yield_change is not None or us10y_change is not None:
        drivers.append("Lợi suất thực/US10Y được dùng để xác nhận áp lực chính sách")

    inputs = [
        {"key": "cpi_yoy", "label": "CPI YoY", "value": cpi_yoy, "unit": "%", "impact": "Cao hơn → hawkish"},
        {"key": "pce_yoy", "label": "PCE YoY", "value": pce_yoy, "unit": "%", "impact": "Cao hơn → hawkish"},
        {"key": "core_pce_yoy", "label": "Core PCE YoY", "value": core_pce_yoy, "unit": "%", "impact": "Cao hơn → hawkish"},
        {"key": "core_pce_mom", "label": "Core PCE MoM", "value": core_pce_mom, "unit": "%", "impact": "Động lượng lạm phát"},
        {"key": "fed_rate", "label": "Fed Funds hiện tại", "value": fed_rate, "unit": "%", "impact": "Mức nền chính sách"},
        {"key": "nfp_change", "label": "NFP thay đổi", "value": nfp_change, "unit": "nghìn", "impact": "Việc làm mạnh → hawkish"},
        {"key": "unemployment", "label": "Thất nghiệp", "value": unemployment, "unit": "%", "impact": "Thất nghiệp thấp → hawkish"},
        {"key": "dxy", "label": "DXY", "value": dxy, "unit": "điểm", "impact": "Tăng → gây áp lực vàng"},
        {"key": "us10y", "label": "US10Y", "value": us10y, "unit": "%", "impact": "Tăng → gây áp lực vàng"},
        {"key": "real_yield", "label": "Real yield", "value": real_yield, "unit": "%", "impact": "Tăng → gây áp lực vàng"},
        {"key": "vix", "label": "VIX", "value": vix, "unit": "điểm", "impact": "Tăng → hỗ trợ trú ẩn"},
        {"key": "brent", "label": "Brent", "value": brent, "unit": "USD/thùng", "impact": "Tác động lạm phát gián tiếp"},
    ]
    for item in inputs:
        item["available"] = item["value"] is not None
        if item["value"] is not None:
            item["value"] = float(item["value"])

    next_event = upcoming_fed_events[0] if upcoming_fed_events else None
    latest_event_payload = None
    if latest_fed_event:
        latest_event_payload = dict(latest_fed_event)
        latest_event_payload["date"] = str(latest_fed_date)

    return {
        "as_of": datetime.now().isoformat(timespec="seconds"),
        "method": "transparent_macro_policy_nowcast",
        "coverage": coverage,
        "confidence_pct": confidence_pct,
        "inflation": {
            "score": inflation_score,
            "label": "Áp lực tăng" if inflation_score >= 0.12 else "Đang hạ nhiệt" if inflation_score <= -0.12 else "Ổn định",
            "target": 2.0,
            "cpi_yoy": cpi_yoy,
            "pce_yoy": pce_yoy,
            "core_pce_yoy": core_pce_yoy,
            "core_pce_mom": core_pce_mom,
        },
        "fed_action": {
            "primary": action_labels[action_key],
            "primary_key": action_key,
            "bias": bias_label,
            "score": policy_score,
            "probabilities": probabilities,
            "horizon": "kỳ họp FED kế tiếp",
        },
        "gold_implication": {
            "label": gold_label,
            "score": gold_score,
            "confidence_pct": float(np.clip(abs(gold_score) * 100.0 * coverage, 5.0, 90.0)),
            "horizon": "ngắn hạn theo kỳ vọng lãi suất",
        },
        "inputs": inputs,
        "drivers": drivers[:6],
        "components": components,
        "latest_fed_event": latest_event_payload,
        "next_fed_event": {
            **dict(next_event[1]),
            "days": int((next_event[0] - today).days),
        } if next_event else None,
        "market_confirmation": {
            "dxy_change": dxy_change,
            "us10y_change": us10y_change,
            "real_yield_change": real_yield_change,
        },
    }


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
            df["world_price"] = df["world_price"].replace(0.0, np.nan).ffill()
            df["close"] = df["world_price"]
            df["open"] = df["world_price"]
            df["high"] = df["world_price"]
            df["low"] = df["world_price"]
            df["volume"] = 1.0
            
            macro_cursor = db.macro_history.find({}, {"_id": 0})
            macro_df = pd.DataFrame(list(macro_cursor))
            if not macro_df.empty and "date" in macro_df.columns:
                df = pd.merge(df, macro_df, on="date", how="left")
                cols = [c for c in ["dxy", "us10y", "vix", "brent", "dji", "spx", "eurusd", "xagusd", "real_yield", "gld", "gld_trust"] if c in df.columns]
                df[cols] = df[cols].ffill()
                
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
                    item["explanations"] = {m: "Tín hiệu kỹ thuật & vĩ mô đi ngang." for m in ["random_forest", "linear_regression", "mlp", "xgboost", "lstm", "cnn", "ensemble"]}
    except Exception as e:
        print(f"Error generating explanations for prediction history: {e}")
        for item in history:
            if "explanations" not in item:
                item["explanations"] = {m: "Không có phân tích." for m in ["random_forest", "linear_regression", "mlp", "xgboost", "lstm", "cnn", "ensemble"]}
                
    return history
