# find_strong_stocks_action.py
# Single-run version for GitHub Actions (runs once and exits)
# Reads TELEGRAM_TOKEN and TELEGRAM_CHAT_ID from environment variables.

import os
import time
import math
import requests
from datetime import datetime
import pandas as pd
import yfinance as yf
import mplfinance as mpf

# Config (no hardcode token here; use env variables via GitHub Secrets)
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
WORKDIR = "strong_output"
CHART_DIR = os.path.join(WORKDIR, "charts")
os.makedirs(WORKDIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

TWSE_URL = "https://www.twse.com.tw/exchangeReport/MI_INDEX?response=json&type=ALLBUT0999"

def telegram_send_text(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram token/chat id not set; skipping send.")
        return False, "no-token"
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=payload, timeout=15)
        return r.ok, r.text
    except Exception as e:
        return False, str(e)

def telegram_send_photo(path):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram token/chat id not set; skipping photo send.")
        return False, "no-token"
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    try:
        with open(path, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": TELEGRAM_CHAT_ID}
            r = requests.post(url, files=files, data=data, timeout=30)
        return r.ok, r.text
    except Exception as e:
        return False, str(e)

def fetch_twse_table():
    try:
        r = requests.get(TWSE_URL, timeout=20)
        r.raise_for_status()
        js = r.json()
    except Exception as e:
        return None, f"TWSE fetch failed: {e}"

    # Try common JSON shapes
    # 1) top-level has 'data' & 'fields'
    if isinstance(js, dict) and "data" in js and "fields" in js:
        try:
            df = pd.DataFrame(js["data"], columns=js["fields"])
            return df, None
        except Exception:
            pass
    # 2) search nested dicts for 'data'/'fields'
    for k,v in js.items():
        if isinstance(v, dict) and "data" in v and "fields" in v:
            try:
                df = pd.DataFrame(v["data"], columns=v["fields"])
                return df, None
            except Exception:
                pass
    return None, "No suitable table structure found in TWSE JSON."

def extract_top300(df_raw):
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]

    col_map = {}
    for c in df.columns:
        if "證券代號" in c: col_map["code"]=c
        if "證券名稱" in c: col_map["name"]=c
        if "成交股數" in c: col_map["vol"]=c
        if "收盤價" in c: col_map["close"]=c

    if not all(k in col_map for k in ["code","name","vol","close"]):
        return None, f"Missing columns; found {list(col_map.keys())}"

    df = df[[col_map["code"], col_map["name"], col_map["vol"], col_map["close"]]].copy()
    df.columns = ["證券代號","證券名稱","成交股數","收盤價"]
    df["證券名稱"] = df["證券名稱"].astype(str).str.replace(r"<.*?>","",regex=True).str.strip()
    df["證券代號"] = df["證券代號"].astype(str).str.strip()

    def to_num(x):
        s = str(x).replace(",","").replace("--","0").strip()
        try: return float(s)
        except: return 0.0
    df["成交股數"] = df["成交股數"].apply(to_num)
    def to_price(x):
        s = str(x).replace(",","").replace("--","0").strip()
        try: return float(s)
        except: return float("nan")
    df["收盤價"] = df["收盤價"].apply(to_price)

    exclude_name_keywords = ["ETF", "權證", "DR", "受益證券", "基金", "富邦", "元大", "國泰", "群益", "永豐", "台新", "銀行", "金控", "金融", "證券", "保險"]
    mask = ~df["證券名稱"].str.contains("|".join(exclude_name_keywords), na=False)
    df = df[mask].copy()
    df = df.sort_values("成交股數", ascending=False).head(300).reset_index(drop=True)
    df["成交張數"] = (df["成交股數"] / 1000.0).round(3)
    df.to_csv(os.path.join(WORKDIR,"top_volume_stocks.csv"), index=False, encoding="utf-8-sig")
    return df, None

def is_strong_stock(symbol_plain, name):
    symbol = symbol_plain + ".TW"
    try:
        data = yf.download(symbol, period="120d", interval="1d", progress=False, threads=False)
        if data is None or data.empty or "Close" not in data.columns:
            return False, "no-data", None
        data = data.dropna(subset=["Close"]).copy()
        if "Volume" not in data.columns:
            return False, "no-volume", None

        data["MA3"]=data["Close"].rolling(3).mean()
        data["MA5"]=data["Close"].rolling(5).mean()
        data["MA8"]=data["Close"].rolling(8).mean()
        data["MA20"]=data["Close"].rolling(20).mean()
        data["MA60"]=data["Close"].rolling(60).mean()
        data["MA5_vol"]=data["Volume"].rolling(5).mean()

        if len(data) < 10:
            return False, "short-history", data

        last_vols = data["Volume"].tail(3).values
        cond_vols_up = len(last_vols)==3 and (last_vols[0] < last_vols[1] < last_vols[2])
        cond_yesterday_vs_ma5 = False
        if not math.isnan(data["MA5_vol"].iloc[-2]):
            cond_yesterday_vs_ma5 = data["Volume"].iloc[-2] > data["MA5_vol"].iloc[-2] * 2
        close_today = data["Close"].iloc[-1]
        ma5_today = data["MA5"].iloc[-1]
        cond_close_above_ma5 = (not math.isnan(ma5_today)) and (close_today > ma5_today)
        ten_high = data["Close"].tail(10).max()
        cond_close_new10 = (close_today >= ten_high) or (close_today >= ten_high*0.95)
        prev_close = data["Close"].iloc[-2]
        pct = (close_today - prev_close) / prev_close * 100.0
        cond_pct = pct > 3.0

        passed = all([cond_vols_up, cond_yesterday_vs_ma5, cond_close_above_ma5, cond_close_new10, cond_pct])
        reason = {
            "cond_vols_up": cond_vols_up,
            "cond_yesterday_vs_ma5": cond_yesterday_vs_ma5,
            "cond_close_above_ma5": cond_close_above_ma5,
            "cond_close_new10": cond_close_new10,
            "cond_pct": cond_pct,
            "pct": round(pct,2)
        }
        return bool(passed), reason, data
    except Exception as e:
        return False, f"error:{e}", None

def plot_chart(symbol_plain, name, data):
    fname = os.path.join(CHART_DIR, f"{symbol_plain}_{name.replace('/','_')}.png")
    try:
        data = data.copy()
        data.index = pd.to_datetime(data.index)
        mc = mpf.make_marketcolors(up='r', down='g', volume='r', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc, rc={'font.sans-serif':['DejaVu Sans','Microsoft JhengHei']})
        mav=[3,5,8,20,60]
        available=[m for m in mav if len(data)>=m]
        mpf.plot(data, type='candle', mav=available, volume=True, style=s, title=f"{symbol_plain} {name}", figsize=(10,6), savefig=dict(fname=fname,dpi=150))
        return fname, None
    except Exception as e:
        return None, str(e)

def main():
    start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    telegram_send_text(f"🚀 {start} 開始執行台股強勢股篩選...")
    df_raw, err = fetch_twse_table()
    if df_raw is None:
        telegram_send_text(f"❌ TWSE 資料讀取失敗：{err}")
        print("TWSE fetch error:", err)
        return
    df_top, err2 = extract_top300(df_raw)
    if df_top is None:
        telegram_send_text(f"❌ TWSE 解析失敗：{err2}")
        print("parse error:", err2)
        return
    telegram_send_text(f"📊 已取前 300 檔（排除 ETF/金融/DR），共 {len(df_top)} 檔")
    strong=[]
    charts=[]
    for i,row in df_top.iterrows():
        code=row["證券代號"].strip()
        name=row["證券名稱"].strip()
        print(f"[{i+1}/{len(df_top)}] 分析 {code} {name} ...", end=" ")
        ok, reason, hist = is_strong_stock(code, name)
        if ok:
            print("✅")
            strong.append((code,name,reason))
            if hist is not None:
                p,e = plot_chart(code,name,hist)
                if p: charts.append(p)
        else:
            print("❌", reason)
        time.sleep(0.6)
    if not strong:
        telegram_send_text("📈 今日無符合條件之強勢股。")
    else:
        txt = "🔥 強勢股名單：\n" + "\n".join([f"{s[0]} {s[1]}" for s in strong])
        telegram_send_text(txt)
        for p in charts[:5]:
            telegram_send_photo(p)
    if strong:
        pd.DataFrame([{"symbol":s[0],"name":s[1],"reason":str(s[2])} for s in strong]).to_csv(os.path.join(WORKDIR,"strong_stocks.csv"), index=False, encoding="utf-8-sig")

if __name__=="__main__":
    main()
