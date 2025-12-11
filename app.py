import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import time
import requests
from bs4 import BeautifulSoup
import numpy as np

# ==========================================
# 1. 資料抓取函數 (技術面)
# ==========================================
@st.cache_data(ttl=3600)
def get_stock_data_v3(stock_code):
    stock_code = str(stock_code).strip()
    suffixes = [".TW", ".TWO"]
    df = pd.DataFrame()
    found_ticker = ""
    for suffix in suffixes:
        try:
            ticker = f"{stock_code}{suffix}"
            stock = yf.Ticker(ticker)
            temp_df = stock.history(period="500d", auto_adjust=False)
            if not temp_df.empty:
                df = temp_df
                found_ticker = ticker
                break
            time.sleep(0.5) 
        except Exception:
            continue
    if df.empty: return pd.DataFrame(), ""
    try:
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        df.columns = [str(c).lower() for c in df.columns]
        df.index.name = 'date'
        return df, found_ticker
    except Exception: return pd.DataFrame(), ""

# ==========================================
# 2. 獲取公司名稱
# ==========================================
@st.cache_data(ttl=86400)
def get_stock_name(stock_code):
    code = str(stock_code).strip()
    stock_map = {
        "0050": "元大台灣50", "0056": "元大高股息", "00878": "國泰永續高股息", "00929": "復華台灣科技優息",
        "2330": "台積電", "2454": "聯發科", "2303": "聯電", "2317": "鴻海",
        "2308": "台達電", "3711": "日月光投控", "2382": "廣達", "3231": "緯創",
        "6669": "緯穎", "2357": "華碩", "2356": "英業達", "3008": "大立光",
        "3034": "聯詠", "2379": "瑞昱", "3037": "欣興", "2603": "長榮", "2609": "陽明",
        "2615": "萬海", "2618": "長榮航", "2610": "華航", "2002": "中鋼",
        "2881": "富邦金", "2882": "國泰金", "2891": "中信金"
    }
    if code in stock_map: return stock_map[code]
    try:
        url = f"https://tw.stock.yahoo.com/quote/{code}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            title = soup.title.string
            if title and "(" in title: return title.split("(")[0].strip()
            return title
    except Exception: pass
    return code

# ==========================================
# 3. 指標計算
# ==========================================
def calculate_indicators(df):
    df = df.copy()
    try:
        # MA & Volume MA
        if len(df) >= 5: df['MA5'] = df['close'].rolling(5).mean()
        if len(df) >= 20: df['MA20'] = df['close'].rolling(20).mean()
        if len(df) >= 60: df['MA60'] = df['close'].rolling(60).mean()
        if len(df) >= 5: df['VolMA5'] = df['volume'].rolling(5).mean()

        # KD & MACD & RSI & BB & BBW
        rsv_min = df['low'].rolling(9).min()
        rsv_max = df['high'].rolling(9).max()
        rsv_den = rsv_max - rsv_min
        rsv_den[rsv_den == 0] = 1 
        df['RSV'] = (df['close'] - rsv_min) / rsv_den * 100
        df['K'] = df['RSV'].ewm(com=2).mean()
        df['D'] = df['K'].ewm(com=2).mean()
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal']
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['BB_Mid'] = df['close'].rolling(window=20).mean()
        df['BB_Std'] = df['close'].rolling(window=20).std()
        df['BB_Up'] = df['BB_Mid'] + 2 * df['BB_Std']
        df['BB_Low'] = df['BB_Mid'] - 2 * df['BB_Std']
        df['BBW'] = (df['BB_Up'] - df['BB_Low']) / df['BB_Mid']
        
        # OBV & ADX
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['UpMove'] = df['high'] - df['high'].shift(1)
        df['DownMove'] = df['low'].shift(1) - df['low']
        df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
        df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)
        df['TR'] = np.where((df['high'] - df['low']) > (df['high'] - df['close'].shift(1)).abs(),
                             np.where((df['high'] - df['low']) > (df['low'] - df['close'].shift(1)).abs(),
                                      df['high'] - df['low'], (df['low'] - df['close'].shift(1)).abs()),
                             (df['high'] - df['close'].shift(1)).abs()).fillna(0)
        n = 14
        df['ATR'] = df['TR'].ewm(span=n, adjust=False).mean()
        df['+DM_EMA'] = df['+DM'].ewm(span=n, adjust=False).mean()
        df['-DM_EMA'] = df['-DM'].ewm(span=n, adjust=False).mean()
        df['+DI'] = (df['+DM_EMA'] / df['ATR']) * 100
        df['-DI'] = (df['-DM_EMA'] / df['ATR']) * 100
        df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['+DI'])) * 100 # 修正分母
        df['ADX'] = df['DX'].ewm(span=n, adjust=False).mean()
        
        # 量能趨勢
        df['Vol_Shift1'] = df['volume'].shift(1)
        df['Vol_Shift2'] = df['volume'].shift(2)
        df['Vol_Inc'] = (df['volume'] > df['Vol_Shift1']) & (df['Vol_Shift1'] > df['Vol_Shift2'])
        df['Vol_Dec'] = (df['volume'] < df['Vol_Shift1']) & (df['Vol_Shift1'] < df['Vol_Shift2'])
        
        # --- 新增：ATR 波動度 (近 20 日平均) ---
        df['ATR_Avg'] = df['ATR'].tail(20).mean()

    except Exception as e:
        print(f"指標計算錯誤: {e}")
        pass
    return df

# ==========================================
# 4. 策略與分析 (納入 ADX 濾鏡和 ATR 波動度)
# ==========================================
def calculate_score(df):
    score = 50 
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 趨勢分數 (40%)
    if last['close'] > last['MA20']: score += 10 
    if last['MA20'] > last['MA60']: score += 10
    if last['close'] > last['MA60']: score += 10
    
    # ADX 濾鏡：只在趨勢強烈時給予動能指標高加權
    adx_filter = last['ADX'] > 25 if 'ADX' in df.columns and not pd.isna(last['ADX']) else True
    
    # 動能分數 (30%)
    if last['MACD'] > 0 and adx_filter: score += 5
    if last['Hist'] > 0 and adx_filter: score += 5
    if last['K'] > last['D'] and adx_filter: score += 5
    
    # 量價分數 (20%)
    vol_ratio = last['volume'] / last['VolMA5'] if 'VolMA5' in df.columns else 1
    if last['close'] > prev['close'] and vol_ratio > 1.2: score += 5 
    if last['close'] < prev['close'] and vol_ratio > 1.2: score -= 5 
    if 'Vol_Inc' in df.columns and last['Vol_Inc'] == True: score += 5
    if 'Vol_Dec' in df.columns and last['Vol_Dec'] == True: score -= 5 
    
    # 突破分數
    if 'BBW' in df.columns and last['BBW'] > df['BBW'].tail(60).quantile(0.85):
        if last['close'] > last['BB_Up']: score = 100 
        
    return max(0, min(100, score))

def analyze_volume(df):
    if 'VolMA5' not in df.columns: return "無量能資料"
    last = df.iloc[-1]
    vol_ratio = last['volume'] / last['VolMA5']
    
    vol_trend_msg = ""
    if 'Vol_Inc' in df.columns and last['Vol_Inc'] == True: vol_trend_msg = "🔥 3日連增"
    elif 'Vol_Dec' in df.columns and last['Vol_Dec'] == True: vol_trend_msg = "❄️ 3日連縮"
    
    status = ""
    if vol_ratio > 1.5: status = "爆量"
    elif vol_ratio > 1.2: status = "放量"
    elif vol_ratio < 0.6: status = "窒息量"
    elif vol_ratio < 0.8: status = "量縮"
    else: status = "量平"

    return f"{status} ({vol_trend_msg if vol_trend_msg else '量能持平'})"

def analyze_signals(df):
    if len(df) < 2: return ["資料不足"]
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []
    
    # --- 新增：波動風險提示 ---
    if 'ATR_Avg' in df.columns and not pd.isna(last['ATR_Avg']):
        current_atr = last['ATR']
        avg_atr = last['ATR_Avg']
        if current_atr > avg_atr * 1.5:
             signals.append(f"🚨 **波動度過高**：ATR({current_atr:.2f})，風險放大，建議減小部位。")
        elif current_atr < avg_atr * 0.5:
             signals.append(f"😴 **波動度極低**：市場極度沉悶，不適合短線操作。")

    # 整理突破訊號
    if 'BBW' in df.columns:
        bbw_avg = df['BBW'].tail(60).mean()
        if last['BBW'] < bbw_avg * 0.8: signals.append("🧘 **低波動整理**：布林通道收斂，等待大行情。")
        elif last['close'] > last['BB_Up'] and last['BBW'] > bbw_avg * 1.2: signals.append("🚀 **趨勢突破確立**：股價創高且布林通道開口放大。")
    
    # 均線趨勢與金死叉
    if 'MA5' in df.columns and 'MA20' in df.columns:
        if last['MA5'] > last['MA20'] > last['MA60']: signals.append("🔥 **趨勢**：多頭排列")
        elif last['MA5'] < last['MA20'] < last['MA60']: signals.append("❄️ **趨勢**：空頭排列")
        if prev['MA5'] < prev['MA20'] and last['MA5'] > last['MA20']: signals.append("✨ **均線金叉**：5日穿月線")
        elif prev['MA5'] > prev['MA20'] and last['MA5'] < last['MA20']: signals.append("💀 **均線死叉**：5日破月線")
        
    # ADX & OBV 整合
    if 'ADX' in df.columns and not pd.isna(last['ADX']):
        adx_val = last['ADX']
        if adx_val > 40: signals.append(f"🚀 **ADX極強 ({adx_val:.1f})**：趨勢爆發，動能最強。")
        elif adx_val > 25
