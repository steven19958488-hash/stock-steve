import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf

# ==========================================
# 1. 資料抓取函數 (回傳 DataFrame 與 股票代號)
# ==========================================
@st.cache_data(ttl=3600)
def get_stock_data_v2(stock_code):
    stock_code = str(stock_code).strip()
    suffixes = [".TW", ".TWO"]
    
    df = pd.DataFrame()
    found_ticker = ""

    for suffix in suffixes:
        try:
            ticker = f"{stock_code}{suffix}"
            temp_df = yf.download(ticker, start="2023-01-01", progress=False)
            
            if not temp_df.empty:
                df = temp_df
                found_ticker = ticker # 記住成功的代號
                break
        except Exception:
            continue

    if df.empty:
        return pd.DataFrame(), ""

    # --- 資料清洗 ---
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.columns = [str(c).lower() for c in df.columns]
        
        df.index.name = 'date'
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        return df, found_ticker
    except Exception as e:
        return pd.DataFrame(), ""

# ==========================================
# 2. 獲取公司名稱 (新增功能)
# ==========================================
@st.cache_data(ttl=86400) # 名稱快取存一天
def get_stock_info(ticker_symbol):
    try:
        t = yf.Ticker(ticker_symbol)
        info = t.info
        # 嘗試抓取長名稱或短名稱
        return info.get('longName') or info.get('shortName') or ticker_symbol
    except:
        return ticker_symbol

# ==========================================
# 3. 指標計算函數
# ==========================================
def calculate_indicators(df):
    df = df.copy()
    try:
        # --- 均線 ---
        if len(
