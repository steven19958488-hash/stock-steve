import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf

# ==========================================
# 1. 資料抓取函數
# ==========================================
@st.cache_data
def get_stock_data(stock_code):
    try:
        ticker = f"{stock_code}.TW"
        df = yf.download(ticker, start="2023-01-01", auto_adjust=False)
        
        if df.empty:
            st.warning(f"找不到 {stock_code} 的資料，請確認代碼。")
            return pd.DataFrame()

        # --- 資料清洗 ---
        if isinstance(df.columns, pd.MultiIndex):
             df.columns = df.columns.get_level_values(0)
        
        df.columns = [str(c).lower() for c in df.columns]
        
        df.index.name = 'date'
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return df

    except Exception as e:
        st.error(f"資料抓取失敗: {e}")
        return pd.DataFrame()

# ==========================================
# 2. 指標計算函數
# ==========================================
def calculate_indicators(df):
    # --- 均線 (MA) ---
    if len(df) >= 5: df['MA5'] = df['close'].rolling(window=5).mean()
    if len(df) >= 10: df['MA10'] = df['close'].rolling(window=10).mean()
    if len(df) >= 20: df['MA20'] = df['close'].rolling(window=20).mean()
    if len(df) >= 60: df['MA60'] = df['close'].rolling(window=60).mean()

    # --- KD ---
    # 分段計算避免行太長
    rsv_num = df['close'] - df['low'].rolling(9).min()
    rsv_den = df['high'].rolling(9).max() - df['low'].rolling(9).min()
    df['RSV'] = (rsv_num / rsv_den) * 100
    
    df['K'] = df['RSV'].ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()

    # --- MACD ---
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    
    return df

# ==========================================
# 3. 訊號判斷邏
