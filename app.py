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
        # 加上 .TW 後綴
        ticker = f"{stock_code}.TW"
        
        # 下載資料
        df = yf.download(ticker, start="2023-01-01", auto_adjust=False)
        
        if df.empty:
            return pd.DataFrame()

        # --- 資料清洗 ---
        # 1. 處理 MultiIndex (移除第一層)
        if isinstance(df.columns, pd.MultiIndex):
             df.columns = df.columns.get_level_values(0)
        
        # 2. 轉小寫
        df.columns = [str(c).lower() for c in df.columns]
        
        # 3. 處理索引與時區
        df.index.name = 'date'
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return df

    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()

# ==========================================
# 2. 指標計算函數
# ==========================================
def calculate_indicators(df):
    # 複製一份以免影響原始資料
    df = df.copy()
    
    # --- 均線 (MA) ---
    # 使用 try-except 避免資料不足時報錯
    try:
        if len(df) >= 5: df['MA5'] = df['close'].rolling(window=5).mean()
        if len(df) >= 10: df['MA10'] = df['close'].rolling(window=10).mean()
        if len(df) >= 20: df['MA20'] = df['close'].rolling(window=20).mean()
        if len(df) >= 60: df['MA60'] = df['close'].rolling(window=60).mean()

        # --- KD ---
        rsv_min = df['low'].rolling(9).min()
        rsv_max = df['high'].rolling(9).max()
        
        df['RSV'] = (df['close'] - rsv_min) / (rsv_max - rsv_min) * 100
        df['K'] = df['RSV'].ewm(com=2).mean()
        df['D'] = df['K'].ewm(com=2).mean()

        # --- MACD ---
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal']
    except Exception:
        pass
    
    return df

# ==========================================
# 3. 訊號判斷邏輯
# ==========================================
def analyze_signals(df):
    if len(df) < 2: return ["資料不足，無法分析"]
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []

    # 1. 均線排列
    has_ma = 'MA5' in df.columns and 'MA20' in df.columns and 'MA60' in df.columns
    if has_ma:
        # 分開寫判斷式，避免語法錯誤
        bull = last['MA5'] > last['MA20'] and last['MA20'] > last['MA60']
        bear = last['MA5'] < last['MA20'] and last['MA20'] < last['MA60']
        
        if bull:
            signals.append("🔥 **均線多頭排列**：短中長期均線向上，趨勢偏多。")
        elif bear:
            signals.append("❄️ **均線空頭排列**：短中長期均線向下，趨勢偏空。")

    # 2. KD 指標 (檢查是否有 NaN)
    if not pd.isna(last['K']) and not pd.isna(last['D']):
        if last['K'] > last['D'] and prev['K'] < prev['D']:
            signals.append("📈 **KD黃金交叉**：K值向上突破D值，短線可能轉強。")
        elif last['K'] < last['D'] and prev['K'] > prev['D']:
            signals.append("📉 **KD死亡交叉**：K值向下跌破D值，短線可能轉弱。")
    
    # 3. MACD 指標
    if not pd.isna(last['Hist']):
        if last['Hist'] > 0 and prev['Hist'] < 0:
            signals.append("🟢 **MACD 翻紅**：柱狀體由負轉正，買方力道增強。")
        elif last['Hist'] < 0 and prev['Hist'] > 0:
            signals.append("🔴 **MACD 翻綠**：柱狀體由正轉負，賣方力道增強。")

    if not signals:
        signals.append("⚖️ 目前無明顯技術訊號，建議觀望。")

    return signals

# ==========================================
# 4. 黃金分割率計算
# ==========================================
def calculate_fibonacci(df):
    # 取最近 120 天
    subset = df.tail(120)
    
    highest = subset['high'].max()
    lowest = subset['low'].min()
    diff = highest - lowest
    
    levels = {}
    levels['0.0 (近期低點)'] = lowest
    levels['0.382 (強支撐)'] = lowest + diff * 0.382
    levels['0.5 (中關)'] = lowest + diff * 0.5
    levels['0.618 (壓力)'] = lowest + diff * 0.618
    levels['1.0 (近期高點)'] = highest
    
    return levels

# ==========================================
# 5. 主程式介面
# ==========================================
st.title("📈 全方位股票技術分析儀表板")

col1, col2 = st.columns([1, 2])
with col1:
    stock_code = st.text_input("輸入股票代碼", "2330")
with col2:
    st.caption("輸入代碼後按 Enter (例如: 2330, 0050, 2603)")

if stock_code:
    df = get_stock_data(stock_code)
    
    if df.empty:
        st.error(f"找不到代碼 {stock_code} 的資料，請確認輸入是否正確。")
    else:
        # 計算指標
        df = calculate_indicators(df)

        # 建立三個分頁
        tab1, tab2, tab3 = st.tabs(["📊 K線圖表", "💡 訊號分析", "📐 黃金分割"])

        # === Tab 1: K線圖 ===
        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                # 預設勾選
                selected_mas = st.multiselect("均線", ["MA5", "MA10", "MA20", "MA60"], ["MA5", "MA20", "MA60"])
            with c2:
                options = st.multiselect("副圖", ["Volume", "KD", "MACD"], ["Volume", "KD"])

            # 準備繪圖
            add_plots = []
            ma_colors = {'MA5': 'orange', 'MA10': 'cyan', 'MA20': 'purple', 'MA60': 'green'}
            
            # 加入均線
            for ma in selected_mas:
                if ma in df.columns:
                    add_plots.append(mpf.make_addplot(df[ma], panel=0, color=ma_colors[ma], width=1.0))

            # 加入副圖
            panel_id = 0
            show_vol = False
            
            if "Volume" in options:
                panel_id += 1
                show_vol = True
            
            if "KD" in options:
                panel_id += 1
                add_plots.append(mpf.make_addplot(df['K'], panel=panel_id, color='orange', title='KD'))
                add_plots.append(mpf.make_addplot(df['D'], panel=panel_id, color='blue'))

            if "MACD" in options:
                panel_id += 1
                add_plots.append(mpf.make_addplot(df['MACD'], panel=panel_id, color='red', title='MACD'))
                add_plots.append(mpf.make_addplot(df['Signal'], panel=panel_id,
