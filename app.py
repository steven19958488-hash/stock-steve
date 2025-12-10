import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf

# ==========================================
# 1. 資料抓取函數 (已修復 yfinance 格式問題)
# ==========================================
@st.cache_data
def get_stock_data(stock_code):
    try:
        ticker = f"{stock_code}.TW"
        # 抓取較長區間以利計算黃金分割
        df = yf.download(ticker, start="2023-01-01", auto_adjust=False)
        
        if df.empty:
            st.warning(f"找不到 {stock_code} 的資料，請確認代碼。")
            return pd.DataFrame()

        # --- 資料清洗與格式修正 ---
        # 處理 MultiIndex (移除第一層)
        if isinstance(df.columns, pd.MultiIndex):
             df.columns = df.columns.get_level_values(0)
        
        # 轉小寫
        df.columns = [str(c).lower() for c in df.columns]
        
        # 確保索引與時區
        df.index.name = 'date'
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return df

    except Exception as e:
        st.error(f"資料抓取失敗: {e}")
        return pd.DataFrame()

# ==========================================
# 2. 指標計算函數 (含 MA, KD, MACD)
# ==========================================
def calculate_indicators(df):
    # --- 均線 (MA) ---
    if len(df) >= 5: df['MA5'] = df['close'].rolling(window=5).mean()
    if len(df) >= 10: df['MA10'] = df['close'].rolling(window=10).mean()
    if len(df) >= 20: df['MA20'] = df['close'].rolling(window=20).mean()
    if len(df) >= 60: df['MA60'] = df['close'].rolling(window=60).mean()

    # --- KD ---
    df['RSV'] = (df['close'] - df['low'].rolling(9).min()) / (df['high'].rolling(9).max() - df['low'].rolling(9).min()) * 100
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
# 3. (補回) 訊號判斷邏輯
# ==========================================
def analyze_signals(df):
    if len(df) < 2: return "資料不足"
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []

    # 1. 均線排列
    if 'MA5' in df.columns and 'MA20' in df.columns and 'MA60' in df.columns:
        if last['MA5'] > last['MA20'] > last['MA60']:
            signals.append("🔥 **均線多頭排列**：短中長期均線向上，趨勢偏多。")
        elif last['MA5'] < last['MA20'] < last['MA60']:
            signals.append("❄️ **均線空頭排列**：短中長期均線向下，趨勢偏空。")

    # 2. KD 指標
    if last['K'] > last['D'] and prev['K'] < prev['D']:
        signals.append("📈 **KD黃金交叉**：K值向上突破D值，短線可能轉強。")
    elif last['K'] < last['D'] and prev['K'] > prev['D']:
        signals.append("📉 **KD死亡交叉**：K值向下跌破D值，短線可能轉弱。")
    
    # 3. MACD 指標
    if last['Hist'] > 0 and prev['Hist'] < 0:
        signals.append("🟢 **MACD 翻紅**：柱狀體由負轉正，買方力道增強。")
    elif last['Hist'] < 0 and prev['Hist'] > 0:
        signals.append("🔴 **MACD 翻綠**：柱狀體由正轉負，賣方力道增強。")

    if not signals:
        signals.append("⚖️ 目前無明顯技術訊號，建議觀望或參考其他資訊。")

    return signals

# ==========================================
# 4. (補回) 黃金分割率計算
# ==========================================
def calculate_fibonacci(df):
    # 取最近 100 天 (或是半年) 的高低點來畫
    lookback = 120 
    subset = df.tail(lookback)
    
    highest = subset['high'].max()
    lowest = subset['low'].min()
    diff = highest - lowest
    
    levels = {
        '0.0 (近期低點)': lowest,
        '0.382 (強支撐)': lowest + diff * 0.382,
        '0.5 (中關)': lowest + diff * 0.5,
        '0.618 (壓力)': lowest + diff * 0.618,
        '1.0 (近期高點)': highest
    }
    return levels

# ==========================================
# 5. 主程式介面
# ==========================================
st.title("📈 全方位股票技術分析儀表板")

col_input, col_info = st.columns([1, 2])
with col_input:
    stock_code = st.text_input("輸入股票代碼", "2330")
with col_info:
    st.info("包含：K線圖、均線、KD/MACD、多空訊號解讀、黃金分割率")

if stock_code:
    df = get_stock_data(stock_code)
    
    if not df.empty:
        df = calculate_indicators(df)

        # --- 版面配置 ---
        tab1, tab2, tab3 = st.tabs(["📊 K線技術圖表", "💡 多空訊號分析", "📐 黃金分割率"])

        # === Tab 1: 圖表區 ===
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                selected_mas = st.multiselect("均線 (MA)", ["MA5", "MA10", "MA20", "MA60"], default=["MA5", "MA20", "MA60"])
            with col2:
                options = st.multiselect("副圖指標", ["Volume", "KD", "MACD"], default=["Volume", "KD"])

            # 準備繪圖
            add_plots = []
            ma_colors = {'MA5': 'orange', 'MA10': 'cyan', 'MA20': 'purple', 'MA60': 'green'}
            
            # 加入均線
            for ma in selected_mas:
                if ma in df.columns and not df[ma].isna().all():
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
                add_plots.append(mpf.make_addplot(df['Signal'], panel=panel_id, color='blue'))
                add_plots.append(mpf.make_addplot(df['Hist'], type='bar', panel=panel_id, color='gray', alpha=0.5))

            current_ratios = [2] + [1] * panel_id

            # 畫圖
            fig, axlist = mpf.plot(
                df, type='candle', style='yahoo', volume=show_vol, 
                addplot=add_plots, returnfig=True,
                panel_ratios=tuple(current_ratios), figsize=(10, 8),
                title=f"{stock_code} Analysis"
            )
            st.pyplot(fig)

        # === Tab 2: 訊號分析區 ===
        with tab2:
            st.subheader("🤖 AI 技術面解讀")
            signals = analyze_signals(df)
            for sig in signals:
                st.write(sig)
            
            st.divider()
            last_price = df.iloc[-1]['close']
            st.metric("最新收盤價", f"{last_price:.2f}")

        # === Tab 3: 黃金分割區 ===
        with tab3:
            st.subheader("📐 黃金分割率 (Fibonacci Retracement)")
            st.write("根據最近 120 個交易日的高低點計算：")
            
            levels = calculate_fibonacci(df)
            last_price = df.iloc[-1]['close']
            
            # 顯示表格
            fibo_df = pd.DataFrame(list(levels.items()), columns=['關鍵位置', '價格'])
            fibo_df['價格'] = fibo_df['價格'].map('{:.2f}'.format)
            
            # 標示目前價格位置
            def highlight_price(val):
                return ['background-color: #d4edda' if val == '目前價格' else '' for _ in val]

            st.table(fibo_df)
            
            st.info(f"💡 觀察重點：若股價回檔至 **0.382 ({levels['0.382 (強支撐)']:.2f})
