import streamlit as st
import twstock
import pandas as pd
import mplfinance as mpf

# ---------------------------------------------------------
# 1. 解決速度問題：使用 @st.cache_data 把資料暫存起來
# ---------------------------------------------------------
@st.cache_data
def get_stock_data(stock_code):
    try:
        stock = twstock.Stock(stock_code)
        # 抓取資料 (這裡抓取較多天數以確保 MA60 算得出來)
        data = stock.fetch_from(2023, 1) 
        
        # 整理成 DataFrame
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # 轉換型別為 float，避免畫圖報錯
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        return df
    except Exception as e:
        st.error(f"無法抓取股票資料，請檢查代碼是否正確。錯誤訊息: {e}")
        return pd.DataFrame()

# ---------------------------------------------------------
# 2. 計算指標的函數 (新增：均線 MA)
# ---------------------------------------------------------
def calculate_indicators(df):
    # --- 新增：計算均線 (MA) ---
    # 使用 rolling().mean() 計算移動平均
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA60'] = df['close'].rolling(window=60).mean()

    # --- 計算 KD (隨機指標) ---
    df['RSV'] = (df['close'] - df['low'].rolling(9).min()) / (df['high'].rolling(9).max() - df['low'].rolling(9).min()) * 100
    df['K'] = df['RSV'].ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()

    # --- 計算 MACD ---
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    
    return df

# ---------------------------------------------------------
# 3. 主程式介面
# ---------------------------------------------------------
st.title("股票技術分析儀表板")
stock_code = st.text_input("輸入股票代碼", "2330")

if stock_code:
    # 步驟 A: 下載資料
    df = get_stock_data(stock_code)
    
    if not df.empty:
        # 步驟 B: 計算指標 (包含 MA)
        df = calculate_indicators(df)

        # 步驟 C: 介面控制
        col1, col2 = st.columns(2)
        
        with col1:
            # 讓使用者選擇要顯示的「均線」
            selected_mas = st.multiselect(
                "選擇均線 (MA)",
                ["MA5", "MA10", "MA20", "MA60"],
                default=["MA5", "MA20", "MA60"]
            )

        with col2:
            # 讓使用者選擇要顯示的「副圖指標」
            options = st.multiselect(
                "選擇副圖指標",
                ["Volume", "KD", "MACD"],
                default=["Volume", "KD"]
            )

        # -----------------------------------------------------
        # 4. 核心畫圖邏輯：使用 addplot 與 panel
        # -----------------------------------------------------
        add_plots = []
        
        # --- 處理均線 (疊在主圖 panel=0) ---
        # 定義顏色：MA5(白/黃), MA10(藍), MA20(紫/橘), MA60(綠)
        ma_colors = {'MA5': 'orange', 'MA10': 'cyan', 'MA20': 'purple', 'MA60': 'green'}
        
        for ma in selected_mas:
            # 這裡 panel=0 代表畫在 K 線那一格
            add_plots.append(mpf.make_addplot(df[ma], panel=0, color=ma_colors[ma], width=1.0))

        # --- 處理副圖 (Panel ID 遞增) ---
        # 設定 panel 順序：主圖是 0
        panel_id = 0 
        
        # 判斷是否顯示成交量 (Volume)
        # 雖然 mplfinance 有 volume=True 參數，但為了排版控制，我們統一計算 panel_id
        if "Volume" in options:
            panel_id += 1
            show_vol = True
        else:
            show_vol = False

        if "KD" in options:
            panel_id += 1
            add_plots.append(mpf.make_addplot(df['K'], panel=panel_id, color='orange', title='KD'))
            add_plots.append(mpf.make_addplot(df['D'], panel=panel_id, color='blue'))

        if "MACD" in options:
            panel_id += 1
            add_plots.append(mpf.make_addplot(df['MACD'], panel=panel_id, color='red', title='MACD'))
            add_plots.append(mpf.make_addplot(df['Signal'], panel=panel_id, color='blue'))
            add_plots.append(mpf.make_addplot(df['Hist'], type='bar', panel=panel_id, color='gray', alpha=0.5))

        # 動態計算 panel_ratios (高度比例)
        # 主圖固定佔 2 份，其他副圖各佔 1 份
        # 比如有 2 個副圖，比例就是 (2, 1, 1)
        current_ratios = [2] + [1] * panel_id

        st.write(f"目前顯示: {stock_code} | 均線: {', '.join(selected_mas)}")
        
        # 畫圖 
        fig, axlist = mpf.plot(
            df, 
            type='candle', 
            style='yahoo', 
            volume=show_vol, 
            addplot=add_plots, 
            returnfig=True,
            panel_ratios=tuple(current_ratios), # 使用動態計算的比例
            figsize=(10, 8),
            title=f"{stock_code} Daily Chart"
        )
        
        st.pyplot(fig)
