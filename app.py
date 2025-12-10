import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import time

# ==========================================
# 1. 資料抓取函數 (v3.1)
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
            # 抓取資料
            temp_df = stock.history(start="2023-01-01", auto_adjust=False)
            
            if not temp_df.empty:
                df = temp_df
                found_ticker = ticker
                break
            time.sleep(0.5) 
        except Exception:
            continue

    if df.empty:
        return pd.DataFrame(), ""

    # 清洗資料
    try:
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.columns = [str(c).lower() for c in df.columns]
        df.index.name = 'date'
        return df, found_ticker
    except Exception:
        return pd.DataFrame(), ""

# ==========================================
# 2. 獲取公司中文名稱
# ==========================================
@st.cache_data(ttl=86400)
def get_stock_name(stock_code):
    code = str(stock_code).strip()
    try:
        import twstock
        if code in twstock.codes:
            return twstock.codes[code].name
    except:
        pass 
    return code

# ==========================================
# 3. 指標計算
# ==========================================
def calculate_indicators(df):
    df = df.copy()
    try:
        # MA
        if len(df) >= 5: df['MA5'] = df['close'].rolling(5).mean()
        if len(df) >= 10: df['MA10'] = df['close'].rolling(10).mean()
        if len(df) >= 20: df['MA20'] = df['close'].rolling(20).mean()
        if len(df) >= 60: df['MA60'] = df['close'].rolling(60).mean()
        # KD
        rsv_min = df['low'].rolling(9).min()
        rsv_max = df['high'].rolling(9).max()
        rsv_den = rsv_max - rsv_min
        rsv_den[rsv_den == 0] = 1 
        df['RSV'] = (df['close'] - rsv_min) / rsv_den * 100
        df['K'] = df['RSV'].ewm(com=2).mean()
        df['D'] = df['K'].ewm(com=2).mean()
        # MACD
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal']
    except:
        pass
    return df

# ==========================================
# 4. 訊號分析
# ==========================================
def analyze_signals(df):
    if len(df) < 2: return ["資料不足"]
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []

    # MA
    if 'MA5' in df.columns and 'MA20' in df.columns and 'MA60' in df.columns:
        if last['MA5'] > last['MA20'] > last['MA60']:
            signals.append("🔥 **均線多頭**：趨勢向上")
        elif last['MA5'] < last['MA20'] < last['MA60']:
            signals.append("❄️ **均線空頭**：趨勢向下")

    # KD
    if 'K' in df.columns and 'D' in df.columns:
        if last['K'] > last['D'] and prev['K'] < prev['D']:
            signals.append("📈 **KD金叉**：短線轉強")
        elif last['K'] < last['D'] and prev['K'] > prev['D']:
            signals.append("📉 **KD死叉**：短線轉弱")
            
    # MACD
    if 'Hist' in df.columns:
        if last['Hist'] > 0 and prev['Hist'] < 0:
            signals.append("🟢 **MACD翻紅**：買氣增強")
        elif last['Hist'] < 0 and prev['Hist'] > 0:
            signals.append("🔴 **MACD翻綠**：賣壓增強")

    return signals if signals else ["⚖️ 盤整中"]

# ==========================================
# 5. 黃金分割
# ==========================================
def calculate_fibonacci(df):
    subset = df.tail(120)
    high = subset['high'].max()
    low = subset['low'].min()
    diff = high - low
    return {
        '0.0 (低)': low,
        '0.382 (支撐)': low + diff * 0.382,
        '0.5 (中關)': low + diff * 0.5,
        '0.618 (壓力)': low + diff * 0.618,
        '1.0 (高)': high
    }

# ==========================================
# 6. 主程式介面
# ==========================================
st.title("📈 股票技術分析儀表板")

col1, col2 = st.columns([1, 2])
with col1:
    stock_code = st.text_input("輸入代碼", "2330")

try:
    df, valid_ticker = get_stock_data_v3(stock_code)
except:
    st.error("系統忙碌中 (Rate Limit)，請稍後再試")
    df = pd.DataFrame()

# UI 顯示區 (這裡可以用中文，因為是網頁 HTML)
with col2:
    if not df.empty:
        # 抓中文名稱
        name = get_stock_name(stock_code)
        
        last = df.iloc[-1]['close']
        prev = df.iloc[-2]['close']
        change = last - prev
        pct = (change / prev) * 100
        
        # 這裡會正常顯示中文
        st.metric(
            label=f"{name} ({stock_code})",
            value=f"{last:.2f}",
            delta=f"{change:.2f} ({pct:.2f}%)"
        )
    else:
        st.caption("請輸入代碼並按 Enter")

if not df.empty:
    df = calculate_indicators(df)
    tab1, tab2, tab3 = st.tabs(["📊 K線圖", "💡 訊號", "📐 黃金分割"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1: mas = st.multiselect("均線", ["MA5","MA10","MA20","MA60"], ["MA5","MA20","MA60"])
        with c2: inds = st.multiselect("副圖", ["Volume","KD","MACD"], ["Volume","KD"])

        add_plots = []
        colors = {'MA5':'orange', 'MA10':'cyan', 'MA20':'purple', 'MA60':'green'}
        
        for ma in mas:
            if ma in df.columns:
                add_plots.append(mpf.make_addplot(df[ma], panel=0, color=colors[ma], width=1.0))

        pid = 0
        vol = False
        if "Volume" in inds: pid+=1; vol=True
        if "KD" in inds and 'K' in df.columns:
            pid+=1
            add_plots.append(mpf.make_addplot(df['K'], panel=pid, color='orange'))
            add_plots.append(mpf.make_addplot(df['D'], panel=pid, color='blue'))
        if "MACD" in inds and 'MACD' in df.columns:
            pid+=1
            add_plots.append(mpf.make_addplot(df['MACD'], panel=pid, color='red'))
            add_plots.append(mpf.make_addplot(df['Signal'], panel=pid, color='blue'))
            add_plots.append(mpf.make_addplot(df['Hist'], type='bar', panel=pid, color='gray', alpha=0.5))

        try:
            # === 關鍵修改 ===
            # 1. title 改用純英文或數字，避免 Glyph 錯誤
            # 2. 加入 warn_too_much_data=10000 忽略資料過多警告
            fig, ax = mpf.plot(
                df, type='candle', style='yahoo', volume=vol, 
                addplot=add_plots, returnfig=True,
                panel_ratios=tuple([2]+[1]*pid), figsize=(10, 8),
                title=f"Stock Code: {stock_code}", 
                warn_too_much_data=10000 
            )
            st.pyplot(fig)
        except Exception as e: st.error(f"Error: {e}")

    with tab2:
        st.subheader("技術分析訊號")
        for s in analyze_signals(df): st.write(s)

    with tab3:
        st.subheader("黃金分割")
        fib = calculate_fibonacci(df)
        st.table(pd.DataFrame([{"位置":k, "價格":f"{v:.2f}"} for k,v in fib.items()]))
        st.info(f"觀察：{fib['0.382 (支撐)']:.2f} 為強支撐；跌破 {fib['0.5 (中關)']:.2f} 轉弱")
