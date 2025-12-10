import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import time
import requests
from bs4 import BeautifulSoup

# ==========================================
# 1. 資料抓取函數
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

    try:
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.columns = [str(c).lower() for c in df.columns]
        df.index.name = 'date'
        return df, found_ticker
    except Exception:
        return pd.DataFrame(), ""

# ==========================================
# 2. 獲取公司名稱
# ==========================================
@st.cache_data(ttl=86400)
def get_stock_name(stock_code):
    code = str(stock_code).strip()
    
    # 內建熱門股字典
    stock_map = {
        "0050": "元大台灣50", "0056": "元大高股息", "00878": "國泰永續高股息", "00929": "復華台灣科技優息",
        "00919": "群益台灣精選高息", "006208": "富邦台50", "00713": "元大台灣高息低波",
        "2330": "台積電", "2454": "聯發科", "2303": "聯電", "2317": "鴻海",
        "2308": "台達電", "3711": "日月光投控", "2382": "廣達", "3231": "緯創",
        "6669": "緯穎", "2357": "華碩", "2356": "英業達", "3008": "大立光",
        "3034": "聯詠", "2379": "瑞昱", "3037": "欣興", "3035": "智原",
        "3443": "創意", "3661": "世芯-KY", "5269": "祥碩", "2408": "南亞科",
        "2344": "華邦電", "5347": "世界先進", "6770": "力積電", "2353": "宏碁",
        "2324": "仁寶", "3017": "奇鋐", "3324": "雙鴻", "2376": "技嘉", "2377": "微星",
        "3293": "鈊象", "2603": "長榮", "2609": "陽明", "2615": "萬海", "2618": "長榮航",
        "2610": "華航", "2002": "中鋼", "1101": "台泥", "1102": "亞泥", "1605": "華新",
        "6505": "台塑化", "1301": "台塑", "1303": "南亞", "1326": "台化",
        "2881": "富邦金", "2882": "國泰金", "2891": "中信金", "2886": "兆豐金",
        "2884": "玉山金", "2885": "元大金", "2880": "華南金", "2883": "開發金",
        "2892": "第一金", "2890": "永豐金", "2887": "台新金", "5880": "合庫金"
    }
    if code in stock_map:
        return stock_map[code]

    try:
        url = f"https://tw.stock.yahoo.com/quote/{code}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title_text = soup.title.string
            if title_text and "(" in title_text:
                return title_text.split("(")[0].strip()
            return title_text
    except Exception:
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

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 布林通道
        df['BB_Mid'] = df['close'].rolling(window=20).mean()
        df['BB_Std'] = df['close'].rolling(window=20).std()
        df['BB_Up'] = df['BB_Mid'] + 2 * df['BB_Std']
        df['BB_Low'] = df['BB_Mid'] - 2 * df['BB_Std']

    except: pass
    return df

# ==========================================
# 4. 訊號與策略分析 (新增操作建議)
# ==========================================
def analyze_signals(df):
    if len(df) < 2: return ["資料不足"]
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []

    # --- 1. 現有訊號 ---
    # MA
    if 'MA5' in df.columns and 'MA20' in df.columns:
        if last['MA5'] > last['MA20'] > last['MA60']: signals.append("🔥 **趨勢**：多頭排列")
        elif last['MA5'] < last['MA20'] < last['MA60']: signals.append("❄️ **趨勢**：空頭排列")
        
        if prev['MA5'] < prev['MA20'] and last['MA5'] > last['MA20']:
            signals.append("✨ **均線金叉**：5日線穿過月線")
        elif prev['MA5'] > prev['MA20'] and last['MA5'] < last['MA20']:
            signals.append("💀 **均線死叉**：5日線跌破月線")

    # KD
    if 'K' in df.columns and 'D' in df.columns:
        if last['K'] > last['D'] and prev['K'] < prev['D']:
            note = "低檔" if last['K'] < 30 else ""
            signals.append(f"📈 **KD{note}金叉** (K={last['K']:.1f})")
        elif last['K'] < last['D'] and prev['K'] > prev['D']:
            note = "高檔" if last['K'] > 80 else ""
            signals.append(f"📉 **KD{note}死叉** (K={last['K']:.1f})")

    # MACD
    if 'Hist' in df.columns:
        if last['Hist'] > 0 and prev['Hist'] < 0: signals.append("🟢 **MACD翻紅**")
        elif last['Hist'] < 0 and prev['Hist'] > 0: signals.append("🔴 **MACD翻綠**")

    # RSI
    if 'RSI' in df.columns:
        if last['RSI'] > 75: signals.append(f"⚠️ **RSI過熱** ({last['RSI']:.1f})")
        elif last['RSI'] < 25: signals.append(f"💎 **RSI超賣** ({last['RSI']:.1f})")

    return signals if signals else ["⚖️ 盤整中"]

def generate_strategy(df):
    """
    生成具體的操作建議
    """
    if len(df) < 60: return None
    
    last_close = df.iloc[-1]['close']
    last = df.iloc[-1]
    
    # 計算支撐與壓力
    # 停損參考：季線 (MA60) 或 前波低點 (60日低)
    support_ma60 = last['MA60'] if 'MA60' in df.columns else 0
    support_low = df['low'].tail(60).min()
    stop_loss_price = max(support_ma60, support_low) # 取比較靠近現在價格的那個
    
    # 停利參考：前波高點 (60日高) 或 布林上軌
    resist_high = df['high'].tail(60).max()
    resist_bb = last['BB_Up'] if 'BB_Up' in df.columns else resist_high
    take_profit_price = min(resist_high, resist_bb) # 保守一點，取比較低的那個壓力
    
    # 如果現在價格已經創新高，壓力就設為布林上軌 * 1.05
    if last_close >= resist_high:
        take_profit_price = last_close * 1.05

    # 判斷策略方向
    strategy = {
        "action": "觀望",
        "reason": "多空不明，建議場外觀望",
        "entry_price": "不建議進場",
        "stop_loss": f"{stop_loss_price:.2f}",
        "take_profit": f"{take_profit_price:.2f}"
    }

    # === 多頭策略 ===
    # 條件：收盤在月線之上 且 月線向上
    ma20_up = df['MA20'].iloc[-1] > df['MA20'].iloc[-5] if 'MA20' in df.columns else False
    
    if last_close > last['MA20'] and ma20_up:
        strategy["action"] = "偏多操作 (Bullish)"
        strategy["reason"] = "股價站上月線且月線翻揚，趨勢偏多"
        strategy["entry_price"] = f"拉回測試 {last['MA20']:.2f} (月線) 不破可佈局"
        strategy["stop_loss"] = f"{stop_loss_price:.2f} (跌破季線或前低)"
        
        # 如果乖離過大 (收盤價 > 月線 10%)
        if last_close > last['MA20'] * 1.1:
            strategy["action"] = "勿追高 (Wait)"
            strategy["reason"] = "短線乖離過大，隨時可能回檔"
            strategy["entry_price"] = f"等回測 {last['MA5']:.2f} (5日線) 再觀察"

    # === 空頭策略 ===
    elif last_close < last['MA20']:
        strategy["action"] = "保守/空手 (Bearish)"
        strategy["reason"] = "股價位於月線之下，中期趨勢偏弱"
        strategy["entry_price"] = "暫不建議接刀，待站回月線"
        strategy["take_profit"] = f"{last['MA20']:.2f} (月線反壓)"
        
        # 如果 RSI 超賣
        if 'RSI' in df.columns and last['RSI'] < 25:
            strategy["action"] = "搶反彈 (Rebound)"
            strategy["reason"] = "RSI嚴重超賣，可能有技術性反彈"
            strategy["entry_price"] = "現價輕倉嘗試"
            strategy["stop_loss"] = f"{last_close * 0.95:.2f} (設 5% 停損)"

    return strategy

# ==========================================
# 5. 黃金分割
# ==========================================
def calculate_fibonacci(df):
    subset = df.tail(120)
    high = subset['high'].max()
    low = subset['low'].min()
    diff = high - low
    return {
        '0.0 (低)': low, '0.382 (支撐)': low + diff * 0.382,
        '0.5 (中關)': low + diff * 0.5, '0.618 (壓力)': low + diff * 0.618,
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
    st.error("系統忙碌中，請稍後再試")
    df = pd.DataFrame()

with col2:
    if not df.empty:
        name = get_stock_name(stock_code)
        last = df.iloc[-1]['close']
        prev = df.iloc[-2]['close']
        change = last - prev
        pct = (change / prev) * 100
        st.metric(label=f"{name} ({stock_code})", value=f"{last:.2f}", delta=f"{change:.2f} ({pct:.2f}%)")
    else:
        st.caption("請輸入代碼並按 Enter")

if not df.empty:
    df = calculate_indicators(df)
    tab1, tab2, tab3 = st.tabs(["📊 K線圖", "💡 訊號診斷", "📐 黃金分割"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1: mas = st.multiselect("均線", ["MA5","MA10","MA20","MA60"], ["MA5","MA20","MA60"])
        with c2: inds = st.multiselect("副圖", ["Volume","KD","MACD","RSI"], ["Volume","KD"])

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
        if "RSI" in inds and 'RSI' in df.columns:
            pid+=1
            add_plots.append(mpf.make_addplot(df['RSI'], panel=pid, color='#9b59b6'))
            add_plots.append(mpf.make_addplot([70]*len(df), panel=pid, color='gray', linestyle='dashed'))
            add_plots.append(mpf.make_addplot([30]*len(df), panel=pid, color='gray', linestyle='dashed'))

        try:
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
        st.subheader("🤖 AI 技術指標診斷")
        
        # 1. 顯示訊號
        col_sig1, col_sig2 = st.columns(2)
        signals = analyze_signals(df)
        mid = len(signals)//2 + 1
        with col_sig1:
            for s in signals[:mid]: st.info(s)
        with col_sig2:
            for s in signals[mid:]: st.info(s)

        st.divider()
        
        # 2. 顯示操作建議
        st.subheader("🎯 AI 操盤室 (僅供參考)")
        strategy = generate_strategy(df)
        
        if strategy:
            # 判斷顏色
            color = "off"
            if "偏多" in strategy['action']: color = "normal" # 綠色/正常
            elif "保守" in strategy['action']: color = "off"   # 灰色
            
            # 使用 Metric 顯示關鍵價位
            m1, m2, m3 = st.columns(3)
            m1.metric("建議方向", strategy['action'], strategy['reason'], delta_color=color)
            m2.metric("停損參考 (Stop Loss)", strategy['stop_loss'], "跌破出場", delta_color="inverse")
            m3.metric("停利參考 (Take Profit)", strategy['take_profit'], "壓力位置")
            
            st.markdown(f"""
            > **💡 進場策略**：{strategy['entry_price']}
            """)
        else:
            st.warning("資料不足，無法計算策略")

    with tab3:
        st.subheader("黃金分割")
        fib = calculate_fibonacci(df)
        st.table(pd.DataFrame([{"位置":k, "價格":f"{v:.2f}"} for k,v in fib.items()]))
        st.info(f"觀察：{fib['0.382 (支撐)']:.2f} 為強支撐；跌破 {fib['0.5 (中關)']:.2f} 轉弱")
