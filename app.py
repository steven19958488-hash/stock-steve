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

        # BB
        df['BB_Mid'] = df['close'].rolling(window=20).mean()
        df['BB_Std'] = df['close'].rolling(window=20).std()
        df['BB_Up'] = df['BB_Mid'] + 2 * df['BB_Std']
        df['BB_Low'] = df['BB_Mid'] - 2 * df['BB_Std']

    except: pass
    return df

# ==========================================
# 4. 雙策略分析 (短線 vs 長線)
# ==========================================
def analyze_signals(df):
    if len(df) < 2: return ["資料不足"]
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []

    # MA
    if 'MA5' in df.columns and 'MA20' in df.columns:
        if last['MA5'] > last['MA20'] > last['MA60']: signals.append("🔥 **趨勢**：多頭排列")
        elif last['MA5'] < last['MA20'] < last['MA60']: signals.append("❄️ **趨勢**：空頭排列")
        if prev['MA5'] < prev['MA20'] and last['MA5'] > last['MA20']: signals.append("✨ **均線金叉**：5日線穿過月線")
        elif prev['MA5'] > prev['MA20'] and last['MA5'] < last['MA20']: signals.append("💀 **均線死叉**：5日線跌破月線")

    # KD
    if 'K' in df.columns and 'D' in df.columns:
        if last['K'] > last['D'] and prev['K'] < prev['D']:
            note = "低檔" if last['K'] < 30 else ""
            signals.append(f"📈 **KD{note}金叉**")
        elif last['K'] < last['D'] and prev['K'] > prev['D']:
            note = "高檔" if last['K'] > 80 else ""
            signals.append(f"📉 **KD{note}死叉**")
            
    # MACD
    if 'Hist' in df.columns:
        if last['Hist'] > 0 and prev['Hist'] < 0: signals.append("🟢 **MACD翻紅**")
        elif last['Hist'] < 0 and prev['Hist'] > 0: signals.append("🔴 **MACD翻綠**")

    # RSI
    if 'RSI' in df.columns:
        if last['RSI'] > 75: signals.append(f"⚠️ **RSI過熱** ({last['RSI']:.1f})")
        elif last['RSI'] < 25: signals.append(f"💎 **RSI超賣** ({last['RSI']:.1f})")

    return signals if signals else ["⚖️ 盤整中"]

def generate_dual_strategy(df):
    """
    生成短線與長線的雙重策略
    """
    if len(df) < 60: return None, None
    
    last = df.iloc[-1]
    last_close = last['close']
    
    # ---------------------------
    # A. 短線策略 (時間約 1 個月)
    # ---------------------------
    # 邏輯：看月線 (MA20) + KD/RSI + 布林通道
    short_term = {
        "title": "中性觀望", "icon": "⚖️", "color": "gray",
        "action": "觀望", "stop_loss": 0, "take_profit": 0, "desc": "多空不明"
    }
    
    # 短線停損參考：月線 或 10日低點
    sl_short = last['MA20'] if 'MA20' in df.columns else last_close * 0.9
    # 短線停利參考：布林上軌 或 10日高點
    tp_short = last['BB_Up'] if 'BB_Up' in df.columns else last_close * 1.1

    # 短多判斷
    if last_close > last['MA20']:
        short_term["title"] = "短多操作"
        short_term["icon"] = "⚡"
        short_term["color"] = "green"
        short_term["action"] = "拉回佈局"
        short_term["desc"] = "股價站上月線，短線強勢。"
        
        # 過熱修正
        if 'RSI' in df.columns and last['RSI'] > 75:
            short_term["title"] = "短線過熱"
            short_term["icon"] = "🔥"
            short_term["color"] = "orange"
            short_term["action"] = "分批獲利"
            short_term["desc"] = "RSI 過高，隨時可能回檔整理。"
            
    # 短空判斷
    elif last_close < last['MA20']:
        short_term["title"] = "短線偏空"
        short_term["icon"] = "📉"
        short_term["color"] = "red"
        short_term["action"] = "反彈減碼"
        short_term["desc"] = "股價跌破月線，短線轉弱。"
        tp_short = last['MA20'] # 壓力變月線

    short_term["stop_loss"] = f"{sl_short:.2f}"
    short_term["take_profit"] = f"{tp_short:.2f}"

    # ---------------------------
    # B. 長線策略 (3 個月以上)
    # ---------------------------
    # 邏輯：看季線 (MA60) + MACD + 黃金分割
    long_term = {
        "title": "中性持有", "icon": "🐢", "color": "gray",
        "action": "續抱", "stop_loss": 0, "take_profit": 0, "desc": "趨勢盤整"
    }

    # 長線停損參考：季線 (MA60)
    sl_long = last['MA60'] if 'MA60' in df.columns else last_close * 0.85
    # 長線停利參考：前波高點 (120日高)
    tp_long = df['high'].tail(120).max()

    # 長多判斷 (股價在季線之上，且 MACD > 0)
    macd_val = last['MACD'] if 'MACD' in df.columns else 0
    
    if last_close > last['MA60']:
        long_term["title"] = "長線多頭"
        long_term["icon"] = "🚀"
        long_term["color"] = "green"
        long_term["action"] = "波段續抱"
        long_term["desc"] = "站穩季線(生命線)，長多格局未變。"
        
        if macd_val < 0:
            long_term["desc"] += " 但動能稍弱。"

    # 長空判斷
    elif last_close < last['MA60']:
        long_term["title"] = "長線轉弱"
        long_term["icon"] = "❄️"
        long_term["color"] = "red"
        long_term["action"] = "保守應對"
        long_term["desc"] = "跌破季線，需提防中期趨勢反轉。"
        tp_long = last['MA60'] # 壓力變季線

    long_term["stop_loss"] = f"{sl_long:.2f}"
    long_term["take_profit"] = f"{tp_long:.2f}"

    return short_term, long_term

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
        signals = analyze_signals(df)
        col_s1, col_s2 = st.columns(2)
        mid = (len(signals) + 1) // 2
        with col_s1:
            for s in signals[:mid]: st.info(s)
        with col_s2:
            for s in signals[mid:]: st.info(s)

        st.divider()
        st.subheader("🎯 AI 操盤室")
        
        # 取得雙策略
        short_strat, long_strat = generate_dual_strategy(df)
        
        if short_strat and long_strat:
            # 使用左右兩欄分割
            col_short, col_long = st.columns(2)
            
            # --- 左欄：短線策略 ---
            with col_short:
                with st.container(border=True):
                    st.markdown(f"### {short_strat['icon']} 短線波段 (1個月)")
                    st.markdown(f"**{short_strat['title']}**")
                    st.caption(short_strat['desc'])
                    
                    st.divider()
                    st.metric("建議動作", short_strat['action'])
                    st.metric("🛑 停損 (月線)", short_strat['stop_loss'])
                    st.metric("💰 停利 (壓力)", short_strat['take_profit'])
                    
                    if short_strat['color'] == 'green':
                        st.success("核心：關注月線支撐與KD變化")
                    elif short_strat['color'] == 'red':
                        st.error("核心：嚴守紀律，反彈站不回月線要跑")
                    else:
                        st.warning("核心：RSI過熱，小心回馬槍")

            # --- 右欄：長線策略 ---
            with col_long:
                with st.container(border=True):
                    st.markdown(f"### {long_strat['icon']} 長線投資 (3個月+)")
                    st.markdown(f"**{long_strat['title']}**")
                    st.caption(long_strat['desc'])
                    
                    st.divider()
                    st.metric("建議動作", long_strat['action'])
                    st.metric("🛡️ 防守 (季線)", long_strat['stop_loss'])
                    st.metric("🎯 目標 (前高)", long_strat['take_profit'])
                    
                    if long_strat['color'] == 'green':
                        st.success("觀點：季線向上，拉回皆是買點")
                    else:
                        st.error("觀點：季線下彎，中期趨勢修正中")
        else:
            st.warning("資料不足，無法生成策略")

    with tab3:
        st.subheader("黃金分割")
        fib = calculate_fibonacci(df)
        st.table(pd.DataFrame([{"位置":k, "價格":f"{v:.2f}"} for k,v in fib.items()]))
        st.info(f"觀察：{fib['0.382 (支撐)']:.2f} 為強支撐；跌破 {fib['0.5 (中關)']:.2f} 轉弱")
