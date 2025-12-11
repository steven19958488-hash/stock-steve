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
        elif adx_val > 25: signals.append(f"💪 **ADX強勢 ({adx_val:.1f})**：趨勢確立，可信度高。")
        elif adx_val < 20: signals.append(f"🟰 **ADX疲弱 ({adx_val:.1f})**：進入盤整，訊號可信度低。")
            
    if 'OBV' in df.columns:
        obv_trend = last['OBV'] > df['OBV'].iloc[-5:-1].mean()
        price_up = last['close'] > df['close'].iloc[-5:-1].mean()
        if obv_trend and price_up: signals.append("✅ **量價同步**：OBV上升，量能推動價格。")
        elif not obv_trend and price_up: signals.append("❌ **量價背離**：價格上漲，但OBV下降，上漲動能不足。")
        
    return signals if signals else ["⚖️ 盤整中"]

def generate_dual_strategy(df):
    if len(df) < 60: return None, None
    last = df.iloc[-1]
    last_close = last['close']
    score = calculate_score(df)
    vol_status = analyze_volume(df)
    
    checklist = {
        "站上月線 (MA20)": last_close > last['MA20'], 
        "季線多頭 (MA60向上)": last['MA20'] > last['MA60'],
        "KD金叉向上": last['K'] > last['D'],
        "MACD偏多 (Hist > 0)": last['Hist'] > 0, 
        "RSI安全 (20~75)": 20 < last['RSI'] < 75
    }
    
    strategy_base = {"title": "中性觀望", "icon": "⚖️", "color": "gray", "action": "觀望", "score": score, "vol": vol_status, "desc": "多空不明，等待訊號。"}
    sl_short = last['MA20'] if 'MA20' in df.columns else last_close * 0.9
    tp_short = last['BB_Up'] if 'BB_Up' in df.columns else last_close * 1.1

    if score >= 95:
        strategy = strategy_base.copy()
        strategy.update({"title": "🚀 趨勢噴發", "icon": "🚀", "color": "green", "action": "現價佈局", 
                         "desc": "訊號極強，已脫離整理區間，建議現價或拉回 5日線佈局。",
                         "entry_text": f"建議現價或回測 **{last['MA5']:.2f}** 佈局 (高風險高報酬)。"})
    elif last_close > last['MA20'] and last['K'] < 80:
        strategy = strategy_base.copy()
        strategy.update({"title": "短多操作", "icon": "⚡", "color": "green", "action": "拉回佈局", 
                         "desc": "股價站上月線，短線強勢。",
                         "entry_text": f"建議拉回測試 **{last['MA20']:.2f} (月線)** 不破時佈局。"})
        
        if last_close > last['close'].shift(1) and last['volume'] < last['VolMA5']:
             strategy.update({"title": "📈 價漲量縮", "icon": "⚠️", "color": "orange", "action": "持股續抱，勿追高", 
                              "desc": "多頭趨勢，但量能不足，追高有風險。",
                              "entry_text": f"持股續抱，空手者等待回測 **{last['MA5']:.2f}** 觀察。"})
        
        if last['RSI'] > 75: 
            strategy.update({"title": "短線過熱", "icon": "🔥", "color": "orange", "action": "分批獲利", 
                             "desc": "雖為多頭但過熱，留意修正。",
                             "entry_text": f"建議等待回測 **{last['MA5']:.2f}** 再觀察。"})
    elif last_close < last['MA20']:
        strategy = strategy_base.copy()
        strategy.update({"title": "短線偏空", "icon": "📉", "color": "red", "action": "反彈減碼", 
                         "desc": "跌破月線，短線轉弱。",
                         "entry_text": "暫不建議進場，待站回月線。"})
        tp_short = last['MA20']
    else:
        strategy = strategy_base.copy()
        strategy["entry_text"] = "暫不建議進場，等待明確訊號。"


    long_term = {"title": "中性持有", "icon": "🐢", "color": "gray", "action": "續抱", "desc": "趨勢盤整"}
    sl_long = last['MA60'] if 'MA60' in df.columns else last_close * 0.85
    tp_long = df['high'].tail(120).max()
    if last_close > last['MA60']:
        long_term.update({"title": "長線多頭", "icon": "🚀", "color": "green", "action": "波段續抱", "desc": "站穩季線，長多格局。"})
    elif last_close < last['MA60']:
        long_term.update({"title": "長線轉弱", "icon": "❄️", "color": "red", "action": "保守應對", "desc": "跌破季線，需提防反轉。"})
        tp_long = last['MA60']

    short_term = strategy
    short_term["stop_loss"] = f"{sl_short:.2f}"
    short_term["take_profit"] = f"{tp_short:.2f}"
    short_term["checklist"] = checklist
    long_term["stop_loss"] = f"{sl_long:.2f}"
    long_term["take_profit"] = f"{tp_long:.2f}"
    
    return short_term, long_term

def calculate_fibonacci_multi(df):
    def get_levels(window_days):
        if len(df) < window_days: return {}
        subset = df.tail(window_days)
        h, l = subset['high'].max(), subset['low'].min()
        d = h - l
        return {'0.0 (低)': l, '0.382': l+d*0.382, '0.5': l+d*0.5, '0.618': l+d*0.618, '1.0 (高)': h}
    return get_levels(20), get_levels(60), get_levels(240)

# ==========================================
# 5. 主程式介面
# ==========================================
st.set_page_config(page_title="股票技術分析儀表板", layout="wide")
st.title("📈 股票技術分析儀表板")

TAIWAN_STYLE = mpf.make_marketcolors(up='g', down='r', edge='inherit', wick='inherit', volume='inherit')
TAIWAN_RC = mpf.make_mpf_style(marketcolors=TAIWAN_STYLE)

col1, col2 = st.columns([1, 2])
with col1:
    stock_code = st.text_input("輸入代碼", "2330")

try:
    df, valid_ticker = get_stock_data_v3(stock_code)
except:
    st.error("系統忙碌中")
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
        time_period = st.radio("範圍：", ["1個月", "3個月", "半年", "1年"], index=1, horizontal=True)
        if time_period == "1個月": plot_df = df.tail(20)
        elif time_period == "3個月": plot_df = df.tail(60)
        elif time_period == "半年": plot_df = df.tail(120)
        else: plot_df = df.tail(240)

        c1, c2 = st.columns(2)
        with c1: mas = st.multiselect("均線", ["MA5","MA10","MA20","MA60"], ["MA5","MA20","MA60"])
        with c2: inds = st.multiselect("副圖", ["Volume","KD","MACD","RSI","BB","ADX","OBV"], ["Volume","KD"])

        add_plots = []
        colors = {'MA5':'orange', 'MA10':'cyan', 'MA20':'purple', 'MA60':'green'}
        for ma in mas:
            if ma in plot_df.columns: add_plots.append(mpf.make_addplot(plot_df[ma], panel=0, color=colors[ma], width=1.0))
        
        if "BB" in inds:
            add_plots.append(mpf.make_addplot(plot_df['BB_Up'], panel=0, color='red', linestyle='dashed', width=0.5))
            add_plots.append(mpf.make_addplot(plot_df['BB_Mid'], panel=0, color='gray', linestyle='dashed', width=0.5))
            add_plots.append(mpf.make_addplot(plot_df['BB_Low'], panel=0, color='green', linestyle='dashed', width=0.5))

        pid = 0
        vol = False
        if "Volume" in inds: pid+=1; vol=True
        if "KD" in inds:
            pid+=1
            add_plots.append(mpf.make_addplot(plot_df['K'], panel=pid, color='orange'))
            add_plots.append(mpf.make_addplot(plot_df['D'], panel=pid, color='blue'))
        if "MACD" in inds:
            pid+=1
            add_plots.append(mpf.make_addplot(plot_df['MACD'], panel=pid, color='red'))
            add_plots.append(mpf.make_addplot(plot_df['Signal'], panel=pid, color='blue'))
            add_plots.append(mpf.make_addplot(plot_df['Hist'], type='bar', panel=pid, color='gray', alpha=0.5))
        if "RSI" in inds:
            pid+=1
            add_plots.append(mpf.make_addplot(plot_df['RSI'], panel=pid, color='#9b59b6'))
            add_plots.append(mpf.make_addplot([70]*len(plot_df), panel=pid, color='gray', linestyle='dashed'))
            add_plots.append(mpf.make_addplot([30]*len(plot_df), panel=pid, color='gray', linestyle='dashed'))
        if "ADX" in inds:
            pid+=1
            add_plots.append(mpf.make_addplot(plot_df['ADX'], panel=pid, color='blue', title='ADX'))
            add_plots.append(mpf.make_addplot([25]*len(plot_df), panel=pid, color='orange', linestyle='dashed', width=0.8))
        if "OBV" in inds:
            pid+=1
            add_plots.append(mpf.make_addplot(plot_df['OBV'], panel=pid, color='purple', type='line', title='OBV'))

        try:
            panel_ratios = tuple([2] + [1] * pid)
            fig, ax = mpf.plot(plot_df, style=TAIWAN_RC, type='candle', volume=vol, addplot=add_plots, returnfig=True, panel_ratios=panel_ratios, figsize=(10, 8), warn_too_much_data=10000)
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
        short_strat, long_strat = generate_dual_strategy(df)
        if short_strat and long_strat:
            col_short, col_long = st.columns(2)
            with col_short:
                with st.container(border=True):
                    st.markdown(f"### {short_strat['icon']} 短線 (1個月)")
                    st.write(f"**AI 信心：{short_strat['score']} 分**")
                    st.progress(short_strat['score'] / 100)
                    st.caption(f"量能：{short_strat['vol']}")
                    st.markdown(f"**{short_strat['title']}**")
                    st.write(short_strat['desc'])
                    st.divider()
                    st.write("**✅ 多空健檢 (純技術面)**")
                    for name, passed in short_strat['checklist'].items():
                        st.write(f"{'✅' if passed else '❌'} {name}")
                    st.divider()
                    st.metric("建議", short_strat['action'])
                    st.metric("🛑 停損", short_strat['stop_loss'])
                    st.metric("💰 停利", short_strat['take_profit'])
            with col_long:
                with st.container(border=True):
                    st.markdown(f"### {long_strat['icon']} 長線 (1年)")
                    st.markdown(f"**{long_strat['title']}**")
                    st.caption(long_strat['desc'])
                    st.divider()
                    st.info("季線(60MA)之上為長多格局。")
                    st.divider()
                    st.metric("建議", long_strat['action'])
                    st.metric("🛡️ 防守", long_strat['stop_loss'])
                    st.metric("🎯 目標", long_strat['take_profit'])

    with tab3:
        st.subheader("📐 黃金分割率")
        u_fib, s_fib, l_fib = calculate_fibonacci_multi(df)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### ⚡ 極短線 (20日)")
            if u_fib: st.table(pd.DataFrame([{"位置":k, "價格":f"{v:.2f}"} for k,v in u_fib.items()]))
        with c2:
            st.markdown("#### 🌊 短線 (60日)")
            if s_fib: st.table(pd.DataFrame([{"位置":k, "價格":f"{v:.2f}"} for k,v in s_fib.items()]))
        with c3:
            st.markdown("#### 🐢 長線 (240日)")
            if l_fib: st.table(pd.DataFrame([{"位置":k, "價格":f"{v:.2f}"} for k,v in l_fib.items()]))
