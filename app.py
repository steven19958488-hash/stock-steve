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
# 2. 籌碼面抓取 (修復版：強化偽裝)
# ==========================================
@st.cache_data(ttl=3600)
def get_institutional_data(stock_code):
    stock_code = str(stock_code).strip()
    data = []
    suffixes = [".TW", ".TWO"]
    
    # 使用完整的 User-Agent 偽裝成真實瀏覽器
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://tw.stock.yahoo.com/',
        'Accept': 'application/json, text/plain, */*'
    }
    
    for suffix in suffixes:
        try:
            # Yahoo 股市 API 接口
            url = f"https://tw.stock.yahoo.com/_td-stock/api/resource/StockServices.institutionalTradingList;count=30;symbol={stock_code}{suffix}"
            res = requests.get(url, headers=headers, timeout=10)
            
            if res.status_code == 200:
                json_data = res.json()
                # 檢查是否有資料
                if 'result' in json_data and json_data['result']:
                    raw_list = json_data['result']
                    for item in raw_list:
                        # 日期處理
                        if 'date' not in item: continue
                        ts = int(item['date']) / 1000
                        date_str = pd.Timestamp(ts, unit='s').strftime('%Y-%m-%d')
                        
                        # 數據處理 (API 單位是股，除以 1000 換算成張)
                        foreign = int(item.get('foreignNetBuySell', 0)) // 1000
                        trust = int(item.get('investmentTrustNetBuySell', 0)) // 1000
                        dealer = int(item.get('dealerNetBuySell', 0)) // 1000
                        
                        data.append({
                            "日期": date_str,
                            "外資": foreign,
                            "投信": trust,
                            "自營商": dealer,
                            "合計": foreign + trust + dealer
                        })
                    if data: break # 成功抓到就跳出迴圈
        except Exception as e:
            print(f"籌碼抓取錯誤: {e}")
            continue
            
    if data:
        df_inst = pd.DataFrame(data)
        df_inst = df_inst.sort_values("日期", ascending=True)
        return df_inst
    return pd.DataFrame()

# ==========================================
# 3. 獲取公司名稱
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
    except: pass
    return code

# ==========================================
# 4. 指標計算
# ==========================================
def calculate_indicators(df):
    df = df.copy()
    try:
        if len(df) >= 5: df['MA5'] = df['close'].rolling(5).mean()
        if len(df) >= 10: df['MA10'] = df['close'].rolling(10).mean()
        if len(df) >= 20: df['MA20'] = df['close'].rolling(20).mean()
        if len(df) >= 60: df['MA60'] = df['close'].rolling(60).mean()
        if len(df) >= 5: df['VolMA5'] = df['volume'].rolling(5).mean()

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
    except: pass
    return df

# ==========================================
# 5. 策略與分析
# ==========================================
def calculate_score(df):
    score = 50 
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    if last['close'] > last['MA20']: score += 10 
    if last['MA20'] > last['MA60']: score += 10
    if last['close'] > last['MA60']: score += 10
    if last['MA5'] > last['MA20']: score += 10
    if last['close'] < last['MA20']: score -= 10
    if last['MA20'] < last['MA60']: score -= 10
    if last['close'] < last['MA60']: score -= 10
    if last['MA5'] < last['MA20']: score -= 10
    
    if last['MACD'] > 0: score += 5
    if last['Hist'] > 0: score += 5
    if last['K'] > last['D']: score += 5
    if last['RSI'] > 80: score -= 5 
    if last['RSI'] < 20: score += 5 
    
    vol_ratio = last['volume'] / last['VolMA5'] if 'VolMA5' in df.columns else 1
    if last['close'] > prev['close'] and vol_ratio > 1.2: score += 5
    if last['close'] < prev['close'] and vol_ratio > 1.2: score -= 5
    return max(0, min(100, score))

def analyze_volume(df):
    if 'VolMA5' not in df.columns: return "無量能資料"
    last = df.iloc[-1]
    vol_ratio = last['volume'] / last['VolMA5']
    if vol_ratio > 1.5: return "🔥 爆量"
    elif vol_ratio > 1.2: return "📈 放量"
    elif vol_ratio < 0.6: return "❄️ 窒息量"
    elif vol_ratio < 0.8: return "📉 量縮"
    else: return "⚖️ 量平"

def analyze_signals(df):
    if len(df) < 2: return ["資料不足"]
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []
    if 'MA5' in df.columns and 'MA20' in df.columns:
        if last['MA5'] > last['MA20'] > last['MA60']: signals.append("🔥 **趨勢**：多頭排列")
        elif last['MA5'] < last['MA20'] < last['MA60']: signals.append("❄️ **趨勢**：空頭排列")
        if prev['MA5'] < prev['MA20'] and last['MA5'] > last['MA20']: signals.append("✨ **均線金叉**：5日穿月線")
        elif prev['MA5'] > prev['MA20'] and last['MA5'] < last['MA20']: signals.append("💀 **均線死叉**：5日破月線")
    if 'K' in df.columns and 'D' in df.columns:
        if last['K'] > last['D'] and prev['K'] < prev['D']: signals.append(f"📈 **KD金叉**")
        elif last['K'] < last['D'] and prev['K'] > prev['D']: signals.append(f"📉 **KD死叉**")
    if 'Hist' in df.columns:
        if last['Hist'] > 0 and prev['Hist'] < 0: signals.append("🟢 **MACD翻紅**")
        elif last['Hist'] < 0 and prev['Hist'] > 0: signals.append("🔴 **MACD翻綠**")
    if 'RSI' in df.columns:
        if last['RSI'] > 75: signals.append(f"⚠️ **RSI過熱**")
        elif last['RSI'] < 25: signals.append(f"💎 **RSI超賣**")
    return signals if signals else ["⚖️ 盤整中"]

def generate_dual_strategy(df):
    if len(df) < 60: return None, None
    last = df.iloc[-1]
    last_close = last['close']
    score = calculate_score(df)
    vol_status = analyze_volume(df)
    
    checklist = {
        "站上月線": last_close > last['MA20'], "KD金叉向上": last['K'] > last['D'],
        "MACD偏多": last['Hist'] > 0, "量能健康": "量" in vol_status, "RSI安全": 20 < last['RSI'] < 75
    }
    short_term = {"title": "中性觀望", "icon": "⚖️", "color": "gray", "action": "觀望", "score": score, "vol": vol_status, "desc": "多空不明"}
    sl_short = last['MA20'] if 'MA20' in df.columns else last_close * 0.9
    tp_short = last['BB_Up'] if 'BB_Up' in df.columns else last_close * 1.1

    if last_close > last['MA20']:
        short_term.update({"title": "短多操作", "icon": "⚡", "color": "green", "action": "拉回佈局", "desc": "股價站上月線，短線強勢。"})
        if last['RSI'] > 75: short_term.update({"title": "短線過熱", "icon": "🔥", "color": "orange", "action": "分批獲利", "desc": "RSI過高。"})
    elif last_close < last['MA20']:
        short_term.update({"title": "短線偏空", "icon": "📉", "color": "red", "action": "反彈減碼", "desc": "跌破月線，短線轉弱。"})
        tp_short = last['MA20']
    
    short_term["stop_loss"] = f"{sl_short:.2f}"
    short_term["take_profit"] = f"{tp_short:.2f}"
    short_term["checklist"] = checklist

    long_term = {"title": "中性持有", "icon": "🐢", "color": "gray", "action": "續抱", "desc": "趨勢盤整"}
    sl_long = last['MA60'] if 'MA60' in df.columns else last_close * 0.85
    tp_long = df['high'].tail(120).max()
    if last_close > last['MA60']:
        long_term.update({"title": "長線多頭", "icon": "🚀", "color": "green", "action": "波段續抱", "desc": "站穩季線，長多格局。"})
    elif last_close < last['MA60']:
        long_term.update({"title": "長線轉弱", "icon": "❄️", "color": "red", "action": "保守應對", "desc": "跌破季線，需提防反轉。"})
        tp_long = last['MA60']
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
# 6. 主程式介面
# ==========================================
st.set_page_config(page_title="股票技術分析儀表板", layout="wide")
st.title("📈 股票技術分析儀表板")

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
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 K線圖", "💡 訊號診斷", "📐 黃金分割", "💰 籌碼分析"])

    with tab1:
        time_period = st.radio("範圍：", ["1個月", "3個月", "半年", "1年"], index=1, horizontal=True)
        if time_period == "1個月": plot_df = df.tail(20)
        elif time_period == "3個月": plot_df = df.tail(60)
        elif time_period == "半年": plot_df = df.tail(120)
        else: plot_df = df.tail(240)

        c1, c2 = st.columns(2)
        with c1: mas = st.multiselect("均線", ["MA5","MA10","MA20","MA60"], ["MA5","MA20","MA60"])
        with c2: inds = st.multiselect("副圖", ["Volume","KD","MACD","RSI"], ["Volume","KD"])

        add_plots = []
        colors = {'MA5':'orange', 'MA10':'cyan', 'MA20':'purple', 'MA60':'green'}
        for ma in mas:
            if ma in plot_df.columns: add_plots.append(mpf.make_addplot(plot_df[ma], panel=0, color=colors[ma], width=1.0))
        
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

        try:
            fig, ax = mpf.plot(plot_df, type='candle', style='yahoo', volume=vol, addplot=add_plots, returnfig=True, panel_ratios=tuple([2]+[1]*pid), figsize=(10, 8), warn_too_much_data=10000)
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
                    st.write("**✅ 多空健檢**")
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

    # === Tab 4: 籌碼分析 (三大法人) ===
    with tab4:
        st.subheader("💰 三大法人買賣超 (單位：張)")
        df_inst = get_institutional_data(stock_code)
        
        if not df_inst.empty:
            chart_data = df_inst.set_index("日期")[["外資", "投信", "自營商"]]
            st.bar_chart(chart_data)
            st.dataframe(df_inst.style.format({
                "外資": "{:,.0f}", "投信": "{:,.0f}", "自營商": "{:,.0f}", "合計": "{:,.0f}"
            }).applymap(lambda x: 'color: red' if x > 0 else 'color: green', subset=['外資','投信','自營商','合計']))
            st.caption("註：數據來源為 Yahoo 股市，僅供參考。紅色買超，綠色賣超。")
        else:
            # 如果真的因為 IP 封鎖抓不到，提供一個外部按鈕給使用者
            st.warning("⚠️ 無法自動抓取籌碼資料 (可能為 ETF 或 IP 限制)。")
            st.markdown(f"👉 [點此前往 Yahoo 股市查看 {stock_code} 籌碼](https://tw.stock.yahoo.com/quote/{stock_code}/institutional-trading)")
