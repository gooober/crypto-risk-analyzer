import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import os

# ========= AI Integration ==========
def ai_trade_commentary(symbol, price, vwap, supertrend, signal, stop, balance, leverage, position_size):
    try:
        import openai
        openai.api_key = st.session_state.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
        prompt = (
            f"You are a trading assistant for a day trader with ${balance} and {leverage}x leverage. "
            f"They want to risk only ${position_size} on this trade. "
            f"Symbol: {symbol}. Price: {price:.2f}, VWAP: {vwap:.2f}, SuperTrend: {supertrend:.2f}. "
            f"Signal: {signal}. Stop Loss: {stop:.2f}. "
            f"Give clear advice on 1) if this is a smart trade for this account size, "
            f"2) why or why not to enter now, 3) any warnings about risk, 4) a one-sentence summary."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=120
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"AI commentary not available ({e})"

# ========= Order Flow ==========
def get_order_flow(symbol):
    try:
        url = f"https://api.bybit.com/v5/market/trade?category=linear&symbol={symbol}&limit=100"
        trades = requests.get(url, timeout=5).json()['result']['list']
        buy_vol, sell_vol = 0, 0
        for trade in trades:
            qty = float(trade['qty'])
            if trade['side'] == 'Buy':
                buy_vol += qty
            else:
                sell_vol += qty
        imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol) * 100 if buy_vol + sell_vol else 0
        return round(buy_vol,2), round(sell_vol,2), round(imbalance,2)
    except:
        return np.nan, np.nan, np.nan

# ========= Indicators ==========
def calculate_vwap(klines):
    volumes = [float(k[5]) for k in klines]
    typicals = [(float(k[2]) + float(k[3]) + float(k[4]))/3 for k in klines]
    tp_vol = [t*v for t, v in zip(typicals, volumes)]
    cum_tp_vol = np.sum(tp_vol)
    cum_vol = np.sum(volumes)
    vwap = cum_tp_vol / cum_vol if cum_vol else np.nan
    return vwap

def calculate_supertrend(df, period=10, multiplier=3):
    hl2 = (df['high'] + df['low']) / 2
    atr = pd.Series(df['high'] - df['low']).rolling(period).mean()
    final_upperband = hl2 + (multiplier * atr)
    final_lowerband = hl2 - (multiplier * atr)
    supertrend = [np.nan]*len(df)
    direction = [1]
    for i in range(1, len(df)):
        if df['close'][i] > final_upperband[i-1]:
            direction.append(1)
        elif df['close'][i] < final_lowerband[i-1]:
            direction.append(-1)
        else:
            direction.append(direction[-1])
        if direction[-1] == 1:
            supertrend[i] = max(final_lowerband[i], supertrend[i-1] if i > 0 and direction[i-1] == 1 else final_lowerband[i])
        else:
            supertrend[i] = min(final_upperband[i], supertrend[i-1] if i > 0 and direction[i-1] == -1 else final_upperband[i])
    supertrend = pd.Series(supertrend).fillna(method='bfill').to_list()
    return supertrend, direction

def calculate_rsi(closes, period=14):
    closes = np.array(closes)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = gains[-period:].mean()
    avg_loss = losses[-period:].mean() if losses[-period:].mean() > 0 else 1
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(closes, fast=12, slow=26):
    closes = np.array(closes)
    if len(closes) < slow:
        return np.nan
    ema_fast = pd.Series(closes).ewm(span=fast, adjust=False).mean().iloc[-1]
    ema_slow = pd.Series(closes).ewm(span=slow, adjust=False).mean().iloc[-1]
    return (ema_fast - ema_slow) / ema_slow * 100

def calculate_atr(highs, lows, closes, period=14):
    tr = np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1]))
    return pd.Series(tr).rolling(period).mean().iloc[-1]

def get_trend(symbol, interval="60"):
    # 1h trend (by default), fallback OKX
    try:
        url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit=50"
        klines = requests.get(url, timeout=5).json()['result']['list']
        closes = [float(k[4]) for k in klines]
        trend = "UP" if closes[-1] > closes[0] else "DOWN"
        return trend
    except:
        return "Unknown"

# ========= Data Fetcher (with fallbacks) ==========
def get_enhanced_data(symbol):
    okx_symbol = symbol.replace("USDT", "-USDT")
    try:
        url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}"
        resp = requests.get(url, timeout=5)
        ticker_data = resp.json()['result']['list'][0]
        kline_url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval=1&limit=100"
        klines = requests.get(kline_url, timeout=5).json()['result']['list']
        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        vwap = calculate_vwap(klines)
        df = pd.DataFrame({"close": closes, "high": highs, "low": lows})
        supertrend, direction = calculate_supertrend(df)
        rsi = calculate_rsi(closes)
        macd = calculate_macd(closes)
        atr = calculate_atr(np.array(highs), np.array(lows), np.array(closes))
        return {
            'data_source': 'Bybit Perpetuals',
            'price': float(ticker_data['lastPrice']),
            'vwap': round(vwap, 2),
            'supertrend': supertrend,
            'supertrend_dir': direction,
            'signal': "Buy" if direction[-1] == 1 and closes[-1] > supertrend[-1] else "Sell",
            'stop_loss': round(supertrend[-1], 2),
            'rsi': round(rsi,2),
            'macd': round(macd,2),
            'atr': round(atr,2),
            'trend_1h': get_trend(symbol, "60"),
            'trend_5m': get_trend(symbol, "5"),
            'price_series': closes,
            'highs': highs,
            'lows': lows,
        }
    except Exception as e1:
        st.sidebar.warning(f"Bybit API failed: {str(e1)[:60]}")
    # fallback...
    st.sidebar.error("All APIs failed. Showing demo/offline mode.")
    return {
        'data_source': f'Demo Mode (All APIs failed)',
        'price': np.nan,
        'vwap': np.nan,
        'supertrend': [np.nan],
        'supertrend_dir': [0],
        'signal': "No Signal",
        'stop_loss': np.nan,
        'rsi': np.nan,
        'macd': np.nan,
        'atr': np.nan,
        'trend_1h': "Unknown",
        'trend_5m': "Unknown",
        'price_series': [],
        'highs': [],
        'lows': [],
    }

# ========= Trade Journal ==========
if 'journal' not in st.session_state:
    st.session_state.journal = []

def record_trade(symbol, action, qty, entry, stop, tp, result):
    st.session_state.journal.append({
        "symbol": symbol,
        "action": action,
        "qty": qty,
        "entry": entry,
        "stop": stop,
        "tp": tp,
        "result": result,
        "timestamp": datetime.now().strftime('%H:%M')
    })

def get_stats():
    df = pd.DataFrame(st.session_state.journal)
    if df.empty:
        return "No trades yet."
    wins = df[df['result'] == 'WIN']
    losses = df[df['result'] == 'LOSS']
    winrate = 100 * len(wins) / len(df) if len(df) else 0
    return f"Trades: {len(df)}, Winrate: {winrate:.1f}%, Net PnL: (demo) {len(wins)-len(losses)}R"

# ========= News Feed ==========
def get_news():
    try:
        url = "https://cryptopanic.com/api/v1/posts/?auth_token=demo&public=true"
        r = requests.get(url, timeout=5)
        news = r.json()['results']
        return [f"[{n['title']}]({n['url']})" for n in news[:8]]
    except:
        return ["(Demo news feed unavailable)"]

# ========= UI ==========
st.set_page_config(
    page_title="Pro Small Account Day Trading Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🚦 Pro Small Account Day Trading Analyzer")

with st.sidebar:
    st.header("Account & Risk")
    default_balance = st.number_input("Account Size (USD)", min_value=10.0, value=50.0, step=1.0)
    max_risk_per_trade = st.number_input("Max Risk % per Trade", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
    leverage = st.slider("Leverage", 1, 20, 5)
    ai_enabled = st.checkbox("Enable AI commentary (OpenAI key)", value=False)
    if ai_enabled:
        openai_key = st.text_input("Paste OpenAI API key", type="password")
        if openai_key:
            st.session_state["openai_api_key"] = openai_key
    st.header("Symbols & Refresh")
    symbols = st.multiselect(
        "Select Cryptocurrencies",
        ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"],
        default=["BTCUSDT"]
    )
    refresh_rate = st.slider("Refresh Rate (seconds)", 5, 60, 10)
    if st.button("🔄 Refresh Now"):
        st.session_state.last_update = datetime.now()
        st.rerun()

tabs = st.tabs(["Dashboard", "Trade Planner", "Journal & Stats", "News"])

with tabs[0]:
    st.markdown("### Dashboard (All Symbols)")
    for sym in symbols:
        data = get_enhanced_data(sym)
        buy_vol, sell_vol, imbalance = get_order_flow(sym)
        st.subheader(f"{sym}")
        st.metric("Price", f"${data['price']:.2f}" if not np.isnan(data['price']) else "N/A")
        st.metric("VWAP", f"{data['vwap']:.2f}" if not np.isnan(data['vwap']) else "N/A")
        st.metric("SuperTrend", f"{data['supertrend'][-1]:.2f}" if not np.isnan(data['supertrend'][-1]) else "N/A")
        st.metric("Signal", data['signal'])
        st.metric("Stop Loss", f"${data['stop_loss']:.2f}" if not np.isnan(data['stop_loss']) else "N/A")
        st.metric("ATR(14)", f"{data['atr']:.2f}" if not np.isnan(data['atr']) else "N/A")
        st.metric("Order Flow Imb.", f"{imbalance}%" if not np.isnan(imbalance) else "N/A")
        st.metric("Trend (1h)", data['trend_1h'])
        st.metric("Trend (5m)", data['trend_5m'])
        st.write(f"BuyVol: {buy_vol}, SellVol: {sell_vol}")
        # Order Flow Bar
        st.progress(int((imbalance+100)/2) if not np.isnan(imbalance) else 50, text="Order Flow Bias")
        # Chart
        if data['price_series'] and not np.isnan(data['vwap']):
            price_series = data['price_series']
            x = [datetime.now() - pd.Timedelta(minutes=len(price_series)-i-1) for i in range(len(price_series))]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=price_series, name="Price"))
            fig.add_trace(go.Scatter(x=x, y=[data['vwap']]*len(price_series), name="VWAP", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=x, y=data['supertrend'], name="SuperTrend", line=dict(color='green' if data['signal']=='Buy' else 'red')))
            st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.markdown("### Trade Planner")
    for sym in symbols:
        data = get_enhanced_data(sym)
        buy_vol, sell_vol, imbalance = get_order_flow(sym)
        st.subheader(f"{sym} Trade Planner")
        buy_in = st.number_input(f"Your Buy-in Amount for {sym} ($)", min_value=1.0, value=default_balance, step=1.0, key=f"buyin_{sym}")
        entry_price = data['price']
        stop = data['stop_loss']
        # ATR-based TP/SL
        if not np.isnan(entry_price) and not np.isnan(data['atr']):
            tp1 = round(entry_price + data['atr'] if data['signal'] == "Buy" else entry_price - data['atr'], 2)
            tp2 = round(entry_price + 2*data['atr'] if data['signal'] == "Buy" else entry_price - 2*data['atr'], 2)
            st.write(f"Suggested Take Profit 1: {tp1}, Take Profit 2: {tp2}")
        # Multi-timeframe filter & confluence
        confluence = (
            data['signal'] == "Buy" and data['trend_1h'] == "UP" and imbalance > 0
            or data['signal'] == "Sell" and data['trend_1h'] == "DOWN" and imbalance < 0
        )
        st.info(f"Confluence Signal: {'🟢 STRONG' if confluence else '⚠️ WEAK'}")
        # Position sizing
        risk_dollars = buy_in * max_risk_per_trade / 100
        if not np.isnan(entry_price) and not np.isnan(stop) and abs(entry_price - stop) > 0:
            pos_size = round(risk_dollars / abs(entry_price - stop) * entry_price / leverage, 2)
            st.info(f"**Suggested Position Size:** ${pos_size} (at {leverage}x, risking {max_risk_per_trade}% of your buy-in)")
            trade_advice = (
                f"{'🟢 **BUY NOW**' if data['signal']=='Buy' and confluence else '🔴 **SELL/AVOID**'} at ${entry_price:.2f}, "
                f"Stop Loss: ${stop:.2f}."
            )
            st.success(trade_advice)
        else:
            st.info("Position size not available (price or stop missing).")
        # AI Commentary
        if ai_enabled and not np.isnan(entry_price) and not np.isnan(stop) and not np.isnan(data['vwap']) and not np.isnan(data['supertrend'][-1]):
            ai_result = ai_trade_commentary(
                sym, entry_price, data['vwap'], data['supertrend'][-1], data['signal'], stop,
                buy_in, leverage, pos_size if 'pos_size' in locals() else 0
            )
            st.warning("**AI Commentary:**\n" + ai_result)
        # Journal entry demo (manual add):
        if st.button(f"Record Trade ({sym})"):
            result = st.radio("Result", ["WIN", "LOSS"], key=f"r_{sym}")
            record_trade(sym, data['signal'], pos_size if 'pos_size' in locals() else 0, entry_price, stop, tp1 if 'tp1' in locals() else 0, result)
            st.success("Trade recorded.")
        # Chart
        if data['price_series'] and not np.isnan(data['vwap']):
            price_series = data['price_series']
            x = [datetime.now() - pd.Timedelta(minutes=len(price_series)-i-1) for i in range(len(price_series))]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=price_series, name="Price"))
            fig.add_trace(go.Scatter(x=x, y=[data['vwap']]*len(price_series), name="VWAP", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=x, y=data['supertrend'], name="SuperTrend", line=dict(color='green' if data['signal']=='Buy' else 'red')))
            st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.markdown("### Trade Journal & Stats")
    st.write(get_stats())
    if st.session_state.journal:
        df = pd.DataFrame(st.session_state.journal)
        st.dataframe(df)

with tabs[3]:
    st.markdown("### Real-time News")
    for n in get_news():
        st.write(f"- {n}")

current_time = datetime.now()
if (current_time - st.session_state.get("last_update",datetime.now())).total_seconds() >= refresh_rate:
    st.session_state.last_update = current_time
    st.rerun()
