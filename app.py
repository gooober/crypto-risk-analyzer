import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime
import plotly.graph_objs as go

# ---------- SETTINGS ----------
st.set_page_config(page_title="US Crypto Day Trading App", layout="wide")
ASSETS = ['BTC-USD', 'ETH-USD']
CB_SYMBOLS = {'BTC-USD': 'BTC-USD', 'ETH-USD': 'ETH-USD'}
BINA_SYMBOLS = {'BTC-USD': 'BTCUSDT', 'ETH-USD': 'ETHUSDT'}

# ---------- API HELPERS ----------

def get_price_coinbase(symbol):
    try:
        resp = requests.get(f'https://api.coinbase.com/v2/prices/{symbol}/spot', timeout=5)
        return float(resp.json()['data']['amount'])
    except Exception as e:
        return None

def get_price_binance_us(symbol):
    binance_symbol = BINA_SYMBOLS[symbol]
    try:
        resp = requests.get(f'https://api.binance.us/api/v3/ticker/price?symbol={binance_symbol}', timeout=5)
        return float(resp.json()['price'])
    except Exception as e:
        return None

def fetch_klines_binance_us(symbol, interval='1m', limit=500):
    binance_symbol = BINA_SYMBOLS[symbol]
    url = f"https://api.binance.us/api/v3/klines?symbol={binance_symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=['Open time','Open','High','Low','Close','Volume',
                                     'Close time','Quote asset vol','Num trades',
                                     'Taker buy base','Taker buy quote','Ignore'])
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df['Close'] = df['Close'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Open'] = df['Open'].astype(float)
    return df[['Open time', 'Open', 'High', 'Low', 'Close']]

# ---------- INDICATORS ----------
def EMA(series, period=9):
    return series.ewm(span=period, adjust=False).mean()

def RSI(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def MACD(series, fast=12, slow=26, signal=9):
    macd = EMA(series, fast) - EMA(series, slow)
    signal_line = EMA(macd, signal)
    return macd, signal_line

def advanced_signal(df):
    df['EMA9'] = EMA(df['Close'], 9)
    df['EMA21'] = EMA(df['Close'], 21)
    df['RSI'] = RSI(df['Close'])
    macd, macd_signal = MACD(df['Close'])
    df['MACD'] = macd
    df['MACD_signal'] = macd_signal
    last = df.iloc[-1]
    buy = (last['EMA9'] > last['EMA21']) and (last['RSI'] < 70) and (last['MACD'] > last['MACD_signal'])
    sell = (last['EMA9'] < last['EMA21']) or (last['RSI'] > 70) or (last['MACD'] < last['MACD_signal'])
    return "Buy" if buy else ("Sell" if sell else "Hold"), last

# ---------- AI TRADING SIGNAL FUNCTION ----------
def ai_trade_signal(latest_price, recent_df, trade_amount_usd, leverage):
    # Calculate indicators
    recent_df['EMA9'] = recent_df['Close'].ewm(span=9, adjust=False).mean()
    recent_df['EMA21'] = recent_df['Close'].ewm(span=21, adjust=False).mean()
    recent_df['RSI'] = RSI(recent_df['Close'])
    macd, macd_signal = MACD(recent_df['Close'])
    recent_df['MACD'] = macd
    recent_df['MACD_signal'] = macd_signal
    
    # Get last row data
    last = recent_df.iloc[-1]
    prev = recent_df.iloc[-2] if len(recent_df) > 1 else last
    
    # Calculate volatility for dynamic stop loss
    recent_volatility = recent_df['Close'].tail(20).pct_change().std() * 100
    stop_distance_pct = max(0.5, min(3.0, recent_volatility * 1.5))
    
    # Determine signal
    signal = 'Hold'
    reasoning = ""
    
    # Buy conditions
    if (last['EMA9'] > last['EMA21'] and prev['EMA9'] <= prev['EMA21'] and 
        last['RSI'] < 65 and last['MACD'] > last['MACD_signal']):
        signal = 'Buy'
        reasoning = f"EMA crossover with RSI at {last['RSI']:.1f} and positive MACD momentum."
        
    # Sell conditions
    elif (last['EMA9'] < last['EMA21'] and prev['EMA9'] >= prev['EMA21'] and 
          last['RSI'] > 35 and last['MACD'] < last['MACD_signal']):
        signal = 'Sell'
        reasoning = f"EMA crossunder with RSI at {last['RSI']:.1f} and negative MACD momentum."
        
    # Strong buy on oversold bounce
    elif last['RSI'] < 30 and last['Close'] > prev['Close'] and last['EMA9'] > prev['EMA9']:
        signal = 'Buy'
        reasoning = f"Oversold bounce with RSI at {last['RSI']:.1f} and price momentum turning positive."
        
    # Strong sell on overbought reversal
    elif last['RSI'] > 70 and last['Close'] < prev['Close'] and last['EMA9'] < prev['EMA9']:
        signal = 'Sell'
        reasoning = f"Overbought reversal with RSI at {last['RSI']:.1f} and price momentum turning negative."
        
    else:
        reasoning = f"No clear setup. RSI at {last['RSI']:.1f}, waiting for better entry conditions."
    
    # Calculate stop loss and take profit
    if signal == 'Buy':
        stop_loss = latest_price * (1 - stop_distance_pct / 100)
        risk_amount = latest_price - stop_loss
        take_profit = latest_price + (risk_amount * 5)
    elif signal == 'Sell':
        stop_loss = latest_price * (1 + stop_distance_pct / 100)
        risk_amount = stop_loss - latest_price
        take_profit = latest_price - (risk_amount * 5)
    else:
        stop_loss = None
        take_profit = None
    
    return signal, stop_loss, take_profit, reasoning

# ---------- UI ----------
st.title("🇺🇸 Small Account US Crypto Day Trading Tool")
st.caption("With AI-powered trading signals and risk management")

st.sidebar.header("⚙️ Trade Settings")
asset = st.sidebar.selectbox("Asset", ASSETS)
leverage = st.sidebar.slider("Leverage", 1, 20, 5)
bankroll = st.sidebar.number_input("Total Bankroll (USD)", 10.0, 10000.0, 100.0)
trade_risk = st.sidebar.slider("Risk % per trade", 1, 10, 5)
auto_refresh = st.sidebar.checkbox("Auto-refresh", False)
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 5, 60, 15)

# ---------- LIVE PRICE FEED ----------
cb_price = get_price_coinbase(CB_SYMBOLS[asset])
binance_price = get_price_binance_us(asset)
price = cb_price if cb_price else binance_price

col1, col2, col3 = st.columns(3)
col1.metric(label=f"{asset} Coinbase Price", value=f"${cb_price:,.2f}" if cb_price else "N/A")
col2.metric(label=f"{asset} Binance.US Price", value=f"${binance_price:,.2f}" if binance_price else "N/A")
if cb_price and binance_price:
    spread = abs(cb_price - binance_price)
    spread_pct = (spread / min(cb_price, binance_price)) * 100
    col3.metric(label="Price Spread", value=f"${spread:.2f}", delta=f"{spread_pct:.2f}%")
    if spread > 2:
        st.warning("⚠️ Price mismatch detected! Check both exchanges before trading.")

if not price:
    st.error("⚠️ Could not get live price from either exchange. Try again later.")
    st.stop()

# ---------- DATA + SIGNALS ----------
with st.spinner("Loading historical data..."):
    df = fetch_klines_binance_us(asset)
    signal, last_row = advanced_signal(df)
    
    # Display basic signal info
    st.markdown("### 📊 Technical Analysis")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Basic Signal", signal, 
                delta="Bullish" if signal == "Buy" else "Bearish" if signal == "Sell" else "Neutral")
    col2.metric("RSI", f"{last_row['RSI']:.1f}",
                delta="Oversold" if last_row['RSI'] < 30 else "Overbought" if last_row['RSI'] > 70 else "Normal")
    col3.metric("EMA9", f"${last_row['EMA9']:,.2f}")
    col4.metric("MACD", f"{last_row['MACD']:.4f}")

    # PLOT
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['Open time'], open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name="Candles"))
    fig.add_trace(go.Scatter(x=df['Open time'], y=df['EMA9'], line=dict(width=1, color='orange'), name="EMA9"))
    fig.add_trace(go.Scatter(x=df['Open time'], y=df['EMA21'], line=dict(width=1, color='blue'), name="EMA21"))
    fig.update_layout(height=400, xaxis_rangeslider_visible=False, showlegend=True,
                      title=f"{asset} Price Chart with EMAs")
    st.plotly_chart(fig, use_container_width=True)

# ---------- TRADE CALCULATIONS ----------
trade_size = bankroll * (trade_risk/100)
position = trade_size * leverage / price

st.markdown("### 💰 Position Sizing")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Risk per Trade", f"${trade_size:.2f}")
    st.caption(f"{trade_risk}% of ${bankroll:.2f} bankroll")
with col2:
    st.metric("Position Size", f"{position:.6f} {asset.split('-')[0]}")
    st.caption(f"With {leverage}x leverage")
with col3:
    st.metric("Notional Value", f"${trade_size * leverage:.2f}")
    st.caption("Total exposure")

# ---------- AI SIGNAL ----------
st.markdown("### 🤖 AI Trading Signal")
signal_ai, stop_ai, tp_ai, reasoning = ai_trade_signal(price, df, trade_size, leverage)

if signal_ai and signal_ai != 'Hold':
    # Calculate risk/reward metrics
    if signal_ai == 'Buy':
        risk = price - stop_ai
        reward = tp_ai - price
    else:
        risk = stop_ai - price
        reward = price - tp_ai
    
    risk_reward_ratio = reward / risk if risk > 0 else 0
    
    # Display AI signal with visual emphasis
    if signal_ai == 'Buy':
        st.success(f"### 📈 AI Signal: **{signal_ai}**")
    else:
        st.error(f"### 📉 AI Signal: **{signal_ai}**")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Entry Price", f"${price:,.2f}")
    with col2:
        st.metric("Stop Loss", f"${stop_ai:,.2f}", 
                  delta=f"-${risk:.2f} ({(risk/price)*100:.1f}%)")
    with col3:
        st.metric("Take Profit", f"${tp_ai:,.2f}", 
                  delta=f"+${reward:.2f} ({(reward/price)*100:.1f}%)")
    with col4:
        st.metric("Risk:Reward", f"1:{risk_reward_ratio:.1f}")
    
    st.info(f"**Reasoning:** {reasoning}")
    
    # Risk management details
    with st.expander("📊 Risk Management Details"):
        st.write(f"**Position Size:** {position:.6f} {asset.split('-')[0]}")
        st.write(f"**Dollar Risk:** ${trade_size:.2f}")
        st.write(f"**Potential Loss:** ${trade_size:.2f} ({trade_risk}% of bankroll)")
        st.write(f"**Potential Gain:** ${trade_size * risk_reward_ratio:.2f} ({trade_risk * risk_reward_ratio:.1f}% of bankroll)")
        
elif signal_ai == 'Hold':
    st.warning(f"### ⏸️ AI Signal: **{signal_ai}**")
    st.info(f"**Reasoning:** {reasoning}")
else:
    st.info("AI signal calculation error. Please refresh the page.")

# ---------- TRADING CHECKLIST ----------
with st.expander("✅ Pre-Trade Checklist"):
    st.write("Before entering any trade, ensure you have:")
    st.write("- [ ] Verified the signal on multiple timeframes")
    st.write("- [ ] Checked for major news or events")
    st.write("- [ ] Confirmed adequate account balance")
    st.write("- [ ] Set stop loss and take profit orders")
    st.write("- [ ] Calculated position size correctly")
    st.write("- [ ] Accepted the risk of loss")

# ---------- AUTO REFRESH ----------
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

st.caption("🚨 **Disclaimer:** This is not financial advice. Trading cryptocurrencies involves substantial risk of loss. Use a demo account for practice. Past performance does not guarantee future results.")
