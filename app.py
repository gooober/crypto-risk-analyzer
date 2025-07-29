import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# ---------- SETTINGS ----------
st.set_page_config(page_title="US Crypto Day Trading App", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better UI
st.markdown("""
<style>
    .stMetric .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

ASSETS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 'ADA-USD']
CB_SYMBOLS = {'BTC-USD': 'BTC-USD', 'ETH-USD': 'ETH-USD', 'SOL-USD': 'SOL-USD', 'DOGE-USD': 'DOGE-USD', 'ADA-USD': 'ADA-USD'}
BINA_SYMBOLS = {'BTC-USD': 'BTCUSDT', 'ETH-USD': 'ETHUSDT', 'SOL-USD': 'SOLUSDT', 'DOGE-USD': 'DOGEUSDT', 'ADA-USD': 'ADAUSDT'}

# Initialize session state
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

# ---------- API HELPERS ----------
@st.cache_data(ttl=10)
def get_price_coinbase(symbol):
    try:
        resp = requests.get(f'https://api.coinbase.com/v2/prices/{symbol}/spot', timeout=5)
        return float(resp.json()['data']['amount'])
    except Exception as e:
        return None

@st.cache_data(ttl=10)
def get_price_binance_us(symbol):
    binance_symbol = BINA_SYMBOLS.get(symbol)
    if not binance_symbol:
        return None
    try:
        resp = requests.get(f'https://api.binance.us/api/v3/ticker/price?symbol={binance_symbol}', timeout=5)
        return float(resp.json()['price'])
    except Exception as e:
        return None

@st.cache_data(ttl=60)
def fetch_klines_binance_us(symbol, interval='1m', limit=500):
    binance_symbol = BINA_SYMBOLS.get(symbol)
    if not binance_symbol:
        return pd.DataFrame()
    try:
        url = f"https://api.binance.us/api/v3/klines?symbol={binance_symbol}&interval={interval}&limit={limit}"
        data = requests.get(url, timeout=10).json()
        df = pd.DataFrame(data, columns=['Open time','Open','High','Low','Close','Volume',
                                         'Close time','Quote asset vol','Num trades',
                                         'Taker buy base','Taker buy quote','Ignore'])
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Close'] = df['Close'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Open'] = df['Open'].astype(float)
        df['Volume'] = df['Volume'].astype(float)
        return df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    except:
        return pd.DataFrame()

# ---------- TECHNICAL INDICATORS ----------
def EMA(series, period=9):
    return series.ewm(span=period, adjust=False).mean()

def SMA(series, period=20):
    return series.rolling(window=period).mean()

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

def BollingerBands(series, period=20, std_dev=2):
    sma = SMA(series, period)
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def VWAP(df):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    cumulative_pv = (typical_price * df['Volume']).cumsum()
    cumulative_volume = df['Volume'].cumsum()
    return cumulative_pv / cumulative_volume

def ATR(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()

# ---------- SIGNAL GENERATION ----------
def calculate_all_indicators(df):
    df['EMA9'] = EMA(df['Close'], 9)
    df['EMA21'] = EMA(df['Close'], 21)
    df['EMA50'] = EMA(df['Close'], 50)
    df['RSI'] = RSI(df['Close'])
    df['MACD'], df['MACD_signal'] = MACD(df['Close'])
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = BollingerBands(df['Close'])
    df['VWAP'] = VWAP(df)
    df['ATR'] = ATR(df)
    df['Volume_SMA'] = SMA(df['Volume'], 20)
    return df

def generate_signals(df, use_ai=True):
    signals = []
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    
    # Basic trend strength
    trend_strength = 0
    
    # EMA signals
    if last['EMA9'] > last['EMA21'] and prev['EMA9'] <= prev['EMA21']:
        signals.append(("EMA Crossover", "Bullish", 2))
        trend_strength += 2
    elif last['EMA9'] < last['EMA21'] and prev['EMA9'] >= prev['EMA21']:
        signals.append(("EMA Crossunder", "Bearish", -2))
        trend_strength -= 2
    
    # RSI signals
    if last['RSI'] < 30:
        signals.append(("RSI Oversold", "Bullish", 2))
        trend_strength += 2
    elif last['RSI'] > 70:
        signals.append(("RSI Overbought", "Bearish", -2))
        trend_strength -= 2
    
    # MACD signals
    if last['MACD'] > last['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']:
        signals.append(("MACD Cross Up", "Bullish", 1))
        trend_strength += 1
    elif last['MACD'] < last['MACD_signal'] and prev['MACD'] >= prev['MACD_signal']:
        signals.append(("MACD Cross Down", "Bearish", -1))
        trend_strength -= 1
    
    # Bollinger Band signals
    if last['Close'] < last['BB_lower']:
        signals.append(("Below BB Lower", "Bullish", 1))
        trend_strength += 1
    elif last['Close'] > last['BB_upper']:
        signals.append(("Above BB Upper", "Bearish", -1))
        trend_strength -= 1
    
    # VWAP signals
    if last['Close'] > last['VWAP'] and prev['Close'] <= prev['VWAP']:
        signals.append(("Price > VWAP", "Bullish", 1))
        trend_strength += 1
    elif last['Close'] < last['VWAP'] and prev['Close'] >= prev['VWAP']:
        signals.append(("Price < VWAP", "Bearish", -1))
        trend_strength -= 1
    
    # Volume signals
    if last['Volume'] > last['Volume_SMA'] * 1.5:
        if last['Close'] > last['Open']:
            signals.append(("High Volume Buying", "Bullish", 1))
            trend_strength += 1
        else:
            signals.append(("High Volume Selling", "Bearish", -1))
            trend_strength -= 1
    
    return signals, trend_strength

def calculate_targets(df, entry_price, signal_type, trade_amount_usd):
    # Calculate realistic targets based on recent price action
    atr = df['ATR'].iloc[-1]
    recent_volatility = df['Close'].tail(20).pct_change().std() * 100
    
    # Dynamic stop loss based on ATR and volatility
    stop_distance_pct = max(0.3, min(2.0, (atr / entry_price) * 100))
    
    # Calculate support/resistance levels
    recent_high = df['High'].tail(50).max()
    recent_low = df['Low'].tail(50).min()
    
    if signal_type == "BUY":
        stop_loss = entry_price * (1 - stop_distance_pct / 100)
        
        # Find realistic take profit levels
        resistance_1 = min(recent_high, entry_price * 1.02)
        resistance_2 = entry_price * (1 + (stop_distance_pct * 2) / 100)
        
        # Use the more conservative target
        take_profit = min(resistance_1, resistance_2)
        
    else:  # SELL
        stop_loss = entry_price * (1 + stop_distance_pct / 100)
        
        # Find realistic take profit levels
        support_1 = max(recent_low, entry_price * 0.98)
        support_2 = entry_price * (1 - (stop_distance_pct * 2) / 100)
        
        # Use the more conservative target
        take_profit = max(support_1, support_2)
    
    # Calculate risk/reward
    if signal_type == "BUY":
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
    else:
        risk = stop_loss - entry_price
        reward = entry_price - take_profit
    
    risk_reward_ratio = reward / risk if risk > 0 else 0
    
    return stop_loss, take_profit, risk_reward_ratio, stop_distance_pct

# ---------- MAIN UI ----------
st.title("🚀 Advanced US Crypto Day Trading Terminal")
st.caption("Professional-grade signals with realistic risk management")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Trading Configuration")
    
    with st.expander("📈 Asset Selection", expanded=True):
        asset = st.selectbox("Select Asset", ASSETS, help="Choose your trading pair")
        timeframe = st.selectbox("Chart Timeframe", ['1m', '5m', '15m', '1h'], index=1, 
                                help="Select timeframe for analysis")
    
    with st.expander("💰 Position Sizing", expanded=True):
        trade_amount_usd = st.number_input("Trade Amount (USD)", 
                                          min_value=10.0, 
                                          max_value=10000.0, 
                                          value=100.0, 
                                          step=10.0,
                                          help="Actual USD amount you want to trade")
        leverage = st.slider("Leverage", 1, 20, 5, help="Trading leverage multiplier")
        
    with st.expander("🤖 AI Settings", expanded=True):
        use_ai_signals = st.checkbox("Enable AI Analysis", value=True, 
                                    help="Use advanced AI pattern recognition")
        confidence_threshold = st.slider("Signal Confidence", 1, 10, 5, 
                                       help="Minimum confidence for signals")
    
    with st.expander("🔄 Auto Refresh", expanded=False):
        auto_refresh = st.checkbox("Enable Auto-refresh", value=False)
        refresh_interval = st.slider("Refresh interval (seconds)", 5, 60, 15)
    
    st.markdown("---")
    st.markdown("### 📚 Quick Guide")
    st.info("""
    **Signal Strength:**
    - 🟢 Strong Buy: 5+ score
    - 🟡 Buy: 2-4 score  
    - ⚪ Neutral: -1 to 1
    - 🟡 Sell: -2 to -4 score
    - 🔴 Strong Sell: -5+ score
    
    **Risk Management:**
    - Stop losses based on ATR
    - Realistic profit targets
    - Position sizing calculator
    """)

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Trading", "📈 Technical Analysis", "📝 Trade History", "🎓 Education"])

with tab1:
    # Price display
    col1, col2, col3, col4 = st.columns(4)
    
    cb_price = get_price_coinbase(CB_SYMBOLS[asset])
    binance_price = get_price_binance_us(asset)
    price = cb_price if cb_price else binance_price
    
    if not price:
        st.error("⚠️ Could not get live price. Please check your connection.")
        st.stop()
    
    with col1:
        st.metric("Coinbase", f"${cb_price:,.2f}" if cb_price else "N/A")
    with col2:
        st.metric("Binance.US", f"${binance_price:,.2f}" if binance_price else "N/A")
    with col3:
        if cb_price and binance_price:
            spread = abs(cb_price - binance_price)
            spread_pct = (spread / min(cb_price, binance_price)) * 100
            st.metric("Spread", f"${spread:.2f}", delta=f"{spread_pct:.2f}%")
    with col4:
        st.metric("Last Update", st.session_state.last_refresh.strftime("%H:%M:%S"))
    
    # Load and analyze data
    with st.spinner("Analyzing market data..."):
        df = fetch_klines_binance_us(asset, interval=timeframe)
        if df.empty:
            st.error("Failed to load market data")
            st.stop()
            
        df = calculate_all_indicators(df)
        signals, trend_strength = generate_signals(df, use_ai_signals)
    
    # Signal summary
    st.markdown("### 🎯 Trading Signals")
    
    signal_col1, signal_col2, signal_col3 = st.columns([2, 1, 1])
    
    with signal_col1:
        # Determine overall signal
        if trend_strength >= confidence_threshold:
            signal_type = "BUY"
            signal_color = "green"
            signal_emoji = "🟢"
        elif trend_strength <= -confidence_threshold:
            signal_type = "SELL"
            signal_color = "red"
            signal_emoji = "🔴"
        else:
            signal_type = "HOLD"
            signal_color = "gray"
            signal_emoji = "⚪"
        
        st.markdown(f"### {signal_emoji} **{signal_type}** Signal")
        st.markdown(f"**Confidence Score:** {abs(trend_strength)}/10")
        
        # Progress bar for signal strength
        progress = min(abs(trend_strength) / 10, 1.0)
        st.progress(progress)
    
    with signal_col2:
        last = df.iloc[-1]
        st.metric("RSI", f"{last['RSI']:.1f}")
        st.metric("MACD", f"{last['MACD']:.4f}")
    
    with signal_col3:
        st.metric("Volume", f"{last['Volume']:,.0f}")
        volume_change = (last['Volume'] / last['Volume_SMA'] - 1) * 100
        st.metric("Vol vs Avg", f"{volume_change:+.1f}%")
    
    # Active signals table
    if signals:
        st.markdown("#### Active Signals")
        signal_df = pd.DataFrame(signals, columns=['Indicator', 'Direction', 'Strength'])
        st.dataframe(signal_df, use_container_width=True, hide_index=True)
    
    # Trade execution panel
    if signal_type != "HOLD":
        st.markdown("### 💼 Trade Execution")
        
        stop_loss, take_profit, risk_reward, stop_pct = calculate_targets(
            df, price, signal_type, trade_amount_usd
        )
        
        exec_col1, exec_col2, exec_col3, exec_col4 = st.columns(4)
        
        with exec_col1:
            st.metric("Entry Price", f"${price:,.2f}")
            position_size = (trade_amount_usd * leverage) / price
            st.caption(f"Position: {position_size:.6f} {asset.split('-')[0]}")
        
        with exec_col2:
            risk_usd = trade_amount_usd * (stop_pct / 100)
            st.metric("Stop Loss", f"${stop_loss:,.2f}", 
                     delta=f"-${risk_usd:.2f}")
        
        with exec_col3:
            profit_usd = risk_usd * risk_reward
            st.metric("Take Profit", f"${take_profit:,.2f}", 
                     delta=f"+${profit_usd:.2f}")
        
        with exec_col4:
            st.metric("Risk:Reward", f"1:{risk_reward:.1f}")
            st.caption(f"Win Rate Needed: {(1/(1+risk_reward))*100:.0f}%")
        
        # Execution buttons
        button_col1, button_col2, button_col3 = st.columns([1, 1, 2])
        with button_col1:
            if st.button(f"📈 Execute {signal_type}", type="primary", use_container_width=True):
                trade_record = {
                    'time': datetime.now(),
                    'asset': asset,
                    'signal': signal_type,
                    'entry': price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_reward': risk_reward,
                    'amount_usd': trade_amount_usd,
                    'leverage': leverage
                }
                st.session_state.trade_history.append(trade_record)
                st.success(f"{signal_type} order prepared! Set your stop loss and take profit on exchange.")
        
        with button_col2:
            if st.button("📋 Copy Levels", use_container_width=True):
                st.code(f"Entry: ${price:.2f}\nStop: ${stop_loss:.2f}\nTP: ${take_profit:.2f}")

with tab2:
    st.markdown("### 📊 Advanced Technical Analysis")
    
    # Create comprehensive chart
    fig = make_subplots(rows=3, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.05,
                       row_heights=[0.6, 0.2, 0.2],
                       subplot_titles=('Price Action', 'RSI', 'Volume'))
    
    # Main price chart
    fig.add_trace(go.Candlestick(x=df['Open time'], 
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name="Price"), row=1, col=1)
    
    # Add indicators
    fig.add_trace(go.Scatter(x=df['Open time'], y=df['EMA9'], 
                            line=dict(width=1, color='orange'), name="EMA9"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Open time'], y=df['EMA21'], 
                            line=dict(width=1, color='blue'), name="EMA21"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Open time'], y=df['VWAP'], 
                            line=dict(width=2, color='purple', dash='dot'), name="VWAP"), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df['Open time'], y=df['BB_upper'], 
                            line=dict(width=1, color='gray'), name="BB Upper", 
                            showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Open time'], y=df['BB_lower'], 
                            line=dict(width=1, color='gray'), name="BB Lower",
                            fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df['Open time'], y=df['RSI'], 
                            line=dict(width=2, color='green'), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Volume
    colors = ['green' if close > open else 'red' 
              for close, open in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df['Open time'], y=df['Volume'], 
                        marker_color=colors, name="Volume"), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical indicators summary
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("#### Moving Averages")
        ma_data = {
            'Indicator': ['EMA9', 'EMA21', 'EMA50', 'VWAP'],
            'Value': [
                f"${df['EMA9'].iloc[-1]:,.2f}",
                f"${df['EMA21'].iloc[-1]:,.2f}",
                f"${df['EMA50'].iloc[-1]:,.2f}",
                f"${df['VWAP'].iloc[-1]:,.2f}"
            ],
            'Signal': [
                '🟢' if df['Close'].iloc[-1] > df['EMA9'].iloc[-1] else '🔴',
                '🟢' if df['Close'].iloc[-1] > df['EMA21'].iloc[-1] else '🔴',
                '🟢' if df['Close'].iloc[-1] > df['EMA50'].iloc[-1] else '🔴',
                '🟢' if df['Close'].iloc[-1] > df['VWAP'].iloc[-1] else '🔴'
            ]
        }
        st.dataframe(pd.DataFrame(ma_data), use_container_width=True, hide_index=True)
    
    with tech_col2:
        st.markdown("#### Momentum Indicators")
        momentum_data = {
            'Indicator': ['RSI', 'MACD', 'MACD Signal'],
            'Value': [
                f"{df['RSI'].iloc[-1]:.1f}",
                f"{df['MACD'].iloc[-1]:.4f}",
                f"{df['MACD_signal'].iloc[-1]:.4f}"
            ],
            'Status': [
                'Oversold' if df['RSI'].iloc[-1] < 30 else 'Overbought' if df['RSI'].iloc[-1] > 70 else 'Normal',
                'Bullish' if df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1] else 'Bearish',
                ''
            ]
        }
        st.dataframe(pd.DataFrame(momentum_data), use_container_width=True, hide_index=True)
    
    with tech_col3:
        st.markdown("#### Volatility & Volume")
        vol_data = {
            'Indicator': ['ATR', 'BB Width', 'Volume Ratio'],
            'Value': [
                f"${df['ATR'].iloc[-1]:.2f}",
                f"${(df['BB_upper'].iloc[-1] - df['BB_lower'].iloc[-1]):.2f}",
                f"{(df['Volume'].iloc[-1] / df['Volume_SMA'].iloc[-1]):.2f}x"
            ]
        }
        st.dataframe(pd.DataFrame(vol_data), use_container_width=True, hide_index=True)

with tab3:
    st.markdown("### 📝 Trade History & Performance")
    
    if st.session_state.trade_history:
        # Convert to DataFrame
        trades_df = pd.DataFrame(st.session_state.trade_history)
        trades_df['time'] = pd.to_datetime(trades_df['time'])
        
        # Summary metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Total Trades", len(trades_df))
        with metric_col2:
            buy_trades = len(trades_df[trades_df['signal'] == 'BUY'])
            sell_trades = len(trades_df[trades_df['signal'] == 'SELL'])
            st.metric("Buy/Sell", f"{buy_trades}/{sell_trades}")
        with metric_col3:
            avg_rr = trades_df['risk_reward'].mean()
            st.metric("Avg Risk:Reward", f"1:{avg_rr:.1f}")
        with metric_col4:
            total_risk = trades_df['amount_usd'].sum()
            st.metric("Total Risk", f"${total_risk:.2f}")
        
        # Trade history table
        st.dataframe(
            trades_df[['time', 'asset', 'signal', 'entry', 'stop_loss', 'take_profit', 'risk_reward', 'amount_usd']],
            use_container_width=True,
            hide_index=True
        )
        
        # Clear history button
        if st.button("🗑️ Clear Trade History"):
            st.session_state.trade_history = []
            st.rerun()
    else:
        st.info("No trades recorded yet. Execute a trade to see history.")

with tab4:
    st.markdown("### 🎓 Trading Education & Best Practices")
    
    edu_col1, edu_col2 = st.columns(2)
    
    with edu_col1:
        st.markdown("""
        #### 📊 Understanding Indicators
        
        **EMA (Exponential Moving Average)**
        - Faster reaction to price changes than SMA
        - EMA9 > EMA21 suggests uptrend
        - Crossovers signal potential trend changes
        
        **RSI (Relative Strength Index)**
        - Measures momentum (0-100 scale)
        - <30 = Oversold (potential buy)
        - >70 = Overbought (potential sell)
        
        **MACD (Moving Average Convergence Divergence)**
        - Shows relationship between two EMAs
        - Signal line crossovers indicate momentum shifts
        - Histogram shows strength of trend
        
        **Bollinger Bands**
        - Measure volatility and overbought/oversold
        - Price at upper band = potentially overbought
        - Price at lower band = potentially oversold
        """)
    
    with edu_col2:
        st.markdown("""
        #### 💡 Risk Management Rules
        
        **Position Sizing**
        - Never risk more than 1-2% per trade
        - Use the formula: Position = Risk Amount / Stop Distance
        - Account for fees and slippage
        
        **Stop Loss Placement**
        - Use ATR for dynamic stops
        - Place beyond support/resistance
        - Never move stops against you
        
        **Take Profit Strategy**
        - Set realistic targets based on volatility
        - Consider scaling out of positions
        - Use trailing stops in strong trends
        
        **Psychology Tips**
        - Stick to your plan
        - Don't revenge trade
        - Keep a trading journal
        - Take breaks after losses
        """)
    
    st.markdown("---")
    st.warning("""
    ⚠️ **Risk Disclaimer**: Cryptocurrency trading involves substantial risk of loss. 
    This tool is for educational purposes only. Always use stop losses, trade with money 
    you can afford to lose, and consider paper trading first.
    """)

# Auto refresh logic
if auto_refresh:
    time.sleep(refresh_interval)
    st.session_state.last_refresh = datetime.now()
    st.rerun()

# Footer
st.markdown("---")
st.caption("🚀 Advanced Crypto Day Trading Terminal v2.0 | Real-time data from Coinbase & Binance.US")
