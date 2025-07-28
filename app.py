import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from collections import deque
import ta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ========= Configuration ==========
st.set_page_config(
    page_title="CryptoEdge Pro - Small Account Trading Suite",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Professional crypto day trading suite for small accounts"
    }
)

# ========= Initialize Session State ==========
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'price_alerts' not in st.session_state:
    st.session_state.price_alerts = {}
if 'active_positions' not in st.session_state:
    st.session_state.active_positions = {}
if 'pnl_history' not in st.session_state:
    st.session_state.pnl_history = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'market_regime' not in st.session_state:
    st.session_state.market_regime = "Neutral"

# ========= Advanced Market Analysis ==========
class MarketAnalyzer:
    @staticmethod
    def calculate_market_structure(highs, lows):
        """Identify market structure (HH, HL, LL, LH)"""
        if len(highs) < 4:
            return "Insufficient Data"
        
        # Find swing points
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(highs)-2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append((i, highs[i]))
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append((i, lows[i]))
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Check for uptrend (HH, HL)
            if swing_highs[-1][1] > swing_highs[-2][1] and swing_lows[-1][1] > swing_lows[-2][1]:
                return "Uptrend (HH/HL)"
            # Check for downtrend (LH, LL)
            elif swing_highs[-1][1] < swing_highs[-2][1] and swing_lows[-1][1] < swing_lows[-2][1]:
                return "Downtrend (LH/LL)"
        
        return "Ranging"
    
    @staticmethod
    def calculate_volume_profile(prices, volumes, bins=20):
        """Calculate volume profile and identify POC, VAH, VAL"""
        price_range = np.linspace(min(prices), max(prices), bins)
        volume_profile = np.zeros(len(price_range)-1)
        
        for i in range(len(prices)):
            for j in range(len(price_range)-1):
                if price_range[j] <= prices[i] < price_range[j+1]:
                    volume_profile[j] += volumes[i]
                    break
        
        # Find Point of Control (POC)
        poc_idx = np.argmax(volume_profile)
        poc_price = (price_range[poc_idx] + price_range[poc_idx+1]) / 2
        
        # Calculate Value Area (70% of volume)
        total_volume = np.sum(volume_profile)
        value_area_volume = 0.7 * total_volume
        
        # Expand from POC to find value area
        lower_idx, upper_idx = poc_idx, poc_idx
        current_volume = volume_profile[poc_idx]
        
        while current_volume < value_area_volume:
            if lower_idx > 0 and upper_idx < len(volume_profile)-1:
                if volume_profile[lower_idx-1] > volume_profile[upper_idx+1]:
                    lower_idx -= 1
                    current_volume += volume_profile[lower_idx]
                else:
                    upper_idx += 1
                    current_volume += volume_profile[upper_idx]
            elif lower_idx > 0:
                lower_idx -= 1
                current_volume += volume_profile[lower_idx]
            elif upper_idx < len(volume_profile)-1:
                upper_idx += 1
                current_volume += volume_profile[upper_idx]
            else:
                break
        
        val = price_range[lower_idx]
        vah = price_range[upper_idx+1]
        
        return poc_price, vah, val, volume_profile, price_range

# ========= Advanced Indicators ==========
def calculate_market_cipher(df):
    """Market Cipher B style indicators"""
    # Money Flow
    mfi = ta.volume.MFI(df['high'], df['low'], df['close'], df['volume'], window=14)
    
    # Wave Trend
    ap = (df['high'] + df['low'] + df['close']) / 3
    esa = ta.trend.EMAIndicator(ap, window=10).ema_indicator()
    d = ta.trend.EMAIndicator(abs(ap - esa), window=10).ema_indicator()
    ci = (ap - esa) / (0.015 * d)
    wt1 = ta.trend.EMAIndicator(ci, window=21).ema_indicator()
    wt2 = ta.trend.SMAIndicator(wt1, window=4).sma_indicator()
    
    return {
        'mfi': mfi.iloc[-1] if len(mfi) > 0 else 0,
        'wt1': wt1.iloc[-1] if len(wt1) > 0 else 0,
        'wt2': wt2.iloc[-1] if len(wt2) > 0 else 0,
        'wt_cross': 'bullish' if wt1.iloc[-1] > wt2.iloc[-1] and wt1.iloc[-2] <= wt2.iloc[-2] else 'bearish' if wt1.iloc[-1] < wt2.iloc[-1] and wt1.iloc[-2] >= wt2.iloc[-2] else 'none'
    }

def calculate_smart_money_indicators(df):
    """Smart Money Concepts indicators"""
    # Order blocks detection
    order_blocks = []
    for i in range(3, len(df)-1):
        # Bullish order block
        if df['close'].iloc[i] > df['open'].iloc[i] and df['close'].iloc[i+1] < df['open'].iloc[i+1]:
            if df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i-2]:
                order_blocks.append({
                    'type': 'bullish',
                    'level': df['low'].iloc[i],
                    'strength': abs(df['close'].iloc[i] - df['open'].iloc[i])
                })
        
        # Bearish order block
        if df['close'].iloc[i] < df['open'].iloc[i] and df['close'].iloc[i+1] > df['open'].iloc[i+1]:
            if df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i-2]:
                order_blocks.append({
                    'type': 'bearish',
                    'level': df['high'].iloc[i],
                    'strength': abs(df['close'].iloc[i] - df['open'].iloc[i])
                })
    
    # Fair Value Gaps (FVG)
    fvgs = []
    for i in range(1, len(df)-1):
        # Bullish FVG
        if df['low'].iloc[i+1] > df['high'].iloc[i-1]:
            fvgs.append({
                'type': 'bullish',
                'top': df['low'].iloc[i+1],
                'bottom': df['high'].iloc[i-1],
                'midpoint': (df['low'].iloc[i+1] + df['high'].iloc[i-1]) / 2
            })
        
        # Bearish FVG
        if df['high'].iloc[i+1] < df['low'].iloc[i-1]:
            fvgs.append({
                'type': 'bearish',
                'top': df['low'].iloc[i-1],
                'bottom': df['high'].iloc[i+1],
                'midpoint': (df['low'].iloc[i-1] + df['high'].iloc[i+1]) / 2
            })
    
    return order_blocks, fvgs

# ========= Enhanced Multi-Exchange Data ==========
def get_advanced_market_data(symbol):
    """Get comprehensive market data with advanced indicators"""
    try:
        # Binance data (primary)
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=5m&limit=200"
        response = requests.get(url, timeout=5)
        klines = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                         'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
                                         'taker_buy_quote', 'ignore'])
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Basic indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['stoch_k'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        
        # VWAP
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # EMA
        df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Market Cipher
        cipher = calculate_market_cipher(df)
        
        # Smart Money
        order_blocks, fvgs = calculate_smart_money_indicators(df)
        
        # Market Structure
        market_structure = MarketAnalyzer.calculate_market_structure(
            df['high'].values[-50:], 
            df['low'].values[-50:]
        )
        
        # Volume Profile
        poc, vah, val, _, _ = MarketAnalyzer.calculate_volume_profile(
            df['close'].values[-100:],
            df['volume'].values[-100:]
        )
        
        # Order Flow
        buy_volume = df['taker_buy_base'].sum()
        sell_volume = df['volume'].sum() - buy_volume
        order_flow_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume) * 100
        
        # Liquidation levels (estimated)
        current_price = df['close'].iloc[-1]
        long_liq_5x = current_price * 0.84  # ~16% down
        long_liq_10x = current_price * 0.91  # ~9% down
        short_liq_5x = current_price * 1.19  # ~19% up
        short_liq_10x = current_price * 1.10  # ~10% up
        
        return {
            'success': True,
            'data_source': 'Binance',
            'price': current_price,
            'vwap': df['vwap'].iloc[-1],
            'rsi': df['rsi'].iloc[-1],
            'stoch_k': df['stoch_k'].iloc[-1],
            'macd': df['macd'].iloc[-1],
            'macd_signal': df['macd_signal'].iloc[-1],
            'bb_upper': df['bb_upper'].iloc[-1],
            'bb_middle': df['bb_middle'].iloc[-1],
            'bb_lower': df['bb_lower'].iloc[-1],
            'bb_width': df['bb_width'].iloc[-1],
            'atr': df['atr'].iloc[-1],
            'ema_9': df['ema_9'].iloc[-1],
            'ema_21': df['ema_21'].iloc[-1],
            'ema_50': df['ema_50'].iloc[-1],
            'volume_24h': df['volume'].sum(),
            'market_cipher': cipher,
            'order_blocks': order_blocks[-5:] if order_blocks else [],
            'fvgs': fvgs[-3:] if fvgs else [],
            'market_structure': market_structure,
            'poc': poc,
            'vah': vah,
            'val': val,
            'order_flow_imbalance': order_flow_imbalance,
            'liquidation_levels': {
                'long_5x': long_liq_5x,
                'long_10x': long_liq_10x,
                'short_5x': short_liq_5x,
                'short_10x': short_liq_10x
            },
            'df': df
        }
        
    except Exception as e:
        st.error(f"Data fetch error: {str(e)}")
        return {'success': False, 'error': str(e)}

# ========= Risk Management System ==========
class RiskManager:
    def __init__(self, account_balance, max_risk_percent, leverage):
        self.balance = account_balance
        self.max_risk_percent = max_risk_percent
        self.leverage = leverage
        
    def calculate_position_size(self, entry_price, stop_loss_price):
        """Calculate position size based on risk"""
        risk_amount = self.balance * (self.max_risk_percent / 100)
        price_difference = abs(entry_price - stop_loss_price)
        risk_percent = (price_difference / entry_price) * 100
        
        # Position size in base currency
        position_size = risk_amount / (risk_percent / 100)
        
        # Adjust for leverage
        actual_capital_required = position_size / self.leverage
        
        # Ensure we don't exceed account balance
        if actual_capital_required > self.balance * 0.95:  # Keep 5% buffer
            position_size = (self.balance * 0.95) * self.leverage
            actual_capital_required = self.balance * 0.95
        
        return {
            'position_size': position_size,
            'capital_required': actual_capital_required,
            'units': position_size / entry_price,
            'risk_amount': risk_amount,
            'risk_reward_1_1': entry_price + (entry_price - stop_loss_price),
            'risk_reward_2_1': entry_price + 2 * (entry_price - stop_loss_price),
            'risk_reward_3_1': entry_price + 3 * (entry_price - stop_loss_price)
        }

# ========= Trading Strategy Engine ==========
class TradingStrategy:
    def __init__(self, data):
        self.data = data
        
    def generate_signals(self):
        """Generate comprehensive trading signals"""
        signals = []
        strength = 0
        
        # Price action signals
        if self.data['price'] > self.data['vwap'] and self.data['price'] > self.data['ema_21']:
            signals.append("Price above VWAP & EMA21")
            strength += 1
            
        # RSI signals
        if 30 < self.data['rsi'] < 70:
            if self.data['rsi'] < 40:
                signals.append("RSI oversold - potential buy")
                strength += 1
            elif self.data['rsi'] > 60:
                signals.append("RSI overbought - potential sell")
                strength -= 1
                
        # MACD signals
        if self.data['macd'] > self.data['macd_signal']:
            signals.append("MACD bullish crossover")
            strength += 1
        else:
            signals.append("MACD bearish")
            strength -= 1
            
        # Bollinger Bands
        if self.data['price'] <= self.data['bb_lower']:
            signals.append("Price at lower BB - potential bounce")
            strength += 2
        elif self.data['price'] >= self.data['bb_upper']:
            signals.append("Price at upper BB - potential reversal")
            strength -= 2
            
        # Market Cipher
        if self.data['market_cipher']['wt_cross'] == 'bullish':
            signals.append("Market Cipher bullish cross")
            strength += 2
        elif self.data['market_cipher']['wt_cross'] == 'bearish':
            signals.append("Market Cipher bearish cross")
            strength -= 2
            
        # Order flow
        if self.data['order_flow_imbalance'] > 20:
            signals.append(f"Strong buy pressure ({self.data['order_flow_imbalance']:.1f}%)")
            strength += 1
        elif self.data['order_flow_imbalance'] < -20:
            signals.append(f"Strong sell pressure ({self.data['order_flow_imbalance']:.1f}%)")
            strength -= 1
            
        # Market structure
        if "Uptrend" in self.data['market_structure']:
            signals.append("Market structure: Uptrend")
            strength += 1
        elif "Downtrend" in self.data['market_structure']:
            signals.append("Market structure: Downtrend")
            strength -= 1
            
        # Determine overall signal
        if strength >= 3:
            overall_signal = "STRONG BUY"
            color = "green"
        elif strength >= 1:
            overall_signal = "BUY"
            color = "lightgreen"
        elif strength <= -3:
            overall_signal = "STRONG SELL"
            color = "red"
        elif strength <= -1:
            overall_signal = "SELL"
            color = "lightcoral"
        else:
            overall_signal = "NEUTRAL"
            color = "gray"
            
        return {
            'signals': signals,
            'strength': strength,
            'overall': overall_signal,
            'color': color
        }

# ========= Main UI ==========
st.title("🚀 CryptoEdge Pro - Small Account Trading Suite")
st.caption("Professional day trading tools optimized for accounts under $1000")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Account Settings
    st.subheader("💰 Account Settings")
    account_balance = st.number_input(
        "Account Balance (USD)", 
        min_value=10.0, 
        max_value=10000.0, 
        value=100.0, 
        step=10.0
    )
    
    max_risk_percent = st.slider(
        "Max Risk per Trade (%)", 
        min_value=0.5, 
        max_value=5.0, 
        value=1.0, 
        step=0.5
    )
    
    leverage = st.slider(
        "Leverage", 
        min_value=1, 
        max_value=20, 
        value=5
    )
    
    # Trading Preferences
    st.subheader("📊 Trading Preferences")
    
    symbols = st.multiselect(
        "Select Symbols",
        ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "MATICUSDT", "DOGEUSDT"],
        default=["BTCUSDT", "ETHUSDT"]
    )
    
    timeframe = st.selectbox(
        "Primary Timeframe",
        ["1m", "5m", "15m", "30m", "1h"],
        index=1
    )
    
    # Auto-refresh
    auto_refresh = st.checkbox("Auto Refresh", value=True)
    refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 15)
    
    # Manual refresh
    if st.button("🔄 Refresh Now", type="primary"):
        st.session_state.last_update = datetime.now()
        st.rerun()
    
    # Display last update
    if 'last_update' in st.session_state:
        st.caption(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")

# Initialize Risk Manager
risk_manager = RiskManager(account_balance, max_risk_percent, leverage)

# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", 
    "📈 Advanced Charts", 
    "🎯 Trade Planner",
    "📝 Trade Journal",
    "🔔 Alerts & Settings"
])

# Tab 1: Dashboard
with tab1:
    st.header("📊 Live Market Dashboard")
    
    for symbol in symbols:
        with st.container():
            st.subheader(f"🪙 {symbol}")
            
            # Get market data
            data = get_advanced_market_data(symbol)
            
            if data['success']:
                # Create columns for metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric(
                        "Price", 
                        f"${data['price']:.2f}",
                        delta=f"{((data['price'] - data['vwap']) / data['vwap'] * 100):.2f}%"
                    )
                
                with col2:
                    st.metric("VWAP", f"${data['vwap']:.2f}")
                
                with col3:
                    st.metric("RSI", f"{data['rsi']:.1f}")
                
                with col4:
                    st.metric("ATR", f"${data['atr']:.2f}")
                
                with col5:
                    st.metric("24h Volume", f"${data['volume_24h']/1e6:.1f}M")
                
                # Trading signals
                strategy = TradingStrategy(data)
                signals = strategy.generate_signals()
                
                # Signal display
                st.markdown(f"### Signal: <span style='color:{signals['color']}'>{signals['overall']}</span> (Strength: {signals['strength']}/7)", unsafe_allow_html=True)
                
                # Signal details
                with st.expander("📋 Signal Details", expanded=True):
                    for signal in signals['signals']:
                        st.write(f"• {signal}")
                
                # Key Levels
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Key Levels:**")
                    st.write(f"• POC: ${data['poc']:.2f}")
                    st.write(f"• VAH: ${data['vah']:.2f}")
                    st.write(f"• VAL: ${data['val']:.2f}")
                    st.write(f"• BB Upper: ${data['bb_upper']:.2f}")
                    st.write(f"• BB Lower: ${data['bb_lower']:.2f}")
                
                with col2:
                    st.markdown("**Liquidation Levels:**")
                    st.write(f"• Long 5x: ${data['liquidation_levels']['long_5x']:.2f}")
                    st.write(f"• Long 10x: ${data['liquidation_levels']['long_10x']:.2f}")
                    st.write(f"• Short 5x: ${data['liquidation_levels']['short_5x']:.2f}")
                    st.write(f"• Short 10x: ${data['liquidation_levels']['short_10x']:.2f}")
                
                # Order Flow
                st.markdown("**Order Flow:**")
                flow_color = "green" if data['order_flow_imbalance'] > 0 else "red"
                st.progress(
                    abs(data['order_flow_imbalance']) / 100,
                    text=f"Imbalance: {data['order_flow_imbalance']:.1f}% {'(Bullish)' if data['order_flow_imbalance'] > 0 else '(Bearish)'}"
                )
                
                # Market Structure
                st.info(f"Market Structure: {data['market_structure']}")
                
                st.divider()

# Tab 2: Advanced Charts
with tab2:
    st.header("📈 Advanced Chart Analysis")
    
    if symbols:
        selected_symbol = st.selectbox("Select Symbol for Charting", symbols)
        chart_data = get_advanced_market_data(selected_symbol)
        
        if chart_data['success']:
            # Create subplots
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.5, 0.2, 0.15, 0.15],
                subplot_titles=(
                    'Price Action & Indicators',
                    'Volume & Order Flow',
                    'RSI & Stochastic',
                    'MACD'
                )
            )
            
            df = chart_data['df']
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Add indicators
            fig.add_trace(
                go.Scatter(x=df.index, y=df['vwap'], name='VWAP', line=dict(color='yellow', width=2)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df.index, y=df['ema_9'], name='EMA 9', line=dict(color='blue', width=1)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df.index, y=df['ema_21'], name='EMA 21', line=dict(color='orange', width=1)),
                row=1, col=1
            )
            
            # Bollinger Bands
            fig.add_trace(
                go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper', line=dict(color='gray', width=1, dash='dash')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower', line=dict(color='gray', width=1, dash='dash')),
                row=1, col=1
            )
            
            # Volume
            colors = ['green' if close > open else 'red' for close, open in zip(df['close'], df['open'])]
            fig.add_trace(
                go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors),
                row=2, col=1
            )
            
            # RSI
            fig.add_trace(
                go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple')),
                row=3, col=1
            )
            
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            # MACD
            fig.add_trace(
                go.Scatter(x=df.index, y=df['macd'], name='MACD', line=dict(color='blue')),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df.index, y=df['macd_signal'], name='Signal', line=dict(color='red')),
                row=4, col=1
            )
            
            # MACD histogram
            colors = ['green' if val > 0 else 'red' for val in df['macd_diff']]
            fig.add_trace(
                go.Bar(x=df.index, y=df['macd_diff'], name='MACD Hist', marker_color=colors),
                row=4, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f"{selected_symbol} - Advanced Technical Analysis",
                xaxis_title="Time",
                yaxis_title="Price",
                template="plotly_dark",
                height=1000,
                showlegend=True
            )
            
            fig.update_xaxes(rangeslider_visible=False)
            
            st.plotly_chart(fig, use_container_width=True)

# Tab 3: Trade Planner
with tab3:
    st.header("🎯 Smart Trade Planner")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Position Calculator")
        
        calc_symbol = st.selectbox("Symbol", symbols, key="calc_symbol")
        
        if calc_symbol:
            market_data = get_advanced_market_data(calc_symbol)
            if market_data['success']:
               entry_price = st.number_input(
                   "Entry Price", 
                   value=float(market_data['price']), 
                   step=0.01,
                   key="entry_price"
               )
               
               stop_loss = st.number_input(
                   "Stop Loss", 
                   value=float(market_data['price'] * 0.98), 
                   step=0.01,
                   key="stop_loss"
               )
               
               # Calculate position
               position = risk_manager.calculate_position_size(entry_price, stop_loss)
               
               st.markdown("### Position Details")
               st.write(f"**Position Size:** ${position['position_size']:.2f}")
               st.write(f"**Units:** {position['units']:.6f} {calc_symbol.replace('USDT', '')}")
               st.write(f"**Capital Required:** ${position['capital_required']:.2f}")
               st.write(f"**Risk Amount:** ${position['risk_amount']:.2f}")
               
               st.markdown("### Take Profit Levels")
               st.write(f"**1:1 R/R:** ${position['risk_reward_1_1']:.2f}")
               st.write(f"**2:1 R/R:** ${position['risk_reward_2_1']:.2f}")
               st.write(f"**3:1 R/R:** ${position['risk_reward_3_1']:.2f}")
               
               # Add to positions button
               if st.button("📝 Add to Active Positions", type="primary"):
                   st.session_state.active_positions[calc_symbol] = {
                       'entry': entry_price,
                       'stop_loss': stop_loss,
                       'position_size': position['position_size'],
                       'units': position['units'],
                       'tp1': position['risk_reward_1_1'],
                       'tp2': position['risk_reward_2_1'],
                       'tp3': position['risk_reward_3_1'],
                       'timestamp': datetime.now()
                   }
                   st.success("Position added to active trades!")
   
   with col2:
       st.subheader("Active Positions")
       
       if st.session_state.active_positions:
           for symbol, pos in st.session_state.active_positions.items():
               with st.expander(f"{symbol} - Entry: ${pos['entry']:.2f}"):
                   current_data = get_advanced_market_data(symbol)
                   if current_data['success']:
                       current_price = current_data['price']
                       pnl = (current_price - pos['entry']) * pos['units']
                       pnl_percent = ((current_price - pos['entry']) / pos['entry']) * 100
                       
                       col_a, col_b = st.columns(2)
                       with col_a:
                           st.metric("Current Price", f"${current_price:.2f}")
                           st.metric("P&L", f"${pnl:.2f}", delta=f"{pnl_percent:.2f}%")
                       
                       with col_b:
                           st.write(f"Stop Loss: ${pos['stop_loss']:.2f}")
                           st.write(f"TP1: ${pos['tp1']:.2f}")
                           st.write(f"TP2: ${pos['tp2']:.2f}")
                           st.write(f"TP3: ${pos['tp3']:.2f}")
                       
                       if st.button(f"Close {symbol}", key=f"close_{symbol}"):
                           # Record trade
                           result = "WIN" if pnl > 0 else "LOSS"
                           st.session_state.trade_history.append({
                               'symbol': symbol,
                               'entry': pos['entry'],
                               'exit': current_price,
                               'pnl': pnl,
                               'pnl_percent': pnl_percent,
                               'result': result,
                               'timestamp': datetime.now()
                           })
                           del st.session_state.active_positions[symbol]
                           st.rerun()
       else:
           st.info("No active positions")

# Tab 4: Trade Journal
with tab4:
   st.header("📝 Trade Journal & Analytics")
   
   if st.session_state.trade_history:
       # Convert to DataFrame
       df_trades = pd.DataFrame(st.session_state.trade_history)
       
       # Summary metrics
       col1, col2, col3, col4 = st.columns(4)
       
       total_trades = len(df_trades)
       winning_trades = len(df_trades[df_trades['result'] == 'WIN'])
       losing_trades = len(df_trades[df_trades['result'] == 'LOSS'])
       win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
       
       total_pnl = df_trades['pnl'].sum()
       avg_win = df_trades[df_trades['result'] == 'WIN']['pnl'].mean() if winning_trades > 0 else 0
       avg_loss = abs(df_trades[df_trades['result'] == 'LOSS']['pnl'].mean()) if losing_trades > 0 else 0
       profit_factor = (avg_win * winning_trades) / (avg_loss * losing_trades) if losing_trades > 0 else 0
       
       with col1:
           st.metric("Total Trades", total_trades)
           st.metric("Win Rate", f"{win_rate:.1f}%")
       
       with col2:
           st.metric("Winning Trades", winning_trades)
           st.metric("Losing Trades", losing_trades)
       
       with col3:
           st.metric("Total P&L", f"${total_pnl:.2f}")
           st.metric("Avg Win", f"${avg_win:.2f}")
       
       with col4:
           st.metric("Avg Loss", f"${avg_loss:.2f}")
           st.metric("Profit Factor", f"{profit_factor:.2f}")
       
       # Trade history table
       st.subheader("Trade History")
       
       # Format display
       display_df = df_trades.copy()
       display_df['entry'] = display_df['entry'].apply(lambda x: f"${x:.2f}")
       display_df['exit'] = display_df['exit'].apply(lambda x: f"${x:.2f}")
       display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:.2f}")
       display_df['pnl_percent'] = display_df['pnl_percent'].apply(lambda x: f"{x:.2f}%")
       display_df['timestamp'] = display_df['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M'))
       
       st.dataframe(
           display_df[['timestamp', 'symbol', 'entry', 'exit', 'pnl', 'pnl_percent', 'result']],
           use_container_width=True
       )
       
       # Performance chart
       st.subheader("Cumulative P&L")
       
       df_trades['cumulative_pnl'] = df_trades['pnl'].cumsum()
       
       fig = go.Figure()
       fig.add_trace(go.Scatter(
           x=df_trades['timestamp'],
           y=df_trades['cumulative_pnl'],
           mode='lines+markers',
           name='Cumulative P&L',
           line=dict(color='green' if df_trades['cumulative_pnl'].iloc[-1] > 0 else 'red', width=2)
       ))
       
       fig.update_layout(
           title="Account Growth",
           xaxis_title="Time",
           yaxis_title="P&L ($)",
           template="plotly_dark",
           height=400
       )
       
       st.plotly_chart(fig, use_container_width=True)
       
       # Export button
       if st.button("📥 Export Trade History"):
           csv = df_trades.to_csv(index=False)
           st.download_button(
               label="Download CSV",
               data=csv,
               file_name=f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
               mime="text/csv"
           )
   else:
       st.info("No trades recorded yet")

# Tab 5: Alerts & Settings
with tab5:
   st.header("🔔 Alerts & Advanced Settings")
   
   col1, col2 = st.columns(2)
   
   with col1:
       st.subheader("Price Alerts")
       
       alert_symbol = st.selectbox("Symbol", symbols, key="alert_symbol")
       alert_type = st.selectbox("Alert Type", ["Above", "Below", "Cross VWAP", "RSI Level"])
       
       if alert_type in ["Above", "Below"]:
           alert_price = st.number_input("Price Level", value=0.0, step=0.01)
           
           if st.button("Add Price Alert"):
               if alert_symbol not in st.session_state.price_alerts:
                   st.session_state.price_alerts[alert_symbol] = []
               
               st.session_state.price_alerts[alert_symbol].append({
                   'type': alert_type,
                   'price': alert_price,
                   'created': datetime.now()
               })
               st.success("Alert added!")
       
       elif alert_type == "RSI Level":
           rsi_level = st.number_input("RSI Level", value=70.0, step=1.0)
           rsi_direction = st.selectbox("Direction", ["Above", "Below"])
           
           if st.button("Add RSI Alert"):
               if alert_symbol not in st.session_state.price_alerts:
                   st.session_state.price_alerts[alert_symbol] = []
               
               st.session_state.price_alerts[alert_symbol].append({
                   'type': 'RSI',
                   'level': rsi_level,
                   'direction': rsi_direction,
                   'created': datetime.now()
               })
               st.success("RSI alert added!")
       
       # Display active alerts
       st.subheader("Active Alerts")
       
       for symbol, alerts in st.session_state.price_alerts.items():
           if alerts:
               st.write(f"**{symbol}:**")
               for i, alert in enumerate(alerts):
                   if alert['type'] in ['Above', 'Below']:
                       st.write(f"• Price {alert['type']} ${alert['price']:.2f}")
                   elif alert['type'] == 'RSI':
                       st.write(f"• RSI {alert['direction']} {alert['level']}")
                   
                   if st.button(f"Remove", key=f"remove_alert_{symbol}_{i}"):
                       st.session_state.price_alerts[symbol].pop(i)
                       st.rerun()
   
   with col2:
       st.subheader("Advanced Settings")
       
       # Market regime detection
       st.markdown("### Market Regime Detection")
       regime_enabled = st.checkbox("Enable Market Regime Detection", value=True)
       
       if regime_enabled:
           st.info(f"Current Market Regime: {st.session_state.market_regime}")
       
       # Risk settings
       st.markdown("### Risk Management")
       
       max_daily_loss = st.number_input(
           "Max Daily Loss ($)",
           value=account_balance * 0.05,
           step=10.0
       )
       
       max_positions = st.number_input(
           "Max Concurrent Positions",
           min_value=1,
           max_value=10,
           value=3
       )
       
       # Trading rules
       st.markdown("### Trading Rules")
       
       use_trailing_stop = st.checkbox("Use Trailing Stop Loss", value=True)
       trailing_stop_percent = st.slider("Trailing Stop %", 0.5, 5.0, 1.5, 0.5)
       
       use_breakeven = st.checkbox("Move to Breakeven at 1:1", value=True)
       
       # News settings
       st.markdown("### News & Events")
       show_news = st.checkbox("Show Crypto News Feed", value=True)
       
       # Save settings
       if st.button("💾 Save Settings"):
           st.success("Settings saved!")

# Footer with key metrics
st.divider()
footer_col1, footer_col2, footer_col3, footer_col4 = st.columns(4)

with footer_col1:
   st.metric("Account Balance", f"${account_balance:.2f}")

with footer_col2:
   active_positions_value = sum([pos['position_size'] for pos in st.session_state.active_positions.values()])
   st.metric("Active Exposure", f"${active_positions_value:.2f}")

with footer_col3:
   available_balance = account_balance - (active_positions_value / leverage)
   st.metric("Available Balance", f"${available_balance:.2f}")

with footer_col4:
   if st.session_state.trade_history:
       today_pnl = sum([t['pnl'] for t in st.session_state.trade_history if t['timestamp'].date() == datetime.now().date()])
       st.metric("Today's P&L", f"${today_pnl:.2f}")
   else:
       st.metric("Today's P&L", "$0.00")

# Check alerts
for symbol in symbols:
   if symbol in st.session_state.price_alerts:
       data = get_advanced_market_data(symbol)
       if data['success']:
           for alert in st.session_state.price_alerts[symbol]:
               triggered = False
               
               if alert['type'] == 'Above' and data['price'] >= alert['price']:
                   triggered = True
                   message = f"🔔 {symbol} crossed above ${alert['price']:.2f}"
               elif alert['type'] == 'Below' and data['price'] <= alert['price']:
                   triggered = True
                   message = f"🔔 {symbol} crossed below ${alert['price']:.2f}"
               elif alert['type'] == 'RSI':
                   if alert['direction'] == 'Above' and data['rsi'] >= alert['level']:
                       triggered = True
                       message = f"🔔 {symbol} RSI crossed above {alert['level']}"
                   elif alert['direction'] == 'Below' and data['rsi'] <= alert['level']:
                       triggered = True
                       message = f"🔔 {symbol} RSI crossed below {alert['level']}"
               
               if triggered:
                   st.toast(message, icon='🚨')
                   st.session_state.alerts.append({
                       'message': message,
                       'timestamp': datetime.now()
                   })

# Auto-refresh logic
if auto_refresh:
   time_since_update = (datetime.now() - st.session_state.last_update).total_seconds()
   if time_since_update >= refresh_interval:
       st.session_state.last_update = datetime.now()
       st.rerun()

# Warning message
st.caption("⚠️ This tool is for educational purposes only. Cryptocurrency trading involves substantial risk. Never trade with money you cannot afford to lose.")
