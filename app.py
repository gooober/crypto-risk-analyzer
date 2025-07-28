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

# Page config
st.set_page_config(
    page_title="Intraday Crypto Trade Risk Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Intraday Crypto Trade Risk Analyzer")

# Initialize session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'price_alerts' not in st.session_state:
    st.session_state.price_alerts = {}
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = []
if 'order_flow_history' not in st.session_state:
    st.session_state.order_flow_history = {}

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    symbols = st.multiselect(
        "Select Cryptocurrencies",
        ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"],
        default=["BTCUSDT", "ETHUSDT"]
    )
    
    leverage = st.slider("Leverage", 1, 20, 5)
    refresh_rate = st.slider("Refresh Rate (seconds)", 5, 60, 10)
    
    if st.button("🔄 Refresh Now"):
        st.session_state.last_update = datetime.now()
        st.rerun()
    
    # Alert Settings
    st.header("🔔 Alert Settings")
    enable_alerts = st.checkbox("Enable Alerts", value=False)
    if enable_alerts:
        alert_prob_threshold = st.slider("Alert when probability >", 60, 90, 70)
    
    if st.button("🗑️ Clear Alerts"):
        st.session_state.alerts = []
    
    # Price Alerts Section
    st.header("🎯 Price Alerts")
    
    with st.expander("Set Price Alert", expanded=False):
        alert_symbol = st.selectbox("Symbol:", symbols if symbols else ["BTCUSDT"], key="alert_symbol")
        alert_type = st.radio("Alert when price goes:", ["Above", "Below"], horizontal=True, key="alert_type")
        alert_price = st.number_input(
            f"Alert price ({alert_type.lower()}):", 
            min_value=0.0, 
            value=50000.0 if "BTC" in alert_symbol else 3000.0,
            key="alert_price"
        )
        
        if st.button("➕ Add Alert", type="primary"):
            # Initialize alerts for symbol if needed
            if alert_symbol not in st.session_state.price_alerts:
                st.session_state.price_alerts[alert_symbol] = {}
            
            # Add the alert
            alert_id = f"{alert_symbol}_{alert_type.lower()}_{alert_price}_{datetime.now().timestamp()}"
            st.session_state.price_alerts[alert_symbol][alert_id] = {
                'type': alert_type.lower(),
                'price': alert_price,
                'created': datetime.now()
            }
            st.success(f"Alert set for {alert_symbol} {alert_type.lower()} ${alert_price:.2f}")
    
    # Show active alerts
    if any(st.session_state.price_alerts.values()):
        st.write("**Active Price Alerts:**")
        for symbol, alerts in st.session_state.price_alerts.items():
            for alert_id, alert in alerts.items():
                st.write(f"• {symbol} {alert['type']} ${alert['price']:.2f}")
    
    # Signal History
    if st.button("📜 View Signal History"):
        if st.session_state.signal_history:
            st.write(f"Last {min(5, len(st.session_state.signal_history))} signals:")
            for signal in st.session_state.signal_history[-5:]:
                st.write(f"• {signal['timestamp'].strftime('%H:%M')} - {signal['symbol']} {signal['signal']}")

# VWAP Calculation Function
def calculate_vwap(klines_data):
    """
    Calculate VWAP (Volume Weighted Average Price) from kline data
    Returns: current VWAP, VWAP bands (1 & 2 std dev)
    """
    if not klines_data or len(klines_data) < 2:
        return None, None, None
    
    cumulative_tpv = 0  # Total Price * Volume
    cumulative_volume = 0
    vwap_values = []
    
    for kline in klines_data:
        # Extract OHLC and volume
        high = float(kline[2])
        low = float(kline[3])
        close = float(kline[4])
        volume = float(kline[5])
        
        # Typical price (HL2 or HLC3)
        typical_price = (high + low + close) / 3
        
        # Accumulate
        cumulative_tpv += typical_price * volume
        cumulative_volume += volume
        
        # Calculate VWAP
        if cumulative_volume > 0:
            vwap = cumulative_tpv / cumulative_volume
            vwap_values.append(vwap)
    
    if not vwap_values:
        return None, None, None
    
    current_vwap = vwap_values[-1]
    
    # Calculate standard deviation for bands
    if len(vwap_values) >= 20:
        recent_vwaps = vwap_values[-20:]
        std_dev = np.std(recent_vwaps)
        upper_band = current_vwap + (2 * std_dev)
        lower_band = current_vwap - (2 * std_dev)
    else:
        upper_band = current_vwap * 1.002  # 0.2% bands as fallback
        lower_band = current_vwap * 0.998
    
    return current_vwap, upper_band, lower_band

# Order Flow Imbalance Calculation
def calculate_order_flow_imbalance(trades_data, window=100):
    """
    Calculate Order Flow Imbalance from recent trades
    Returns: imbalance ratio, aggressive buy volume, aggressive sell volume
    """
    if not trades_data:
        return 0, 0, 0
    
    aggressive_buys = 0
    aggressive_sells = 0
    
    for trade in trades_data[-window:]:  # Last 100 trades
        price = float(trade['price'])
        qty = float(trade['qty'])
        is_buyer_maker = trade['isBuyerMaker']
        
        # If buyer is maker, then seller was aggressive (market sell)
        # If buyer is taker, then buyer was aggressive (market buy)
        if is_buyer_maker:
            aggressive_sells += qty * price
        else:
            aggressive_buys += qty * price
    
    total_volume = aggressive_buys + aggressive_sells
    
    if total_volume > 0:
        # Imbalance ratio: -100 (all sells) to +100 (all buys)
        imbalance = ((aggressive_buys - aggressive_sells) / total_volume) * 100
    else:
        imbalance = 0
    
    return imbalance, aggressive_buys, aggressive_sells

# Enhanced data fetching
@st.cache_data(ttl=5)
def get_enhanced_data(symbol):
    """Fetch enhanced data with technical indicators, VWAP, and order flow"""
    
    # Try multiple sources in order of preference
    price_data = None
    attempts = []
    
    # Method 1: Binance Spot API (most reliable)
    try:
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
        response = requests.get(url, timeout=3)
        
        if response.status_code == 200:
            stats = response.json()
            
            # Get additional kline data for better indicators
            kline_url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=100"
            kline_response = requests.get(kline_url, timeout=3)
            
            # Get recent trades for order flow
            trades_url = f"https://api.binance.com/api/v3/trades?symbol={symbol}&limit=500"
            trades_response = requests.get(trades_url, timeout=3)
            
            rsi = 50
            macd = 0
            vwap = float(stats['lastPrice'])
            vwap_upper = vwap * 1.002
            vwap_lower = vwap * 0.998
            order_flow_imbalance = 0
            aggressive_buy_vol = 0
            aggressive_sell_vol = 0
            
            if kline_response.status_code == 200:
                klines = kline_response.json()
                if len(klines) > 14:
                    # Calculate proper RSI
                    closes = [float(k[4]) for k in klines]
                    
                    # RSI calculation
                    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
                    gains = [d if d > 0 else 0 for d in deltas]
                    losses = [-d if d < 0 else 0 for d in deltas]
                    
                    avg_gain = sum(gains[-14:]) / 14
                    avg_loss = sum(losses[-14:]) / 14
                    
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                    
                    # Simple MACD
                    if len(closes) >= 26:
                        ema_12 = sum(closes[-12:]) / 12
                        ema_26 = sum(closes[-26:]) / 26
                        macd = ((ema_12 - ema_26) / ema_26) * 100  # As percentage
                    
                    # Calculate VWAP
                    vwap_calc, vwap_upper_calc, vwap_lower_calc = calculate_vwap(klines)
                    if vwap_calc:
                        vwap = vwap_calc
                        vwap_upper = vwap_upper_calc
                        vwap_lower = vwap_lower_calc
            
            # Calculate Order Flow Imbalance
            if trades_response.status_code == 200:
                trades = trades_response.json()
                order_flow_imbalance, aggressive_buy_vol, aggressive_sell_vol = calculate_order_flow_imbalance(trades)
                
                # Store order flow history
                if symbol not in st.session_state.order_flow_history:
                    st.session_state.order_flow_history[symbol] = deque(maxlen=50)
                
                st.session_state.order_flow_history[symbol].append({
                    'timestamp': datetime.now(),
                    'imbalance': order_flow_imbalance,
                    'buy_vol': aggressive_buy_vol,
                    'sell_vol': aggressive_sell_vol
                })
            
            price_data = {
                "current_price": float(stats['lastPrice']),
                "price_change_24h": float(stats['priceChangePercent']),
                "volume": float(stats['volume']),
                "high_24h": float(stats['highPrice']),
                "low_24h": float(stats['lowPrice']),
                "rsi": round(rsi, 2),
                "macd": round(macd, 2),
                "vwap": vwap,
                "vwap_upper": vwap_upper,
                "vwap_lower": vwap_lower,
                "price_vs_vwap": ((float(stats['lastPrice']) - vwap) / vwap * 100) if vwap else 0,
                "order_flow_imbalance": round(order_flow_imbalance, 2),
                "aggressive_buy_volume": aggressive_buy_vol,
                "aggressive_sell_volume": aggressive_sell_vol,
                "data_source": "Binance Spot",
                "last_updated": datetime.now().strftime("%H:%M:%S"),
                "bid": float(stats.get('bidPrice', stats['lastPrice'])),
                "ask": float(stats.get('askPrice', stats['lastPrice'])),
                "spread": round((float(stats.get('askPrice', stats['lastPrice'])) - float(stats.get('bidPrice', stats['lastPrice']))) / float(stats['lastPrice']) * 100, 4)
            }
            attempts.append("Binance Spot: Success")
            return price_data
        else:
            attempts.append(f"Binance Spot: HTTP {response.status_code}")
    except Exception as e:
        attempts.append(f"Binance Spot: {str(e)[:30]}")
    
    # Fallback demo data as last resort
    return {
        "current_price": 50000.0 if "BTC" in symbol else 3000.0 if "ETH" in symbol else 100.0,
        "price_change_24h": np.random.uniform(-5, 5),
        "volume": np.random.uniform(500000, 2000000),
        "high_24h": 51000 if "BTC" in symbol else 3100 if "ETH" in symbol else 105,
        "low_24h": 49000 if "BTC" in symbol else 2900 if "ETH" in symbol else 95,
        "rsi": np.random.uniform(30, 70),
        "macd": np.random.uniform(-5, 5),
        "vwap": 50000.0 if "BTC" in symbol else 3000.0,
        "vwap_upper": 50500.0 if "BTC" in symbol else 3030.0,
        "vwap_lower": 49500.0 if "BTC" in symbol else 2970.0,
        "price_vs_vwap": 0,
        "order_flow_imbalance": np.random.uniform(-50, 50),
        "aggressive_buy_volume": np.random.uniform(100000, 500000),
        "aggressive_sell_volume": np.random.uniform(100000, 500000),
        "data_source": f"Demo Mode ({len(attempts)} sources failed)",
        "last_updated": datetime.now().strftime("%H:%M:%S"),
        "bid": 0,
        "ask": 0,
        "spread": 0
    }

# Calculate trading probabilities
def calculate_probabilities(data, leverage):
    """Calculate trading probabilities based on indicators"""
    
    base_prob = 50
    
    # Price momentum
    change = data['price_change_24h']
    if change > 3:
        base_prob += 15
    elif change > 1:
        base_prob += 8
    elif change < -3:
        base_prob -= 15
    elif change < -1:
        base_prob -= 8
    
    # RSI signal
    rsi = data['rsi']
    if rsi > 70:
        base_prob -= 10
    elif rsi < 30:
        base_prob += 10
    
    # MACD signal
    if data['macd'] > 0:
        base_prob += 5
    elif data['macd'] < 0:
        base_prob -= 5
    
    # VWAP signal
    price_vs_vwap = data.get('price_vs_vwap', 0)
    if price_vs_vwap > 0.5:
        base_prob += 5
    elif price_vs_vwap < -0.5:
        base_prob -= 5
    
    # Order Flow signal
    order_flow = data.get('order_flow_imbalance', 0)
    if order_flow > 30:
        base_prob += 10
    elif order_flow < -30:
        base_prob -= 10
    
    # Calculate final probabilities
    long_prob = max(10, min(90, base_prob))
    short_prob = max(10, min(90, 100 - long_prob))
    
    # Risk calculation
    volatility = abs(change)
    risk_score = min(100, volatility * leverage * 3)
    
    # Recommendation
    if long_prob > 65:
        recommendation = "STRONG LONG 🚀"
    elif long_prob > 55:
        recommendation = "LONG 📈"
    elif short_prob > 65:
        recommendation = "STRONG SHORT 📉"
    elif short_prob > 55:
        recommendation = "SHORT 📉"
    else:
        recommendation = "NEUTRAL ⚖️"
    
    return {
        "long_prob": long_prob,
        "short_prob": short_prob,
        "risk_score": risk_score,
        "recommendation": recommendation
    }

# Generate trading signals for different strategies
def generate_trading_signals(data, strategy_type="perp"):
    """Generate specific buy/sell signals for different trading strategies"""
    
    signals = {
        "perp": {},
        "day_trading": {},
        "spot": {}
    }
    
    price = data['current_price']
    change_24h = data['price_change_24h']
    rsi = data['rsi']
    macd = data['macd']
    volume = data['volume']
    vwap = data.get('vwap', price)
    price_vs_vwap = data.get('price_vs_vwap', 0)
    order_flow = data.get('order_flow_imbalance', 0)
    
    # Perpetual Trading Signals with VWAP and Order Flow
    if strategy_type in ["perp", "all"]:
        perp_signal = "WAIT"
        perp_action = None
        
        # VWAP mean reversion with order flow confirmation
        if price_vs_vwap < -0.5 and rsi < 35 and order_flow > 20:
            perp_signal = "BUY"
            perp_action = {
                "entry": price,
                "stop_loss": data.get('vwap_lower', price * 0.995),
                "take_profit": vwap,
                "leverage": min(5, leverage),
                "reason": "VWAP bounce + RSI oversold + Bullish order flow"
            }
        elif price_vs_vwap > 0.5 and rsi > 65 and order_flow < -20:
            perp_signal = "SELL"
            perp_action = {
                "entry": price,
                "stop_loss": data.get('vwap_upper', price * 1.005),
                "take_profit": vwap,
                "leverage": min(5, leverage),
                "reason": "VWAP rejection + RSI overbought + Bearish order flow"
            }
        # Order flow divergence setup
        elif order_flow > 50 and price < vwap:
            perp_signal = "BUY"
            perp_action = {
                "entry": price,
                "stop_loss": price * 0.99,
                "take_profit": vwap * 1.01,
                "leverage": min(3, leverage),
                "reason": "Strong buy pressure below VWAP"
            }
        elif order_flow < -50 and price > vwap:
            perp_signal = "SELL"
            perp_action = {
                "entry": price,
                "stop_loss": price * 1.01,
                "take_profit": vwap * 0.99,
                "leverage": min(3, leverage),
                "reason": "Strong sell pressure above VWAP"
            }
        
        signals["perp"] = {
            "signal": perp_signal,
            "action": perp_action
        }
    
    # Day Trading Signals
    if strategy_type in ["day_trading", "all"]:
        day_signal = "WAIT"
        day_action = None
        
        # VWAP breakout with volume
        if price > vwap and price_vs_vwap < 0.3 and order_flow > 30:
            day_signal = "BUY"
            day_action = {
                "entry": price,
                "stop_loss": vwap,
                "take_profit": price * 1.015,
                "size": "50%",
                "reason": "VWAP breakout with strong order flow"
            }
        # VWAP breakdown
        elif price < vwap and price_vs_vwap > -0.3 and order_flow < -30:
            day_signal = "SELL"
            day_action = {
                "entry": price,
                "stop_loss": vwap,
                "take_profit": price * 0.985,
                "size": "50%",
                "reason": "VWAP breakdown with bearish flow"
            }
        
        signals["day_trading"] = {
            "signal": day_signal,
            "action": day_action
        }
    
    # Spot Trading Signals
    if strategy_type in ["spot", "all"]:
        spot_signal = "WAIT"
        spot_action = None
        
        # Deep value buy with order flow confirmation
        if rsi < 30 and change_24h < -10 and order_flow > 0:
            spot_signal = "STRONG BUY"
            spot_action = {
                "entry": price,
                "target_1": price * 1.10,
                "target_2": price * 1.20,
                "size": "10-20%",
                "reason": "Deep value + positive order flow"
            }
        
        signals["spot"] = {
            "signal": spot_signal,
            "action": spot_action
        }
    
    return signals

# Check alerts
def check_alerts(symbol, analysis):
    """Check if alerts should be triggered"""
    if not enable_alerts:
        return
    
    max_prob = max(analysis['long_prob'], analysis['short_prob'])
    if max_prob >= alert_prob_threshold:
        direction = "Long" if analysis['long_prob'] > analysis['short_prob'] else "Short"
        alert_msg = f"🎯 {symbol}: Strong {direction} signal! ({max_prob}%)"
        
        # Avoid duplicate alerts
        if not any(a['message'] == alert_msg for a in st.session_state.alerts[-5:]):
            st.session_state.alerts.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'symbol': symbol,
                'message': alert_msg
            })

# Check price alerts
def check_price_alerts(symbol, current_price):
    """Check if any price alerts should trigger"""
    if symbol not in st.session_state.price_alerts:
        return
    
    alerts_to_remove = []
    
    for alert_id, alert in st.session_state.price_alerts[symbol].items():
        triggered = False
        
        if alert['type'] == 'above' and current_price >= alert['price']:
            triggered = True
            alert_msg = f"🔔 {symbol} hit ${alert['price']:.2f} (Above alert)"
        elif alert['type'] == 'below' and current_price <= alert['price']:
            triggered = True
            alert_msg = f"🔔 {symbol} hit ${alert['price']:.2f} (Below alert)"
        
        if triggered:
            st.session_state.alerts.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'symbol': symbol,
                'message': alert_msg,
                'type': 'price_alert'
            })
            alerts_to_remove.append(alert_id)
            
            # Show notification
            st.balloons()
    
    # Remove triggered alerts
    for alert_id in alerts_to_remove:
        del st.session_state.price_alerts[symbol][alert_id]

# Save signal to history
def save_signal_to_history(symbol, signal_type, signal_data):
    """Save trading signals to history for tracking"""
    if signal_data['signal'] != "WAIT":
        st.session_state.signal_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'type': signal_type,
            'signal': signal_data['signal'],
            'action': signal_data['action'],
            'price': signal_data['action']['entry'] if signal_data['action'] else None
        })

# Main interface with tabs
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Charts", "📋 Portfolio"])

with tab1:
    st.header("📊 Trading Dashboard")
    
    # Connection status
    col_status1, col_status2, col_status3 = st.columns([2, 1, 1])
    with col_status1:
        selected_strategy = st.radio(
            "Trading Strategy:",
            ["Perpetual (Futures)", "Day Trading", "Spot Trading", "All Strategies"],
            horizontal=True
        )
    with col_status2:
        # Check connection by testing one symbol
        if symbols:
            test_data = get_enhanced_data(symbols[0])
            if "Demo Mode" not in test_data['data_source']:
                st.metric("Status", "🟢 Live", delta="Connected")
            else:
                st.metric("Status", "🟡 Demo", delta="Check connection")
    with col_status3:
        st.metric("Refresh", f"{int((datetime.now() - st.session_state.last_update).total_seconds())}s ago", delta=None)
    
    # Display alerts
    if st.session_state.alerts:
        with st.expander("🔔 Recent Alerts", expanded=True):
            for alert in st.session_state.alerts[-5:]:
                st.info(f"{alert['time']} - {alert['message']}")
    
    # Display symbols
    if symbols:
        st.caption(f"Last updated: {st.session_state.last_update.strftime('%H:%M:%S')} | Auto-refresh ON")
        
        cols = st.columns(min(len(symbols), 3))
        
        for idx, symbol in enumerate(symbols):
            with cols[idx % len(cols)]:
                data = get_enhanced_data(symbol)
                analysis = calculate_probabilities(data, leverage)
                check_alerts(symbol, analysis)
                
                # Check price alerts
                check_price_alerts(symbol, data['current_price'])
                
                # Generate signals based on selected strategy
                strategy_map = {
                    "Perpetual (Futures)": "perp",
                    "Day Trading": "day_trading",
                    "Spot Trading": "spot",
                    "All Strategies": "all"
                }
                signals = generate_trading_signals(data, strategy_map[selected_strategy])
                
                # Save signals to history
                if selected_strategy != "All Strategies":
                    save_signal_to_history(symbol, strategy_map[selected_strategy], signals[strategy_map[selected_strategy]])
                
                st.subheader(f"🪙 {symbol}")
                
                # Show data source and spread
                source_col1, source_col2 = st.columns(2)
                with source_col1:
                    source_emoji = "🟢" if "Live" in data['data_source'] else "🟡"
                    st.caption(f"{source_emoji} {data['data_source']}")
                with source_col2:
                    if 'spread' in data and data['spread'] > 0:
                        st.caption(f"Spread: {data['spread']:.3f}%")
                
                # Price metrics with bid/ask
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Price",
                        f"${data['current_price']:,.2f}",
                        delta=f"{data['price_change_24h']:+.2f}%"
                    )
                    if 'bid' in data and data['bid'] > 0:
                        st.caption(f"Bid: ${data['bid']:,.2f}")
                with col2:
                    st.metric(
                        "Volume",
                        f"${data['volume']/1e6:.1f}M"
                    )
                    if 'ask' in data and data['ask'] > 0:
                        st.caption(f"Ask: ${data['ask']:,.2f}")
                
                # Indicators
                st.write("**📊 Indicators**")
                ind_col1, ind_col2 = st.columns(2)
                with ind_col1:
                    rsi_emoji = "🔴" if data['rsi'] > 70 else "🟢" if data['rsi'] < 30 else "🟡"
                    st.write(f"{rsi_emoji} RSI: {data['rsi']:.1f}")
                with ind_col2:
                    macd_emoji = "🟢" if data['macd'] > 0 else "🔴"
                    st.write(f"{macd_emoji} MACD: {data['macd']:.2f}")
                
                # VWAP Analysis
                if 'vwap' in data and data['vwap']:
                    st.write("**📊 VWAP Analysis**")
                    vwap_col1, vwap_col2 = st.columns(2)
                    with vwap_col1:
                        vwap_position = "Above" if data['current_price'] > data['vwap'] else "Below"
                        vwap_color = "🟢" if vwap_position == "Above" else "🔴"
                        st.write(f"{vwap_color} Price {vwap_position} VWAP")
                        st.caption(f"VWAP: ${data['vwap']:.2f}")
                    with vwap_col2:
                        distance_pct = data.get('price_vs_vwap', 0)
                        st.write(f"Distance: {distance_pct:+.2f}%")
                        if abs(distance_pct) > 0.5:
                            st.caption("⚠️ Extended from VWAP")
                
                # Order Flow Analysis
                st.write("**🌊 Order Flow Analysis**")
                flow_col1, flow_col2 = st.columns(2)
                with flow_col1:
                    imbalance = data.get('order_flow_imbalance', 0)
                    flow_emoji = "🟢" if imbalance > 20 else "🔴" if imbalance < -20 else "🟡"
                    st.write(f"{flow_emoji} Imbalance: {imbalance:+.1f}%")
                    
                    # Show buy/sell pressure
                    if imbalance > 30:
                        st.caption("💪 Strong buy pressure")
                    elif imbalance < -30:
                        st.caption("💪 Strong sell pressure")
                with flow_col2:
                    buy_vol = data.get('aggressive_buy_volume', 0)
                    sell_vol = data.get('aggressive_sell_volume', 0)
                    total_vol = buy_vol + sell_vol
                    if total_vol > 0:
                        buy_pct = (buy_vol / total_vol) * 100
                        st.write(f"Buy/Sell: {buy_pct:.0f}/{100-buy_pct:.0f}%")
                        st.caption(f"${total_vol/1e6:.1f}M traded")
                
                # Show specific signals based on strategy
                if selected_strategy == "All Strategies":
                    # Show all signals
                    for strat_name, strat_data in signals.items():
                        if strat_data['signal'] != "WAIT":
                            st.success(f"**{strat_name.upper()} Signal: {strat_data['signal']}**")
                            if strat_data['action']:
                                with st.expander(f"📋 {strat_name.title()} Details", expanded=True):
                                    action = strat_data['action']
                                    st.write(f"**Reason:** {action['reason']}")
                                    st.write(f"**Entry:** ${action.get('entry', price):.2f}")
                                    if 'stop_loss' in action:
                                        st.write(f"**Stop Loss:** ${action['stop_loss']:.2f}")
                                    if 'take_profit' in action:
                                        st.write(f"**Take Profit:** ${action['take_profit']:.2f}")
                                    if 'leverage' in action:
                                        st.write(f"**Suggested Leverage:** {action['leverage']}x")
                                    if 'size' in action:
                                        st.write(f"**Position Size:** {action['size']}")
                else:
                    # Show selected strategy signal
                    strat_key = strategy_map[selected_strategy]
                    signal_data = signals[strat_key]
                    
                    if signal_data['signal'] != "WAIT":
                        st.success(f"**🎯 {signal_data['signal']} Signal Active!**")
                        if signal_data['action']:
                            action = signal_data['action']
                            st.write(f"**Reason:** {action['reason']}")
                            
                            # Display action details in columns
                            act_col1, act_col2 = st.columns(2)
                            with act_col1:
                                st.write(f"**Entry:** ${action.get('entry', data['current_price']):.2f}")
                                if 'stop_loss' in action:
                                    sl_pct = ((action['stop_loss'] / action['entry']) - 1) * 100
                                    st.write(f"**Stop Loss:** ${action['stop_loss']:.2f} ({sl_pct:+.1f}%)")
                            with act_col2:
                                if 'take_profit' in action:
                                    tp_pct = ((action['take_profit'] / action['entry']) - 1) * 100
                                    st.write(f"**Take Profit:** ${action['take_profit']:.2f} ({tp_pct:+.1f}%)")
                                if 'leverage' in action:
                                    st.write(f"**Leverage:** {action['leverage']}x")
                                elif 'size' in action:
                                    st.write(f"**Size:** {action['size']}")
                    else:
                        st.info("⏳ Waiting for signal...")
                
                # Traditional probability display (collapsed by default)
                with st.expander("📊 Probability Analysis"):
                    # Long probability
                    st.write(f"📈 Long: {analysis['long_prob']}%")
                    st.progress(analysis['long_prob'] / 100)
                    
                    # Short probability
                    st.write(f"📉 Short: {analysis['short_prob']}%")
                    st.progress(analysis['short_prob'] / 100)
                    
                    # Risk level
                    risk_emoji = "🟢" if analysis['risk_score'] < 30 else "🟡" if analysis['risk_score'] < 60 else "🔴"
                    st.write(f"**⚠️ Risk**: {risk_emoji} {analysis['risk_score']:.0f}%")
                
                st.divider()

with tab2:
    st.header("📈 Price Charts")
    
    if symbols:
        selected_symbol = st.selectbox("Select symbol for chart:", symbols)
        data = get_enhanced_data(selected_symbol)
        
        # Create subplots with order flow
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=('Price & VWAP', 'Order Flow Imbalance', 'Volume')
        )
        
        # Generate sample price data with VWAP
        hours = 24
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(hours, 0, -1)]
        base_price = data['current_price']
        
        # Simple price simulation
        prices = []
        vwap_values = []
        for i in range(hours):
            variation = np.random.uniform(-0.5, 0.5) / 100
            price = base_price * (1 + variation)
            prices.append(price)
            # Simulate VWAP
            vwap_values.append(np.mean(prices))
            base_price = price
        
        # Make sure last price matches current
        prices[-1] = data['current_price']
        vwap_values[-1] = data['vwap']
        
        # Price trace
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=prices,
            mode='lines',
            name='Price',
            line=dict(color='cyan', width=2)
        ), row=1, col=1)
        
        # VWAP trace
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=vwap_values,
            mode='lines',
            name='VWAP',
            line=dict(color='yellow', width=2, dash='dash')
        ), row=1, col=1)
        
        # VWAP bands
        upper_band = [v * 1.002 for v in vwap_values]
        lower_band = [v * 0.998 for v in vwap_values]
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=upper_band,
            mode='lines',
            name='VWAP Upper',
            line=dict(color='red', width=1, dash='dot'),
            showlegend=False
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=lower_band,
            mode='lines',
            name='VWAP Lower',
            line=dict(color='green', width=1, dash='dot'),
            showlegend=False
        ), row=1, col=1)
        
        # Order Flow Imbalance
        if selected_symbol in st.session_state.order_flow_history:
            flow_history = list(st.session_state.order_flow_history[selected_symbol])
            if flow_history:
                flow_times = [f['timestamp'] for f in flow_history]
                flow_values = [f['imbalance'] for f in flow_history]
                
                fig.add_trace(go.Bar(
                    x=flow_times,
                    y=flow_values,
                    name='Order Flow',
                    marker_color=['green' if v > 0 else 'red' for v in flow_values]
                ), row=2, col=1)
        
        # Volume bars
        volumes = [np.random.uniform(0.8, 1.2) * data['volume'] / 24 for _ in range(hours)]
        fig.add_trace(go.Bar(
            x=timestamps,
            y=volumes,
            name='Volume',
            marker_color='lightblue'
        ), row=3, col=1)
        
        fig.update_layout(
            title=f"{selected_symbol} Analysis Dashboard",
            xaxis_title="Time",
            yaxis_title="Price ($)",
            template="plotly_dark",
            height=700,
            showlegend=True
        )
        
        # Add zero line to order flow
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display current stats with VWAP and Order Flow
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Current", f"${data['current_price']:,.2f}")
        with col2:
            st.metric("VWAP", f"${data['vwap']:,.2f}")
        with col3:
            st.metric("24h High", f"${data['high_24h']:,.2f}")
        with col4:
            st.metric("24h Low", f"${data['low_24h']:,.2f}")
        with col5:
            st.metric("Order Flow", f"{data['order_flow_imbalance']:+.1f}%")
        with col6:
            st.metric("24h Change", f"{data['price_change_24h']:+.2f}%")
        
        # Order Flow Statistics
        if selected_symbol in st.session_state.order_flow_history and len(st.session_state.order_flow_history[selected_symbol]) > 0:
            st.subheader("🌊 Order Flow Statistics")
            
            flow_data = list(st.session_state.order_flow_history[selected_symbol])
            recent_flows = [f['imbalance'] for f in flow_data[-20:]]  # Last 20 readings
            
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            with stat_col1:
                avg_flow = np.mean(recent_flows)
                st.metric("Avg Flow (20)", f"{avg_flow:+.1f}%")
            with stat_col2:
                max_flow = max(recent_flows)
                st.metric("Max Buy Flow", f"{max_flow:+.1f}%")
            with stat_col3:
                min_flow = min(recent_flows)
                st.metric("Max Sell Flow", f"{min_flow:+.1f}%")
            with stat_col4:
                flow_trend = "Bullish" if avg_flow > 10 else "Bearish" if avg_flow < -10 else "Neutral"
                st.metric("Flow Trend", flow_trend)

with tab3:
    st.header("📋 Portfolio Tracker")
    
    # Simple trade input
    with st.expander("➕ Add Trade", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            trade_symbol = st.selectbox("Symbol:", symbols if symbols else ["BTCUSDT"])
        with col2:
            trade_entry = st.number_input("Entry Price:", min_value=0.0, value=0.0)
        with col3:
            trade_direction = st.radio("Direction:", ["Long", "Short"])
        with col4:
            trade_size = st.number_input("Size (USDT):", min_value=0.0, value=100.0)
        
        if st.button("Add Trade"):
            st.session_state.trade_history.append({
                'symbol': trade_symbol,
                'entry_price': trade_entry,
                'direction': trade_direction,
                'size': trade_size,
                'timestamp': datetime.now()
            })
            st.success("Trade added!")
    
    # Display trades with VWAP analysis
    if st.session_state.trade_history:
        st.subheader("Active Trades")
        
        trades_data = []
        total_pnl = 0
        
        for trade in st.session_state.trade_history:
            current_data = get_enhanced_data(trade['symbol'])
            current_price = current_data['current_price']
            vwap = current_data['vwap']
            
            if trade['direction'] == 'Long':
                pnl = (current_price - trade['entry_price']) / trade['entry_price'] * 100
            else:
                pnl = (trade['entry_price'] - current_price) / trade['entry_price'] * 100
            
            pnl_usd = pnl / 100 * trade['size']
            total_pnl += pnl_usd
            
            # VWAP analysis for trade
            entry_vs_vwap = ((trade['entry_price'] - vwap) / vwap * 100)
            current_vs_vwap = current_data['price_vs_vwap']
            
            trades_data.append({
                'Symbol': trade['symbol'],
                'Direction': trade['direction'],
                'Entry': f"${trade['entry_price']:.2f}",
                'Current': f"${current_price:.2f}",
                'VWAP': f"${vwap:.2f}",
                'Entry vs VWAP': f"{entry_vs_vwap:+.1f}%",
                'Size': f"${trade['size']:.2f}",
                'P&L %': f"{pnl:+.2f}%",
                'P&L 
            })
        
        # Display summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total P&L", f"${total_pnl:+.2f}")
        with col2:
            st.metric("Active Trades", len(st.session_state.trade_history))
        with col3:
            win_rate = sum(1 for t in trades_data if '+' in t['P&L %']) / len(trades_data) * 100
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Display trades table
        df = pd.DataFrame(trades_data)
        st.dataframe(df, use_container_width=True)
        
        if st.button("Clear All Trades"):
            st.session_state.trade_history = []
            st.rerun()

# Footer
st.divider()
st.caption("⚠️ This tool is for educational purposes only. Not financial advice. Trade responsibly!")

# Tips
with st.expander("💡 Trading Tips - VWAP & Order Flow"):
    st.markdown("""
    ### VWAP Trading Strategies
    
    **1. Mean Reversion**
    - Buy when price is below VWAP with RSI < 30
    - Sell when price is above VWAP with RSI > 70
    - Target: Return to VWAP
    
    **2. Trend Following**
    - Long bias when price stays above VWAP
    - Short bias when price stays below VWAP
    - Use VWAP as dynamic stop loss
    
    ### Order Flow Imbalance
    
    **1. Imbalance Signals**
    - **>30%**: Strong buying pressure
    - **<-30%**: Strong selling pressure
    - **-20% to 20%**: Neutral zone
    
    **2. Divergence Trading**
    - Price down + positive flow = Bullish divergence
    - Price up + negative flow = Bearish divergence
    
    **3. Flow Confirmation**
    - Only take longs with positive flow
    - Only take shorts with negative flow
    - Avoid trades against strong flow
    
    ### Risk Management with VWAP
    - Use VWAP bands as stop loss levels
    - Reduce position size when extended from VWAP
    - Take partial profits at VWAP
    """)

# Auto-refresh logic
current_time = datetime.now()
if (current_time - st.session_state.last_update).total_seconds() >= refresh_rate:
    st.session_state.last_update = current_time
    st.rerun(): f"${pnl_usd:+.2f}"
            })
        
        # Display summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total P&L", f"${total_pnl:+.2f}")
        with col2:
            st.metric("Active Trades", len(st.session_state.trade_history))
        with col3:
            win_rate = sum(1 for t in trades_data if '+' in t['P&L %']) / len(trades_data) * 100
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Display trades table
        df = pd.DataFrame(trades_data)
        st.dataframe(df, use_container_width=True)
        
        if st.button("Clear All Trades"):
            st.session_state.trade_history = []
            st.rerun()

# Footer
st.divider()
st.caption("⚠️ This tool is for educational purposes only. Not financial advice. Trade responsibly!")

# Tips
with st.expander("💡 Trading Tips - VWAP & Order Flow"):
    st.markdown("""
    ### VWAP Trading Strategies
    
    **1. Mean Reversion**
    - Buy when price is below VWAP with RSI < 30
    - Sell when price is above VWAP with RSI > 70
    - Target: Return to VWAP
    
    **2. Trend Following**
    - Long bias when price stays above VWAP
    - Short bias when price stays below VWAP
    - Use VWAP as dynamic stop loss
    
    ### Order Flow Imbalance
    
    **1. Imbalance Signals**
    - **>30%**: Strong buying pressure
    - **<-30%**: Strong selling pressure
    - **-20% to 20%**: Neutral zone
    
    **2. Divergence Trading**
    - Price down + positive flow = Bullish divergence
    - Price up + negative flow = Bearish divergence
    
    **3. Flow Confirmation**
    - Only take longs with positive flow
    - Only take shorts with negative flow
    - Avoid trades against strong flow
    
    ### Risk Management with VWAP
    - Use VWAP bands as stop loss levels
    - Reduce position size when extended from VWAP
    - Take partial profits at VWAP
    """)

# Auto-refresh logic
current_time = datetime.now()
if (current_time - st.session_state.last_update).total_seconds() >= refresh_rate:
    st.session_state.last_update = current_time
    st.rerun()
