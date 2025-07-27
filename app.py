import streamlit as st
import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# Page configuration
st.set_page_config(
    page_title="Advanced Crypto Trade Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Advanced Crypto Trade Analyzer")

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

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    symbols = st.multiselect(
        "Select symbols to analyze:",
        ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT", "BNBUSDT"],
        default=["BTCUSDT", "ETHUSDT"]
    )
    leverage = st.slider("Set your leverage (X):", 1, 50, 10)
    
    st.header("🔄 Auto-Refresh Settings")
    auto_refresh = st.checkbox("Enable Auto-Refresh", value=True)
    refresh_rate = st.slider("Refresh every (seconds):", 5, 120, 15)
    
    # Alert Settings
    st.header("🔔 Alert Settings")
    enable_alerts = st.checkbox("Enable Alerts", value=False)
    if enable_alerts:
        alert_prob_threshold = st.slider("Alert when probability >", 60, 90, 70)
    
    if st.button("🔄 Manual Refresh"):
        st.session_state.last_update = datetime.now() - timedelta(seconds=refresh_rate)
        st.rerun()
    
    if st.button("🗑️ Clear Alerts"):
        st.session_state.alerts = []
    
    # NEW: Price Alerts Section
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
    
    # NEW: Signal History
    if st.button("📜 View Signal History"):
        if st.session_state.signal_history:
            st.write(f"Last {min(5, len(st.session_state.signal_history))} signals:")
            for signal in st.session_state.signal_history[-5:]:
                st.write(f"• {signal['timestamp'].strftime('%H:%M')} - {signal['symbol']} {signal['signal']}")

# Auto-refresh logic
if auto_refresh:
    current_time = datetime.now()
    time_diff = (current_time - st.session_state.last_update).total_seconds()
    
    if time_diff >= refresh_rate:
        st.session_state.last_update = current_time
        st.rerun()

# Enhanced data fetching
@st.cache_data(ttl=5)
def get_enhanced_data(symbol):
    """Fetch enhanced data with technical indicators"""
    
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
            kline_url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=30"
            kline_response = requests.get(kline_url, timeout=3)
            
            rsi = 50
            macd = 0
            
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
            
            price_data = {
                "current_price": float(stats['lastPrice']),
                "price_change_24h": float(stats['priceChangePercent']),
                "volume": float(stats['volume']),
                "high_24h": float(stats['highPrice']),
                "low_24h": float(stats['lowPrice']),
                "rsi": round(rsi, 2),
                "macd": round(macd, 2),
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
    
    # Method 2: Binance US API (backup)
    try:
        url = f"https://api.binance.us/api/v3/ticker/24hr?symbol={symbol}"
        response = requests.get(url, timeout=3)
        
        if response.status_code == 200:
            stats = response.json()
            
            price_data = {
                "current_price": float(stats['lastPrice']),
                "price_change_24h": float(stats['priceChangePercent']),
                "volume": float(stats['volume']),
                "high_24h": float(stats['highPrice']),
                "low_24h": float(stats['lowPrice']),
                "rsi": 50,  # Default
                "macd": 0,
                "data_source": "Binance US",
                "last_updated": datetime.now().strftime("%H:%M:%S"),
                "bid": float(stats.get('bidPrice', stats['lastPrice'])),
                "ask": float(stats.get('askPrice', stats['lastPrice'])),
                "spread": 0
            }
            attempts.append("Binance US: Success")
            return price_data
    except Exception as e:
        attempts.append(f"Binance US: {str(e)[:30]}")
    
    # Method 3: CryptoCompare API
    try:
        # Extract base currency from symbol (remove USDT)
        base_symbol = symbol.replace('USDT', '')
        url = f"https://min-api.cryptocompare.com/data/pricemultifull?fsyms={base_symbol}&tsyms=USD"
        response = requests.get(url, timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            if 'DISPLAY' in data and base_symbol in data['DISPLAY']:
                raw_data = data['RAW'][base_symbol]['USD']
                display_data = data['DISPLAY'][base_symbol]['USD']
                
                price_data = {
                    "current_price": float(raw_data['PRICE']),
                    "price_change_24h": float(raw_data['CHANGEPCT24HOUR']),
                    "volume": float(raw_data['VOLUME24HOUR']),
                    "high_24h": float(raw_data['HIGH24HOUR']),
                    "low_24h": float(raw_data['LOW24HOUR']),
                    "rsi": 50,
                    "macd": 0,
                    "data_source": "CryptoCompare",
                    "last_updated": datetime.now().strftime("%H:%M:%S"),
                    "bid": float(raw_data['PRICE']) * 0.999,  # Estimate
                    "ask": float(raw_data['PRICE']) * 1.001,  # Estimate
                    "spread": 0.2  # Estimate
                }
                attempts.append("CryptoCompare: Success")
                return price_data
    except Exception as e:
        attempts.append(f"CryptoCompare: {str(e)[:30]}")
    
    # Method 4: CoinGecko (very reliable but rate limited)
    try:
        symbol_map = {
            "BTCUSDT": "bitcoin",
            "ETHUSDT": "ethereum",
            "BNBUSDT": "binancecoin",
            "SOLUSDT": "solana",
            "ADAUSDT": "cardano",
            "DOTUSDT": "polkadot",
            "LINKUSDT": "chainlink"
        }
        
        if symbol in symbol_map:
            coin_id = symbol_map[symbol]
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                market_data = data['market_data']
                
                price_data = {
                    "current_price": float(market_data['current_price']['usd']),
                    "price_change_24h": float(market_data['price_change_percentage_24h']),
                    "volume": float(market_data['total_volume']['usd']),
                    "high_24h": float(market_data['high_24h']['usd']),
                    "low_24h": float(market_data['low_24h']['usd']),
                    "rsi": 50,
                    "macd": 0,
                    "data_source": "CoinGecko",
                    "last_updated": datetime.now().strftime("%H:%M:%S"),
                    "bid": float(market_data['current_price']['usd']) * 0.999,
                    "ask": float(market_data['current_price']['usd']) * 1.001,
                    "spread": 0.2
                }
                attempts.append("CoinGecko: Success")
                return price_data
    except Exception as e:
        attempts.append(f"CoinGecko: {str(e)[:30]}")
    
    # If all methods fail, show what went wrong
    error_msg = " | ".join(attempts)
    st.warning(f"All price sources failed for {symbol}: {error_msg}")
    
    # Return demo data as last resort
    return {
        "current_price": 50000.0 if "BTC" in symbol else 3000.0 if "ETH" in symbol else 100.0,
        "price_change_24h": np.random.uniform(-5, 5),
        "volume": np.random.uniform(500000, 2000000),
        "high_24h": 51000 if "BTC" in symbol else 3100 if "ETH" in symbol else 105,
        "low_24h": 49000 if "BTC" in symbol else 2900 if "ETH" in symbol else 95,
        "rsi": np.random.uniform(30, 70),
        "macd": np.random.uniform(-5, 5),
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

# NEW FUNCTION: Generate trading signals for different strategies
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
    
    # Perpetual Trading Signals (Higher leverage, shorter timeframe)
    if strategy_type in ["perp", "all"]:
        perp_signal = "WAIT"
        perp_action = None
        
        # Strong oversold bounce
        if rsi < 25 and change_24h < -5:
            perp_signal = "BUY"
            perp_action = {
                "entry": price,
                "stop_loss": price * 0.98,  # 2% stop loss
                "take_profit": price * 1.03,  # 3% take profit
                "leverage": min(10, leverage),
                "reason": "Oversold bounce setup"
            }
        # Strong overbought short
        elif rsi > 75 and change_24h > 5:
            perp_signal = "SELL"
            perp_action = {
                "entry": price,
                "stop_loss": price * 1.02,
                "take_profit": price * 0.97,
                "leverage": min(10, leverage),
                "reason": "Overbought reversal setup"
            }
        # Momentum breakout
        elif change_24h > 3 and macd > 0 and rsi < 65:
            perp_signal = "BUY"
            perp_action = {
                "entry": price,
                "stop_loss": price * 0.97,
                "take_profit": price * 1.05,
                "leverage": min(5, leverage),
                "reason": "Momentum breakout"
            }
        
        signals["perp"] = {
            "signal": perp_signal,
            "action": perp_action
        }
    
    # Day Trading Signals (Lower leverage, medium timeframe)
    if strategy_type in ["day_trading", "all"]:
        day_signal = "WAIT"
        day_action = None
        
        # Morning dip buy
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 11 and change_24h < -2 and rsi < 40:
            day_signal = "BUY"
            day_action = {
                "entry": price,
                "stop_loss": price * 0.995,  # 0.5% stop loss
                "take_profit": price * 1.015,  # 1.5% take profit
                "size": "50%",  # Use 50% of capital
                "reason": "Morning dip opportunity"
            }
        # Trend following
        elif macd > 0 and rsi > 45 and rsi < 65 and change_24h > 0:
            day_signal = "BUY"
            day_action = {
                "entry": price,
                "stop_loss": price * 0.99,
                "take_profit": price * 1.025,
                "size": "30%",
                "reason": "Trend continuation"
            }
        # Range trading
        elif 40 < rsi < 60 and abs(change_24h) < 2:
            if rsi < 45:
                day_signal = "BUY"
                day_action = {
                    "entry": price,
                    "stop_loss": price * 0.995,
                    "take_profit": price * 1.01,
                    "size": "25%",
                    "reason": "Range trading - Buy low"
                }
            elif rsi > 55:
                day_signal = "SELL"
                day_action = {
                    "entry": price,
                    "stop_loss": price * 1.005,
                    "take_profit": price * 0.99,
                    "size": "25%",
                    "reason": "Range trading - Sell high"
                }
        
        signals["day_trading"] = {
            "signal": day_signal,
            "action": day_action
        }
    
    # Spot Trading Signals (No leverage, longer timeframe)
    if strategy_type in ["spot", "all"]:
        spot_signal = "WAIT"
        spot_action = None
        
        # Deep value buy
        if rsi < 30 and change_24h < -10:
            spot_signal = "STRONG BUY"
            spot_action = {
                "entry": price,
                "target_1": price * 1.10,  # 10% target
                "target_2": price * 1.20,  # 20% target
                "size": "10-20%",
                "reason": "Deep value opportunity"
            }
        # DCA opportunity
        elif rsi < 40 and change_24h < -5:
            spot_signal = "BUY"
            spot_action = {
                "entry": price,
                "target": price * 1.15,
                "size": "5-10%",
                "reason": "DCA opportunity"
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

# NEW FUNCTION: Check price alerts
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

# NEW FUNCTION: Save signal to history
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

# Main interface
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
        st.caption(f"Last updated: {st.session_state.last_update.strftime('%H:%M:%S')} | Auto-refresh {'ON' if auto_refresh else 'OFF'}")
        
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
        
        # Create simple price chart
        fig = go.Figure()
        
        # Generate sample price data
        hours = 24
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(hours, 0, -1)]
        base_price = data['current_price']
        
        # Simple price simulation
        prices = []
        for i in range(hours):
            variation = np.random.uniform(-1, 1) / 100
            price = base_price * (1 + variation)
            prices.append(price)
            base_price = price
        
        # Make sure last price matches current
        prices[-1] = data['current_price']
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=prices,
            mode='lines',
            name='Price',
            line=dict(color='cyan', width=2)
        ))
        
        fig.update_layout(
            title=f"{selected_symbol} Price Chart (24h)",
            xaxis_title="Time",
            yaxis_title="Price ($)",
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display current stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current", f"${data['current_price']:,.2f}")
        with col2:
            st.metric("24h High", f"${data['high_24h']:,.2f}")
        with col3:
            st.metric("24h Low", f"${data['low_24h']:,.2f}")
        with col4:
            st.metric("24h Change", f"{data['price_change_24h']:+.2f}%")

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
    
    # Display trades
    if st.session_state.trade_history:
        st.subheader("Active Trades")
        
        trades_data = []
        total_pnl = 0
        
        for trade in st.session_state.trade_history:
            current_data = get_enhanced_data(trade['symbol'])
            current_price = current_data['current_price']
            
            if trade['direction'] == 'Long':
                pnl = (current_price - trade['entry_price']) / trade['entry_price'] * 100
            else:
                pnl = (trade['entry_price'] - current_price) / trade['entry_price'] * 100
            
            pnl_usd = pnl / 100 * trade['size']
            total_pnl += pnl_usd
            
            trades_data.append({
                'Symbol': trade['symbol'],
                'Direction': trade['direction'],
                'Entry': f"${trade['entry_price']:.2f}",
                'Current': f"${current_price:.2f}",
                'Size': f"${trade['size']:.2f}",
                'P&L %': f"{pnl:+.2f}%",
                'P&L $': f"${pnl_usd:+.2f}"
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
with st.expander("💡 Trading Tips"):
    st.markdown("""
    ### Risk Management
    - Never risk more than 2% per trade
    - Use stop losses always
    - Start with low leverage
    - Diversify your positions
    
    ### Signal Interpretation
    - **Strong signals (>65%)**: High confidence trades
    - **Moderate signals (55-65%)**: Normal trades
    - **Weak signals (<55%)**: Wait or opposite direction
    
    ### Best Practices
    - Trade with the trend
    - Wait for confirmation
    - Keep a trading journal
    - Review your performance regularly
    """)
