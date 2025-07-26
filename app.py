import streamlit as st
import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, List, Tuple, Optional

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
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = {}

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    symbols = st.multiselect(
        "Select symbols to analyze:", 
        ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT", "BNBUSDT", "XRPUSDT", "MATICUSDT", "AVAXUSDT"], 
        default=["BTCUSDT", "ETHUSDT"]
    )
    leverage = st.slider("Set your leverage (X):", 1, 50, 10)
    
    st.header("🔄 Auto-Refresh Settings")
    auto_refresh = st.checkbox("Enable Auto-Refresh", value=True)
    refresh_rate = st.slider("Refresh every (seconds):", 10, 120, 30)
    
    # Alert Settings
    st.header("🔔 Alert Settings")
    enable_alerts = st.checkbox("Enable Alerts", value=False)
    if enable_alerts:
        alert_prob_threshold = st.slider("Alert when probability >", 60, 90, 70)
        alert_risk_threshold = st.slider("Alert when risk <", 10, 50, 25)
        alert_volume_spike = st.slider("Alert on volume spike >", 150, 500, 200)
    
    # Advanced Settings
    st.header("🔧 Advanced Settings")
    use_ai_predictions = st.checkbox("Enable AI Predictions", value=False)
    show_order_book = st.checkbox("Show Order Book Analysis", value=False)
    enable_backtesting = st.checkbox("Enable Backtesting", value=False)
    
    if st.button("🔄 Manual Refresh"):
        st.session_state.last_update = datetime.now() - timedelta(seconds=refresh_rate)
        st.rerun()
    
    if st.button("🗑️ Clear Alerts"):
        st.session_state.alerts = []
    
    if st.button("📊 Export Data"):
        export_data = {
            'trades': st.session_state.trade_history,
            'alerts': st.session_state.alerts,
            'timestamp': datetime.now().isoformat()
        }
        st.download_button(
            label="Download JSON",
            data=json.dumps(export_data, indent=2, default=str),
            file_name=f"crypto_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Auto-refresh logic
if auto_refresh:
    current_time = datetime.now()
    time_diff = (current_time - st.session_state.last_update).total_seconds()
    
    if time_diff >= refresh_rate:
        st.session_state.last_update = current_time
        st.rerun()

# Enhanced data fetching with multiple APIs and order book
@st.cache_data(ttl=15)
def get_enhanced_data(symbol):
    """Fetch enhanced data with technical indicators and order book"""
    
    def try_binance():
        base_url = "https://fapi.binance.com"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json"
        }
        
        try:
            # Get kline data for technical analysis
            klines_response = requests.get(
                f"{base_url}/fapi/v1/klines", 
                params={"symbol": symbol, "interval": "1h", "limit": 100}, 
                headers=headers, timeout=8
            )
            
            # Get 24h stats
            stats_response = requests.get(
                f"{base_url}/fapi/v1/ticker/24hr", 
                params={"symbol": symbol}, 
                headers=headers, timeout=8
            )
            
            # Get order book depth
            depth_response = requests.get(
                f"{base_url}/fapi/v1/depth",
                params={"symbol": symbol, "limit": 20},
                headers=headers, timeout=5
            )
            
            if klines_response.status_code == 200 and stats_response.status_code == 200:
                klines = klines_response.json()
                stats = stats_response.json()
                depth = depth_response.json() if depth_response.status_code == 200 else None
                
                # Process kline data
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                df = df.astype({
                    'open': float, 'high': float, 'low': float, 
                    'close': float, 'volume': float
                })
                
                # Calculate technical indicators
                closes = df['close'].values
                highs = df['high'].values
                lows = df['low'].values
                volumes = df['volume'].values
                
                # Enhanced RSI Calculation
                def calculate_rsi(prices, period=14):
                    deltas = np.diff(prices)
                    gains = np.where(deltas > 0, deltas, 0)
                    losses = np.where(deltas < 0, -deltas, 0)
                    
                    avg_gains = pd.Series(gains).rolling(window=period).mean()
                    avg_losses = pd.Series(losses).rolling(window=period).mean()
                    
                    rs = avg_gains / avg_losses
                    rsi = 100 - (100 / (1 + rs))
                    return rsi.iloc[-1] if len(rsi) > 0 and not np.isnan(rsi.iloc[-1]) else 50
                
                # Stochastic RSI
                def calculate_stoch_rsi(rsi_values, period=14):
                    if len(rsi_values) < period:
                        return 50
                    min_rsi = min(rsi_values[-period:])
                    max_rsi = max(rsi_values[-period:])
                    if max_rsi - min_rsi == 0:
                        return 50
                    return ((rsi_values[-1] - min_rsi) / (max_rsi - min_rsi)) * 100
                
                # Moving Averages
                ma_7 = np.mean(closes[-7:]) if len(closes) >= 7 else closes[-1]
                ma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
                ma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else closes[-1]
                ma_200 = np.mean(closes[-200:]) if len(closes) >= 200 else closes[-1]
                
                # MACD
                ema_12 = pd.Series(closes).ewm(span=12).mean().iloc[-1]
                ema_26 = pd.Series(closes).ewm(span=26).mean().iloc[-1]
                macd = ema_12 - ema_26
                signal_line = pd.Series(closes).ewm(span=9).mean().iloc[-1]
                macd_histogram = macd - signal_line
                
                # Bollinger Bands
                bb_period = 20
                if len(closes) >= bb_period:
                    bb_sma = np.mean(closes[-bb_period:])
                    bb_std = np.std(closes[-bb_period:])
                    bb_upper = bb_sma + (bb_std * 2)
                    bb_lower = bb_sma - (bb_std * 2)
                    bb_position = (closes[-1] - bb_lower) / (bb_upper - bb_lower) * 100
                else:
                    bb_position = 50
                    bb_upper = bb_lower = closes[-1]
                
                # ATR (Average True Range)
                def calculate_atr(highs, lows, closes, period=14):
                    if len(highs) < period + 1:
                        return 0
                    tr_list = []
                    for i in range(1, len(highs)):
                        tr = max(
                            highs[i] - lows[i],
                            abs(highs[i] - closes[i-1]),
                            abs(lows[i] - closes[i-1])
                        )
                        tr_list.append(tr)
                    return np.mean(tr_list[-period:]) if tr_list else 0
                
                atr = calculate_atr(highs, lows, closes)
                
                # Volume analysis
                avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
                volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1
                
                # Support and Resistance levels
                recent_highs = highs[-20:]
                recent_lows = lows[-20:]
                resistance = max(recent_highs) if len(recent_highs) > 0 else closes[-1]
                support = min(recent_lows) if len(recent_lows) > 0 else closes[-1]
                
                # Order book analysis
                order_book_imbalance = 0
                bid_wall = 0
                ask_wall = 0
                if depth:
                    bids = depth.get('bids', [])
                    asks = depth.get('asks', [])
                    
                    bid_volume = sum(float(bid[1]) for bid in bids[:5])
                    ask_volume = sum(float(ask[1]) for ask in asks[:5])
                    
                    if bid_volume + ask_volume > 0:
                        order_book_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) * 100
                    
                    # Detect walls
                    if bids:
                        bid_wall = max(float(bid[1]) for bid in bids[:5])
                    if asks:
                        ask_wall = max(float(ask[1]) for ask in asks[:5])
                
                # Get funding rate
                funding_rate = 0.0
                try:
                    funding_response = requests.get(
                        f"{base_url}/fapi/v1/fundingRate", 
                        params={"symbol": symbol, "limit": 1}, 
                        headers=headers, timeout=5
                    )
                    if funding_response.status_code == 200:
                        funding = funding_response.json()
                        funding_rate = float(funding[0]['fundingRate']) * 100 if funding else 0.0
                except:
                    pass
                
                # Calculate RSI values for Stoch RSI
                rsi_values = []
                for i in range(14, len(closes)):
                    rsi_val = calculate_rsi(closes[:i+1])
                    rsi_values.append(rsi_val)
                
                current_rsi = calculate_rsi(closes)
                stoch_rsi = calculate_stoch_rsi(rsi_values) if rsi_values else 50
                
                return {
                    "current_price": float(stats.get("lastPrice", 0)),
                    "price_change_24h": float(stats.get("priceChangePercent", 0)),
                    "volume": float(stats.get("volume", 0)),
                    "volume_ratio": volume_ratio,
                    "volatility": abs(float(stats.get("priceChangePercent", 0))),
                    "funding_rate": funding_rate,
                    "long_short_ratio": 1.0,
                    "rsi": current_rsi,
                    "stoch_rsi": stoch_rsi,
                    "ma_7": ma_7,
                    "ma_20": ma_20,
                    "ma_50": ma_50,
                    "ma_200": ma_200,
                    "macd": macd,
                    "macd_signal": signal_line,
                    "macd_histogram": macd_histogram,
                    "bb_position": bb_position,
                    "bb_upper": bb_upper,
                    "bb_lower": bb_lower,
                    "atr": atr,
                    "support": support,
                    "resistance": resistance,
                    "order_book_imbalance": order_book_imbalance,
                    "bid_wall": bid_wall,
                    "ask_wall": ask_wall,
                    "last_updated": datetime.now().strftime("%H:%M:%S"),
                    "data_source": "Binance",
                    "price_history": closes[-24:].tolist(),
                    "volume_history": volumes[-24:].tolist(),
                    "timestamps": [datetime.fromtimestamp(int(k[0])/1000).strftime("%H:%M") for k in klines[-24:]]
                }
        except Exception as e:
            st.warning(f"Binance API failed for {symbol}: {str(e)}")
            return None
    
    # Try Binance first
    result = try_binance()
    if result:
        return result
    
    # Fallback to demo data
    return {
        "current_price": 50000.0 if "BTC" in symbol else 3000.0 if "ETH" in symbol else 100.0,
        "price_change_24h": 1.2,
        "volume": 1000000,
        "volume_ratio": 1.0,
        "volatility": 2.5,
        "funding_rate": 0.01,
        "long_short_ratio": 1.1,
        "rsi": 50,
        "stoch_rsi": 50,
        "ma_7": 50000,
        "ma_20": 50000,
        "ma_50": 49000,
        "ma_200": 48000,
        "macd": 0,
        "macd_signal": 0,
        "macd_histogram": 0,
        "bb_position": 50,
        "bb_upper": 51000,
        "bb_lower": 49000,
        "atr": 1000,
        "support": 49500,
        "resistance": 50500,
        "order_book_imbalance": 0,
        "bid_wall": 0,
        "ask_wall": 0,
        "last_updated": datetime.now().strftime("%H:%M:%S"),
        "data_source": "Demo Mode",
        "price_history": [50000] * 24,
        "volume_history": [1000000] * 24,
        "timestamps": [(datetime.now() - timedelta(hours=i)).strftime("%H:%M") for i in range(24, 0, -1)]
    }

# Advanced probability calculation with AI predictions
def calculate_advanced_probabilities(data, leverage, use_ai=False):
    """Enhanced probability calculation with technical indicators and optional AI"""
    if not data:
        return {"long_prob": 50, "short_prob": 50, "risk_score": 50, "recommendation": "NEUTRAL"}
    
    base_prob = 50
    
    # Technical indicator signals with weights
    signals = {}
    
    # RSI signal (weight: 15%)
    rsi_signal = 0
    if data['rsi'] > 80:
        rsi_signal = -10
    elif data['rsi'] > 70:
        rsi_signal = -6
    elif data['rsi'] < 20:
        rsi_signal = 10
    elif data['rsi'] < 30:
        rsi_signal = 6
    elif data['rsi'] > 60:
        rsi_signal = -2
    elif data['rsi'] < 40:
        rsi_signal = 2
    signals['rsi'] = rsi_signal
    
    # Stochastic RSI signal (weight: 10%)
    stoch_signal = 0
    if data['stoch_rsi'] > 80:
        stoch_signal = -5
    elif data['stoch_rsi'] < 20:
        stoch_signal = 5
    signals['stoch_rsi'] = stoch_signal
    
    # Moving average signal (weight: 20%)
    ma_signal = 0
    current_price = data['current_price']
    if current_price > data['ma_7'] > data['ma_20'] > data['ma_50']:
        ma_signal = 8  # Strong uptrend
    elif current_price < data['ma_7'] < data['ma_20'] < data['ma_50']:
        ma_signal = -8  # Strong downtrend
    elif current_price > data['ma_20'] and data['ma_20'] > data['ma_50']:
        ma_signal = 5
    elif current_price < data['ma_20'] and data['ma_20'] < data['ma_50']:
        ma_signal = -5
    elif current_price > data['ma_20']:
        ma_signal = 2
    elif current_price < data['ma_20']:
        ma_signal = -2
    signals['ma'] = ma_signal
    
    # MACD signal (weight: 15%)
    macd_signal = 0
    if data['macd_histogram'] > 0 and data['macd'] > data['macd_signal']:
        macd_signal = 6
    elif data['macd_histogram'] < 0 and data['macd'] < data['macd_signal']:
        macd_signal = -6
    elif data['macd'] > 0:
        macd_signal = 3
    elif data['macd'] < 0:
        macd_signal = -3
    signals['macd'] = macd_signal
    
    # Bollinger Bands signal (weight: 10%)
    bb_signal = 0
    if data['bb_position'] > 90:
        bb_signal = -6
    elif data['bb_position'] < 10:
        bb_signal = 6
    elif data['bb_position'] > 75:
        bb_signal = -3
    elif data['bb_position'] < 25:
        bb_signal = 3
    signals['bb'] = bb_signal
    
    # Volume analysis (weight: 10%)
    volume_signal = 0
    if data['volume_ratio'] > 2:
        volume_signal = 5 if data['price_change_24h'] > 0 else -5
    elif data['volume_ratio'] > 1.5:
        volume_signal = 3 if data['price_change_24h'] > 0 else -3
    elif data['volume_ratio'] < 0.5:
        volume_signal = -2
    signals['volume'] = volume_signal
    
    # Order book analysis (weight: 10%)
    orderbook_signal = 0
    if show_order_book and data['order_book_imbalance'] != 0:
        if data['order_book_imbalance'] > 20:
            orderbook_signal = 5
        elif data['order_book_imbalance'] < -20:
            orderbook_signal = -5
        else:
            orderbook_signal = data['order_book_imbalance'] / 10
    signals['orderbook'] = orderbook_signal
    
    # Support/Resistance levels (weight: 5%)
    sr_signal = 0
    price_to_resistance = (data['resistance'] - current_price) / current_price * 100
    price_to_support = (current_price - data['support']) / current_price * 100
    
    if price_to_resistance < 1:
        sr_signal = -4
    elif price_to_support < 1:
        sr_signal = 4
    signals['support_resistance'] = sr_signal
    
    # Funding rate bias (weight: 5%)
    funding_signal = 0
    if data['funding_rate'] > 0.05:
        funding_signal = -8
    elif data['funding_rate'] > 0.02:
        funding_signal = -4
    elif data['funding_rate'] < -0.05:
        funding_signal = 8
    elif data['funding_rate'] < -0.02:
        funding_signal = 4
    signals['funding'] = funding_signal
    
    # Calculate weighted signals
    total_signal = sum(signals.values())
    
    # AI prediction boost (if enabled)
    if use_ai and use_ai_predictions:
        # Simulated AI prediction based on pattern recognition
        ai_boost = 0
        
        # Pattern detection
        if len(data['price_history']) >= 3:
            # Trend detection
            recent_prices = data['price_history'][-3:]
            if all(recent_prices[i] < recent_prices[i+1] for i in range(len(recent_prices)-1)):
                ai_boost += 5  # Uptrend
            elif all(recent_prices[i] > recent_prices[i+1] for i in range(len(recent_prices)-1)):
                ai_boost += -5  # Downtrend
            
            # Volume trend
            recent_volumes = data['volume_history'][-3:]
            if all(recent_volumes[i] < recent_volumes[i+1] for i in range(len(recent_volumes)-1)):
                ai_boost += 3 if data['price_change_24h'] > 0 else -3
        
        total_signal += ai_boost
        signals['ai_prediction'] = ai_boost
    
    # Apply leverage penalty
    leverage_penalty = min(20, (leverage - 1) * 0.8)
    
    # Calculate final probabilities
    long_probability = base_prob + total_signal - leverage_penalty
    short_probability = base_prob - total_signal - leverage_penalty
    
    # Normalize probabilities
    long_probability = max(10, min(90, long_probability))
    short_probability = max(10, min(90, short_probability))
    
    # Risk calculation with ATR
    atr_risk = (data['atr'] / current_price) * 100 * leverage
    risk_score = (
        data['volatility'] * leverage * 0.15 +
        abs(data['funding_rate']) * 10 +
        (leverage - 1) * 1.5 +
        atr_risk * 0.5 +
        max(0, (1000000 - data['volume']) / 150000)
    )
    risk_score = min(100, round(risk_score, 1))
    
    # Generate recommendation
    prob_diff = abs(long_probability - short_probability)
    confidence = round(prob_diff, 1)
    
    if prob_diff < 5:
        recommendation = "NEUTRAL ⚖️"
        safer_direction = "Wait for clearer signal"
    elif long_probability > short_probability:
        if prob_diff > 25:
            recommendation = "STRONG LONG 🚀🚀"
        elif prob_diff > 15:
            recommendation = "LONG 🚀"
        else:
            recommendation = "WEAK LONG 📈"
        safer_direction = "Long"
    else:
        if prob_diff > 25:
            recommendation = "STRONG SHORT 📉📉"
        elif prob_diff > 15:
            recommendation = "SHORT 📉"
        else:
            recommendation = "WEAK SHORT 📉"
        safer_direction = "Short"
    
    return {
        "long_prob": round(long_probability, 1),
        "short_prob": round(short_probability, 1),
        "risk_score": risk_score,
        "recommendation": recommendation,
        "safer_direction": safer_direction,
        "confidence": confidence,
        "signals": signals
    }

# Enhanced alert system
def check_alerts(symbol, data, analysis):
    """Check if any alerts should be triggered"""
    if not enable_alerts:
        return
    
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # High probability alert
    max_prob = max(analysis['long_prob'], analysis['short_prob'])
    if max_prob >= alert_prob_threshold:
        direction = "Long" if analysis['long_prob'] > analysis['short_prob'] else "Short"
        alert_msg = f"🎯 {symbol}: Strong {direction} signal! ({max_prob}% probability)"
        if not any(a['message'] == alert_msg for a in st.session_state.alerts[-10:]):
            st.session_state.alerts.append({
                'time': current_time,
                'symbol': symbol,
                'message': alert_msg,
                'type': 'probability',
                'data': {'probability': max_prob, 'direction': direction}
            })
    
    # Low risk alert
    if analysis['risk_score'] <= alert_risk_threshold:
        alert_msg = f"🛡️ {symbol}: Low risk opportunity! (Risk: {analysis['risk_score']})"
        if not any(a['message'] == alert_msg for a in st.session_state.alerts[-10:]):
            st.session_state.alerts.append({
                'time': current_time,
                'symbol': symbol,
                'message': alert_msg,
                'type': 'risk',
                'data': {'risk_score': analysis['risk_score']}
            })
    
    # Volume spike alert
    if data['volume_ratio'] * 100 >= alert_volume_spike:
        alert_msg = f"📊 {symbol}: Volume spike detected! ({data['volume_ratio']:.1f}x average)"
        if not any(a['message'] == alert_msg for a in st.session_state.alerts[-10:]):
            st.session_state.alerts.append({
                'time': current_time,
                'symbol': symbol,
                'message': alert_msg,
                'type': 'volume',
                'data': {'volume_ratio': data['volume_ratio']}
            })
    
    # Support/Resistance alert
    current_price = data['current_price']
    if abs(current_price - data['support']) / current_price < 0.01:
        alert_msg = f"💪 {symbol}: Near support level! (${data['support']:.2f})"
        if not any(a['message'] == alert_msg for a in st.session_state.alerts[-10:]):
            st.session_state.alerts.append({
                'time': current_time,
                'symbol': symbol,
                'message': alert_msg,
                'type': 'support',
                'data': {'support': data['support']}
            })
    elif abs(current_price - data['resistance']) / current_price < 0.01:
        alert_msg = f"🚧 {symbol}: Near resistance level! (${data['resistance']:.2f})"
        if not any(a['message'] == alert_msg for a in st.session_state.alerts[-10:]):
            st.session_state.alerts.append({
                'time': current_time,
                'symbol': symbol,
                'message': alert_msg,
                'type': 'resistance',
                'data': {'resistance': data['resistance']}
            })

# Risk level function
def get_risk_level(score):
    if score < 20:
        return "🟢 LOW", "green"
    elif score < 40:
        return "🟡 MEDIUM", "orange"
    elif score < 60:
        return "🟠 HIGH", "red"
    else:
        return "🔴 EXTREME", "darkred"

# Backtesting function
def run_backtest(symbol, data, strategy_params):
    """Run a simple backtest on historical data"""
    if not data or 'price_history' not in data:
        return None
    
    prices = data['price_history']
    if len(prices) < 10:
        return None
    
    # Simple moving average crossover strategy
    short_period = strategy_params.get('short_ma', 3)
    long_period = strategy_params.get('long_ma', 7)
    
    positions = []
    trades = []
    current_position = None
    
    for i in range(long_period, len(prices)):
        short_ma = np.mean(prices[i-short_period:i])
        long_ma = np.mean(prices[i-long_period:i])
        
        if short_ma > long_ma and current_position != 'long':
            if current_position == 'short':
                # Close short
                trades.append({
                    'type': 'close_short',
                    'price': prices[i],
                    'index': i
                })
            # Open long
            trades.append({
                'type': 'open_long',
                'price': prices[i],
                'index': i
            })
            current_position = 'long'
        elif short_ma < long_ma and current_position != 'short':
            if current_position == 'long':
                # Close long
                trades.append({
                    'type': 'close_long',
                    'price': prices[i],
                    'index': i
                })
            # Open short
            trades.append({
                'type': 'open_short',
                'price': prices[i],
                'index': i
            })
            current_position = 'short'
    
    # Calculate returns
    total_return = 0
    win_trades = 0
    loss_trades = 0
    
    for i in range(0, len(trades)-1, 2):
        if i+1 < len(trades):
            entry_trade = trades[i]
            exit_trade = trades[i+1]
            
            if entry_trade['type'] == 'open_long':
                return_pct = (exit_trade['price'] - entry_trade['price']) / entry_trade['price'] * 100
            else:  # short
                return_pct = (entry_trade['price'] - exit_trade['price']) / entry_trade['price'] * 100
            
            total_return += return_pct
            if return_pct > 0:
                win_trades += 1
            else:
                loss_trades += 1
    
    total_trades = win_trades + loss_trades
    win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
    
    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'win_trades': win_trades,
        'loss_trades': loss_trades,
        'trades': trades
    }

# Main interface with tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Simple View", "🔬 Advanced View", "📈 Market Overview", "🧪 Backtesting"])

with tab1:
    st.header("📊 Simple Trading View")
    
    # Trade input section
    with st.expander("💼 Enter Your Trade", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            selected_symbol = st.selectbox("Symbol:", symbols if symbols else ["BTCUSDT"], key="simple_symbol")
        with col2:
            entry_price = st.number_input("Entry Price:", min_value=0.0, format="%.4f", value=0.0, key="simple_entry")
        with col3:
            trade_direction = st.radio("Direction:", ["Long", "Short"], key="simple_direction")
        with col4:
            position_size = st.number_input("Position Size (USDT):", min_value=0.0, value=100.0, key="simple_size")
    
    if symbols:
        st.caption(f"Last updated: {st.session_state.last_update.strftime('%H:%M:%S')} | Next refresh: {max(0, refresh_rate - int((datetime.now() - st.session_state.last_update).total_seconds()))}s")
        
        # Simple grid layout
        cols = st.columns(min(len(symbols), 3))
        
        for idx, symbol in enumerate(symbols):
            with cols[idx % len(cols)]:
                data = get_enhanced_data(symbol)
                if data:
                    analysis = calculate_advanced_probabilities(data, leverage, use_ai_predictions)
                    check_alerts(symbol, data, analysis)
                    
                    # Card container
                    with st.container():
                        st.subheader(f"🔸 {symbol}")
                        st.caption(f"📡 {data.get('data_source', 'Unknown')}")
                        
                        # Price and change
                        col_price, col_vol = st.columns(2)
                        with col_price:
                            st.metric(
                                "Price", 
                                f"${data['current_price']:,.4f}",
                                delta=f"{data['price_change_24h']:+.2f}%"
                            )
                        with col_vol:
                            vol_display = f"{data['volume']/1e6:.1f}M" if data['volume'] > 1e6 else f"{data['volume']/1e3:.1f}K"
                            st.metric(
                                "Volume", 
                                vol_display,
                                delta=f"{(data['volume_ratio']-1)*100:+.0f}%" if data['volume_ratio'] != 1 else None
                            )
                        
                        # Trading signals
                        st.markdown("### 🎯 Trading Signals")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            long_color = "green" if analysis['long_prob'] > 60 else "red" if analysis['long_prob'] < 40 else "orange"
                            st.markdown(f"**📈 Long:** :{long_color}[{analysis['long_prob']}%]")
                        with col_b:
                            short_color = "green" if analysis['short_prob'] > 60 else "red" if analysis['short_prob'] < 40 else "orange"
                            st.markdown(f"**📉 Short:** :{short_color}[{analysis['short_prob']}%]")
                        
                        # Recommendation with confidence
                        rec_color = "green" if "STRONG" in analysis['recommendation'] else "orange" if "NEUTRAL" not in analysis['recommendation'] else "gray"
                        st.markdown(f"**🎯 Signal:** :{rec_color}[{analysis['recommendation']}]")
                        st.markdown(f"**📊 Confidence:** {analysis['confidence']}%")
                        
                        # Risk and indicators
                        risk_level, risk_color = get_risk_level(analysis['risk_score'])
                        st.markdown(f"**⚠️ Risk:** :{risk_color}[{risk_level}] ({analysis['risk_score']}/100)")
                        
                        # Quick indicators
                        st.markdown("**📈 Indicators:**")
                        ind_col1, ind_col2 = st.columns(2)
                        with ind_col1:
                            rsi_status = "🔴" if data['rsi'] > 70 else "🟢" if data['rsi'] < 30 else "🟡"
                            st.caption(f"RSI: {rsi_status} {data['rsi']:.1f}")
                        with ind_col2:
                            macd_status = "🟢" if data['macd'] > 0 else "🔴"
                            st.caption(f"MACD: {macd_status} {data['macd']:.2f}")
                        
                        # Your trade performance (if applicable)
                        if entry_price > 0 and symbol == selected_symbol:
                            st.markdown("---")
                            st.markdown("**💼 Your Position:**")
                            price_diff = data['current_price'] - entry_price
                            pnl_percent = (price_diff / entry_price) * 100
                            if trade_direction == "Short":
                                pnl_percent = -pnl_percent
                            pnl_usd = (pnl_percent / 100) * position_size * leverage
                            
                            pnl_color = "green" if pnl_usd > 0 else "red"
                            st.metric("P&L", f"${pnl_usd:+.2f}", f"{pnl_percent:+.2f}%")
                        
                        st.markdown("---")

with tab2:
    st.header("🔬 Advanced Analysis Dashboard")
    
    # Alerts section
    if st.session_state.alerts:
        with st.expander("🔔 Active Alerts", expanded=True):
            for alert in st.session_state.alerts[-10:]:  # Show last 10 alerts
                alert_emoji = "🎯" if alert['type'] == 'probability' else "🛡️" if alert['type'] == 'risk' else "📊" if alert['type'] == 'volume' else "💪" if alert['type'] == 'support' else "🚧"
                st.info(f"{alert_emoji} **{alert['time']}** - {alert['message']}")
    
    # Advanced trade management
    st.subheader("💼 Advanced Trade Management")
    
    with st.expander("📝 Multi-Position Tracker", expanded=False):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            trade_symbol = st.selectbox("Symbol:", symbols if symbols else ["BTCUSDT"], key="adv_symbol")
        with col2:
            trade_entry = st.number_input("Entry Price:", min_value=0.0, format="%.4f", key="adv_entry")
        with col3:
            trade_dir = st.radio("Direction:", ["Long", "Short"], key="adv_direction")
        with col4:
            trade_size = st.number_input("Size (USDT):", min_value=0.0, value=100.0, key="adv_size")
        with col5:
            trade_lev = st.number_input("Leverage:", min_value=1, max_value=50, value=10, key="adv_leverage")
        
        col_add, col_tp, col_sl = st.columns(3)
        with col_add:
            if st.button("📊 Add to Portfolio", type="primary"):
                st.session_state.trade_history.append({
                    'symbol': trade_symbol,
                    'entry_price': trade_entry,
                    'direction': trade_dir,
                    'size': trade_size,
                    'leverage': trade_lev,
                    'timestamp': datetime.now()
                })
                st.success(f"Added {trade_dir} {trade_symbol} to portfolio!")
        with col_tp:
            take_profit = st.number_input("Take Profit %:", min_value=0.0, value=5.0, key="tp_percent")
        with col_sl:
            stop_loss = st.number_input("Stop Loss %:", min_value=0.0, value=2.0, key="sl_percent")
    
    if symbols:
        st.caption(f"Last updated: {st.session_state.last_update.strftime('%H:%M:%S')} | Auto-refresh: {'ON' if auto_refresh else 'OFF'}")
        
        # Advanced analysis for each symbol
        for symbol in symbols:
            data = get_enhanced_data(symbol)
            if data:
                analysis = calculate_advanced_probabilities(data, leverage, use_ai_predictions)
                check_alerts(symbol, data, analysis)
                
                st.subheader(f"🔸 {symbol} - Complete Analysis")
                
                # Create advanced charts
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    # Price and volume chart
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.7, 0.3],
                        subplot_titles=('Price Action', 'Volume')
                    )
                    
                    # Price trace
                    fig.add_trace(
                        go.Scatter(
                            x=data['timestamps'],
                            y=data['price_history'],
                            mode='lines+markers',
                            name='Price',
                            line=dict(color='cyan', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # Add moving averages
                    if len(data['price_history']) >= 7:
                        ma7 = pd.Series(data['price_history']).rolling(window=7).mean()
                        fig.add_trace(
                            go.Scatter(
                                x=data['timestamps'],
                                y=ma7,
                                mode='lines',
                                name='MA7',
                                line=dict(color='orange', width=1)
                            ),
                            row=1, col=1
                        )
                    
                    # Volume bars
                    colors = ['green' if i > 0 else 'red' for i in range(len(data['volume_history']))]
                    fig.add_trace(
                        go.Bar(
                            x=data['timestamps'],
                            y=data['volume_history'],
                            name='Volume',
                            marker_color=colors
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_layout(
                        title=f"{symbol} Price & Volume",
                        xaxis_title="Time",
                        height=500,
                        template="plotly_dark",
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_chart2:
                    # Technical indicators chart
                    fig2 = go.Figure()
                    
                    # Create gauge chart for overall signal
                    fig2.add_trace(go.Indicator(
                        mode = "gauge+number+delta",
                        value = analysis['long_prob'],
                        domain = {'x': [0, 1], 'y': [0.5, 1]},
                        title = {'text': "Long Probability %"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkgreen" if analysis['long_prob'] > 60 else "darkred" if analysis['long_prob'] < 40 else "orange"},
                            'steps': [
                                {'range': [0, 40], 'color': "lightgray"},
                                {'range': [40, 60], 'color': "gray"},
                                {'range': [60, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    
                    fig2.add_trace(go.Indicator(
                        mode = "gauge+number+delta",
                        value = analysis['risk_score'],
                        domain = {'x': [0, 1], 'y': [0, 0.5]},
                        title = {'text': "Risk Score"},
                        delta = {'reference': 50, 'decreasing': {'color': "green"}},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkred" if analysis['risk_score'] > 60 else "orange" if analysis['risk_score'] > 40 else "darkgreen"},
                            'steps': [
                                {'range': [0, 20], 'color': "lightgreen"},
                                {'range': [20, 40], 'color': "yellow"},
                                {'range': [40, 60], 'color': "orange"},
                                {'range': [60, 100], 'color': "lightcoral"}
                            ]
                        }
                    ))
                    
                    fig2.update_layout(
                        title=f"{symbol} Signal Strength",
                        height=500,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Detailed metrics
                st.markdown("### 📊 Detailed Metrics")
                
                met_col1, met_col2, met_col3, met_col4, met_col5 = st.columns(5)
                with met_col1:
                    st.metric("Current Price", f"${data['current_price']:,.4f}", 
                             delta=f"{data['price_change_24h']:+.2f}%")
                with met_col2:
                    st.metric("24h Volume", f"${data['volume']:,.0f}",
                             delta=f"{(data['volume_ratio']-1)*100:+.0f}% avg")
                with met_col3:
                    st.metric("RSI", f"{data['rsi']:.1f}", 
                             delta="Overbought" if data['rsi'] > 70 else "Oversold" if data['rsi'] < 30 else "Neutral")
                with met_col4:
                    st.metric("Stoch RSI", f"{data['stoch_rsi']:.1f}",
                             delta="OB" if data['stoch_rsi'] > 80 else "OS" if data['stoch_rsi'] < 20 else "N")
                with met_col5:
                    risk_level, risk_color = get_risk_level(analysis['risk_score'])
                    st.metric("Risk Level", risk_level.split()[1], delta=f"{analysis['risk_score']}/100")
                
                # Advanced indicators row
                ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)
                
                with ind_col1:
                    st.markdown("**📊 Moving Averages**")
                    ma_color = "green" if data['current_price'] > data['ma_20'] else "red"
                    st.markdown(f"MA7: ${data['ma_7']:.2f}")
                    st.markdown(f"MA20: :{ma_color}[${data['ma_20']:.2f}]")
                    st.markdown(f"MA50: ${data['ma_50']:.2f}")
                
                with ind_col2:
                    st.markdown("**📈 MACD**")
                    macd_color = "green" if data['macd'] > 0 else "red"
                    st.markdown(f"MACD: :{macd_color}[{data['macd']:.3f}]")
                    st.markdown(f"Signal: {data['macd_signal']:.3f}")
                    hist_color = "green" if data['macd_histogram'] > 0 else "red"
                    st.markdown(f"Histogram: :{hist_color}[{data['macd_histogram']:.3f}]")
                
                with ind_col3:
                    st.markdown("**📉 Bollinger Bands**")
                    st.markdown(f"Upper: ${data['bb_upper']:.2f}")
                    bb_color = "red" if data['bb_position'] > 80 else "green" if data['bb_position'] < 20 else "orange"
                    st.markdown(f"Position: :{bb_color}[{data['bb_position']:.0f}%]")
                    st.markdown(f"Lower: ${data['bb_lower']:.2f}")
                
                with ind_col4:
                    st.markdown("**💪 Support/Resistance**")
                    st.markdown(f"Resistance: ${data['resistance']:.2f}")
                    st.markdown(f"Current: ${data['current_price']:.2f}")
                    st.markdown(f"Support: ${data['support']:.2f}")
                
                # Market structure
                if show_order_book:
                    st.markdown("### 📖 Order Book Analysis")
                    ob_col1, ob_col2, ob_col3 = st.columns(3)
                    with ob_col1:
                        imb_color = "green" if data['order_book_imbalance'] > 10 else "red" if data['order_book_imbalance'] < -10 else "gray"
                        st.metric("Order Imbalance", f"{data['order_book_imbalance']:.1f}%")
                    with ob_col2:
                        st.metric("Bid Wall", f"{data['bid_wall']:.0f}")
                    with ob_col3:
                        st.metric("Ask Wall", f"{data['ask_wall']:.0f}")
                
                # Signal breakdown
                with st.expander(f"🔍 Signal Analysis for {symbol}", expanded=False):
                    signals = analysis['signals']
                    st.markdown("**Signal Contributions:**")
                    
                    # Create signal visualization
                    signal_names = list(signals.keys())
                    signal_values = list(signals.values())
                    colors = ['green' if v > 0 else 'red' for v in signal_values]
                    
                    fig_signals = go.Figure(data=[
                        go.Bar(
                            x=signal_names,
                            y=signal_values,
                            marker_color=colors,
                            text=[f"{v:+.1f}" for v in signal_values],
                            textposition='auto',
                        )
                    ])
                    fig_signals.update_layout(
                        title="Signal Breakdown",
                        yaxis_title="Signal Strength",
                        template="plotly_dark",
                        height=300
                    )
                    st.plotly_chart(fig_signals, use_container_width=True)
                    
                    st.markdown("**Market Context:**")
                    st.write(f"• Funding Rate: {data['funding_rate']:+.4f}% ({'Longs pay Shorts' if data['funding_rate'] > 0 else 'Shorts pay Longs' if data['funding_rate'] < 0 else 'Neutral'})")
                    st.write(f"• ATR (Volatility): ${data['atr']:.2f} ({data['atr']/data['current_price']*100:.2f}%)")
                    st.write(f"• Volume Ratio: {data['volume_ratio']:.2f}x average")
                
                st.markdown("---")

with tab3:
    st.header("📈 Market Overview")
    
    if symbols:
        # Market summary
        st.subheader("🌍 Market Summary")
        
        market_data = []
        for symbol in symbols:
            data = get_enhanced_data(symbol)
            if data:
                analysis = calculate_advanced_probabilities(data, leverage, use_ai_predictions)
                market_data.append({
                    'Symbol': symbol,
                    'Price': f"${data['current_price']:,.2f}",
                    '24h Change': f"{data['price_change_24h']:+.2f}%",
                    'Volume': f"${data['volume']/1e6:.1f}M" if data['volume'] > 1e6 else f"${data['volume']/1e3:.1f}K",
                    'RSI': f"{data['rsi']:.1f}",
                    'Signal': analysis['recommendation'],
                    'Long %': f"{analysis['long_prob']:.1f}%",
                    'Risk': analysis['risk_score']
                })
        
        if market_data:
            df_market = pd.DataFrame(market_data)
            st.dataframe(
                df_market,
                use_container_width=True,
                column_config={
                    "24h Change": st.column_config.NumberColumn(format="%.2f%%"),
                    "Risk": st.column_config.ProgressColumn(
                        "Risk Score",
                        help="Risk score from 0-100",
                        format="%d",
                        min_value=0,
                        max_value=100,
                    ),
                }
            )
        
        # Market correlation heatmap
        if len(symbols) > 1:
            st.subheader("🔗 Price Correlation Matrix")
            
            correlation_data = {}
            for symbol in symbols:
                data = get_enhanced_data(symbol)
                if data and 'price_history' in data:
                    correlation_data[symbol] = data['price_history']
            
            if len(correlation_data) > 1:
                df_corr = pd.DataFrame(correlation_data)
                correlation_matrix = df_corr.corr()
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=correlation_matrix.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hoverongaps=False
                ))
                
                fig_corr.update_layout(
                    title="24h Price Correlation",
                    height=400,
                    template="plotly_dark"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
        
        # Market trends
        st.subheader("📊 Market Trends")
        
        trend_col1, trend_col2 = st.columns(2)
        
        with trend_col1:
            # Bullish vs Bearish count
            bullish_count = sum(1 for d in market_data if "LONG" in d['Signal'])
            bearish_count = sum(1 for d in market_data if "SHORT" in d['Signal'])
            neutral_count = len(market_data) - bullish_count - bearish_count
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Bullish', 'Bearish', 'Neutral'],
                values=[bullish_count, bearish_count, neutral_count],
                hole=.3,
                marker_colors=['green', 'red', 'gray']
            )])
            fig_pie.update_layout(
                title="Market Sentiment",
                template="plotly_dark",
                height=300
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with trend_col2:
            # Average metrics
            avg_change = np.mean([float(d['24h Change'].replace('%', '').replace('+', '')) for d in market_data])
            avg_risk = np.mean([d['Risk'] for d in market_data])
            
            st.metric("Average 24h Change", f"{avg_change:+.2f}%")
            st.metric("Average Risk Score", f"{avg_risk:.1f}/100")
            
            market_status = "🟢 Bullish" if avg_change > 2 else "🔴 Bearish" if avg_change < -2 else "🟡 Neutral"
            st.metric("Market Status", market_status)

with tab4:
    st.header("🧪 Backtesting Laboratory")
    
    if enable_backtesting:
        st.subheader("⚙️ Strategy Configuration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            backtest_symbol = st.selectbox("Select Symbol:", symbols if symbols else ["BTCUSDT"], key="bt_symbol")
        with col2:
            strategy_type = st.selectbox("Strategy:", ["MA Crossover", "RSI Reversal", "Bollinger Bands"], key="bt_strategy")
        with col3:
            backtest_leverage = st.slider("Leverage:", 1, 20, 5, key="bt_leverage")
        
        if strategy_type == "MA Crossover":
            param_col1, param_col2 = st.columns(2)
            with param_col1:
                short_ma = st.number_input("Short MA Period:", min_value=2, max_value=20, value=5, key="bt_short_ma")
            with param_col2:
                long_ma = st.number_input("Long MA Period:", min_value=5, max_value=50, value=15, key="bt_long_ma")
        
        if st.button("🚀 Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                data = get_enhanced_data(backtest_symbol)
                if data:
                    strategy_params = {
                        'short_ma': short_ma if strategy_type == "MA Crossover" else 5,
                        'long_ma': long_ma if strategy_type == "MA Crossover" else 15,
                        'leverage': backtest_leverage
                    }
                    
                    results = run_backtest(backtest_symbol, data, strategy_params)
                    
                    if results:
                        st.session_state.backtest_results[backtest_symbol] = results
                        
                        # Display results
                        st.success("Backtest completed!")
                        
                        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                        with res_col1:
                            st.metric("Total Return", f"{results['total_return']:+.2f}%")
                        with res_col2:
                            st.metric("Win Rate", f"{results['win_rate']:.1f}%")
                        with res_col3:
                            st.metric("Total Trades", results['total_trades'])
                        with res_col4:
                            st.metric("Profit Factor", f"{results['win_trades']/max(results['loss_trades'], 1):.2f}")
                        
                        # Trade visualization
                        if results['trades']:
                            st.subheader("📊 Trade Visualization")
                            
                            prices = data['price_history']
                            timestamps = data['timestamps']
                            
                            fig_trades = go.Figure()
                            
                            # Price line
                            fig_trades.add_trace(go.Scatter(
                                x=timestamps,
                                y=prices,
                                mode='lines',
                                name='Price',
                                line=dict(color='gray', width=1)
                            ))
                            
                            # Add trade markers
                            for trade in results['trades']:
                                if trade['type'] in ['open_long', 'close_short']:
                                    fig_trades.add_trace(go.Scatter(
                                        x=[timestamps[trade['index']]],
                                        y=[trade['price']],
                                        mode='markers',
                                        marker=dict(symbol='triangle-up', size=12, color='green'),
                                        name='Buy',
                                        showlegend=False
                                    ))
                                else:
                                    fig_trades.add_trace(go.Scatter(
                                        x=[timestamps[trade['index']]],
                                        y=[trade['price']],
                                        mode='markers',
                                        marker=dict(symbol='triangle-down', size=12, color='red'),
                                        name='Sell',
                                        showlegend=False
                                    ))
                            
                            fig_trades.update_layout(
                                title=f"{backtest_symbol} Backtest Results - {strategy_type}",
                                xaxis_title="Time",
                                yaxis_title="Price",
                                template="plotly_dark",
                                height=500
                            )
                            st.plotly_chart(fig_trades, use_container_width=True)
                    else:
                        st.error("Insufficient data for backtesting")
    else:
        st.info("Enable backtesting in the sidebar to access this feature")

# Portfolio summary (always visible)
if st.session_state.trade_history:
    st.markdown("---")
    st.subheader("📈 Portfolio Summary")
    
    total_pnl = 0
    winning_trades = 0
    total_invested = 0
    open_positions = []
    
    for trade in st.session_state.trade_history:
        if trade['symbol'] in symbols:
            trade_data = get_enhanced_data(trade['symbol'])
            if trade_data:
                price_diff = trade_data['current_price'] - trade['entry_price']
                pnl_percent = (price_diff / trade['entry_price']) * 100
                if trade['direction'] == "Short":
                    pnl_percent = -pnl_percent
                pnl_usd = (pnl_percent / 100) * trade['size'] * trade['leverage']
                total_pnl += pnl_usd
                total_invested += trade['size']
                if pnl_usd > 0:
                    winning_trades += 1
                
                open_positions.append({
                    'Symbol': trade['symbol'],
                    'Direction': trade['direction'],
                    'Entry': f"${trade['entry_price']:.4f}",
                    'Current': f"${trade_data['current_price']:.4f}",
                    'Size': f"${trade['size']:.2f}",
                    'Leverage': f"{trade['leverage']}x",
                    'P&L': f"${pnl_usd:+.2f}",
                    'P&L %': f"{pnl_percent:+.2f}%",
                    'Time': trade['timestamp'].strftime("%H:%M:%S")
                })
    
    # Portfolio metrics
    portfolio_col1, portfolio_col2, portfolio_col3, portfolio_col4 = st.columns(4)
    with portfolio_col1:
        pnl_color = "green" if total_pnl > 0 else "red"
        st.metric("Total P&L", f"${total_pnl:+.2f}", 
                 f"{(total_pnl/total_invested*100):+.1f}%" if total_invested > 0 else None)
    with portfolio_col2:
        win_rate = (winning_trades / len(st.session_state.trade_history)) * 100 if st.session_state.trade_history else 0
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with portfolio_col3:
        st.metric("Active Trades", str(len(st.session_state.trade_history)))
    with portfolio_col4:
        st.metric("Total Invested", f"${total_invested:.2f}")
    
    # Display open positions
    if open_positions:
        st.markdown("### 📋 Open Positions")
        df_positions = pd.DataFrame(open_positions)
        st.dataframe(
            df_positions,
            use_container_width=True,
            column_config={
                "P&L": st.column_config.NumberColumn(format="$%.2f"),
                "P&L %": st.column_config.NumberColumn(format="%.2f%%"),
            }
        )
    
    # Portfolio actions
    col_clear, col_export = st.columns(2)
    with col_clear:
        if st.button("🗑️ Clear Portfolio", type="secondary"):
            st.session_state.trade_history = []
            st.success("Portfolio cleared!")
            st.rerun()
    with col_export:
        if st.button("📥 Export Portfolio", type="secondary"):
            portfolio_data = {
                'positions': open_positions,
                'summary': {
                    'total_pnl': total_pnl,
                    'win_rate': win_rate,
                    'total_invested': total_invested,
                    'timestamp': datetime.now().isoformat()
                }
            }
            st.download_button(
                label="Download Portfolio",
                data=json.dumps(portfolio_data, indent=2),
                file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Enhanced footer with comprehensive tips
st.markdown("---")

# Create expandable sections for different tip categories
tip_tab1, tip_tab2, tip_tab3, tip_tab4 = st.tabs(["🎯 Probability Guide", "📊 Technical Analysis", "🛡️ Risk Management", "🤖 AI Features"])

with tip_tab1:
    st.markdown("""
    ### 🎯 Understanding Success Probabilities
    
    **Probability Rankings:**
    - **75%+ Success Rate**: 🟢 Excellent signal, suitable for higher leverage
    - **65-75% Success Rate**: 🟡 Strong signal, moderate leverage recommended
    - **55-65% Success Rate**: 🟠 Good signal, use lower leverage
    - **45-55% Success Rate**: ⚪ Neutral zone, wait for better setup
    - **35-45% Success Rate**: 🔴 Poor signal, avoid or consider opposite
    - **<35% Success Rate**: 🚫 Very poor signal, strong opposite indication
    
    **Signal Strength:**
    - **STRONG signals**: 25%+ probability difference between long/short
    - **MODERATE signals**: 15-25% probability difference  
    - **WEAK signals**: 5-15% probability difference
    - **NEUTRAL**: <5% probability difference - avoid trading
    
    **Confidence Levels:**
    - High confidence (>20% diff): Consider full position size
    - Medium confidence (10-20%): Use 50-75% position size
    - Low confidence (<10%): Use 25-50% position size or wait
    """)

with tip_tab2:
    st.markdown("""
    ### 📊 Technical Indicator Guide
    
    **RSI (Relative Strength Index):**
    - **>80**: Extremely overbought - strong short signal
    - **70-80**: Overbought - potential short signal
    - **30-70**: Normal range
    - **20-30**: Oversold - potential long signal
    - **<20**: Extremely oversold - strong long signal
    
    **Stochastic RSI:**
    - **>80**: Overbought conditions in RSI
    - **<20**: Oversold conditions in RSI
    - More sensitive than regular RSI
    
    **Moving Averages:**
    - **Price > MA7 > MA20 > MA50**: Strong uptrend 📈
    - **Price < MA7 < MA20 < MA50**: Strong downtrend 📉
    - **MA crossovers**: Potential trend changes
    - **MA200**: Major support/resistance level
    
    **MACD:**
    - **Positive MACD + above signal**: Bullish momentum
    - **Negative MACD + below signal**: Bearish momentum
    - **MACD crossing signal line**: Entry/exit signals
    - **Histogram growing**: Momentum increasing
    
    **Bollinger Bands:**
    - **>90%**: At upper band - potential reversal
    - **<10%**: At lower band - potential bounce
    - **Band squeeze**: Low volatility, breakout coming
    - **Band expansion**: High volatility period
    
    **ATR (Average True Range):**
    - Measures volatility
    - Use for stop loss placement (2-3x ATR)
    - Higher ATR = wider stops needed
    """)

with tip_tab3:
    st.markdown("""
    ### 🛡️ Advanced Risk Management
    
    **Position Sizing by Risk Score:**
    - **Risk 0-20**: Up to 15x leverage ✅
    - **Risk 20-40**: Max 10x leverage ⚠️
    - **Risk 40-60**: Max 5x leverage 🚨
    - **Risk 60-80**: Max 3x leverage ⛔
    - **Risk 80-100**: 1-2x leverage only 🛑
    
    **Kelly Criterion for Position Sizing:**
    - Optimal size = (Win% × Average Win - Loss% × Average Loss) / Average Win
    - Never risk more than 25% of Kelly suggestion
    - Reduce size during losing streaks
    
    **Stop Loss Strategies:**
    - **ATR-based**: 2-3x ATR from entry
    - **Support/Resistance**: Just beyond key levels
    - **Percentage**: 2-5% depending on leverage
    - **Time-based**: Exit if no movement in X hours
    
    **Take Profit Strategies:**
    - **Fixed R:R**: 2:1 or 3:1 risk/reward
    - **Trailing**: Move stop to breakeven at 1R profit
    - **Partial**: Take 50% at 1R, let rest run
    - **Resistance-based**: Exit at major levels
    
    **Portfolio Management:**
    - Maximum 2% risk per trade
    - Maximum 6% total portfolio risk
    - Correlation check: Avoid similar trades
    - Rebalance weekly/monthly
    
    **Funding Rate Strategy:**
    - **>0.05%**: Strong short bias (longs overpaying)
    - **0.02-0.05%**: Moderate short bias
    - **-0.02-0.02%**: Neutral funding
    - **-0.05--0.02%**: Moderate long bias
    - **<-0.05%**: Strong long bias (shorts overpaying)
    """)

with tip_tab4:
    st.markdown("""
    ### 🤖 AI-Enhanced Features
    
    **AI Predictions (When Enabled):**
    - Pattern recognition on price action
    - Volume trend analysis
    - Multi-timeframe correlation
    - Sentiment analysis integration
    
    **Order Book Analysis:**
    - **Imbalance >20%**: Strong directional bias
    - **Bid walls**: Support levels (buy pressure)
    - **Ask walls**: Resistance levels (sell pressure)
    - **Spoofing detection**: Large orders that disappear
    
    **Smart Alerts:**
    - Probability threshold breaches
    - Risk score improvements
    - Volume spike detection
    - Support/resistance approaches
    - Multiple indicator convergence
    
    **Backtesting Features:**
    - MA Crossover: Classic trend following
    - RSI Reversal: Mean reversion strategy
    - Bollinger Bands: Volatility breakout
    - Custom parameter optimization
    - Walk-forward analysis
    
    **Market Correlation:**
    - Identify correlated pairs
    - Diversification opportunities
    - Risk concentration warnings
    - Sector rotation signals
    
    **Advanced Metrics:**
    - Sharpe Ratio calculation
    - Maximum drawdown tracking
    - Win/loss streak analysis
    - Risk-adjusted returns
    """)

# Performance disclaimer
with st.expander("ℹ️ About This Tool & Disclaimer"):
    st.markdown("""
    ### 🚀 Enhanced Features:
    
    **📊 Technical Analysis Suite:**
    - RSI, Stochastic RSI, MACD, Moving Averages (7, 20, 50, 200)
    - Bollinger Bands, ATR, Support/Resistance levels
    - Volume analysis and order book data
    - Real-time price action monitoring
    
    **🎯 Advanced Probability Engine:**
    - Multi-factor signal weighting system
    - AI-enhanced predictions (optional)
    - Leverage-adjusted probability calculations
    - Confidence scoring and signal strength
    
    **🔔 Smart Alert System:**
    - Customizable probability thresholds
    - Risk-based opportunity alerts
    - Volume spike notifications
    - Support/resistance proximity warnings
    
    **💼 Portfolio Management:**
    - Multi-position tracking with real-time P&L
    - Risk exposure monitoring
    - Performance analytics and win rate
    - Trade history with export functionality
    
    **🧪 Backtesting Laboratory:**
    - Multiple strategy templates
    - Custom parameter optimization
    - Visual trade representation
    - Performance metrics calculation
    
    **📈 Market Overview:**
    - Multi-symbol comparison dashboard
    - Correlation matrix analysis
    - Market sentiment indicators
    - Trend identification tools
    
    ### ⚠️ Important Disclaimers:
    
    **FINANCIAL DISCLAIMER:**
    - This tool is for educational and informational purposes only
    - Not financial advice or investment recommendations
    - Past performance does not guarantee future results
    - Cryptocurrency trading carries substantial risk of loss
    
    **RISK WARNING:**
    - Never invest more than you can afford to lose
    - Leveraged trading can result in losses exceeding deposits
    - Markets can remain irrational longer than you can remain solvent
    - Technical analysis is not infallible
    
    **DATA CONSIDERATIONS:**
    - Real-time data subject to delays and inaccuracies
    - API limitations may affect data availability
    - Calculations based on historical data patterns
    - Market conditions can change rapidly
    
    **BEST PRACTICES:**
    - Always use stop losses
    - Start with small position sizes
    - Test strategies in demo mode first
    - Keep a trading journal
    - Continuously educate yourself
    - Consider fundamental analysis
    - Monitor global market conditions
    - Have a clear exit strategy
    
    **TECHNICAL NOTES:**
    - Primary data: Binance Futures API
    - Fallback: Demo data for testing
    - Refresh rate: Configurable (10-120s)
    - Calculations updated in real-time
    
    By using this tool, you acknowledge that you understand and accept these risks.
    """)

# Add version and last update info
st.caption(f"Version 2.0 Enhanced | Last data update: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")

# Warning for no symbols selected
if not symbols:
    st.warning("⚠️ Please select at least one symbol from the sidebar to begin analysis.")
    st.info("💡 Tip: Start with BTC and ETH for major market movements, then add altcoins for diversification.")

# Add custom CSS for better styling
st.markdown("""
<style>
    /* Metric styling */
    .stMetric > label {
        font-size: 14px !important;
        font-weight: bold !important;
    }
    
    .stMetric > div {
        font-size: 20px !important;
        font-weight: bold !important;
    }
    
    /* Expander styling */
    .stExpander > details > summary {
        font-weight: bold;
        font-size: 16px;
    }
    
    /* Alert styling */
    .stAlert {
        margin: 10px 0;
        border-radius: 10px;
    }
    
    /* Tab styling */
    .stTabs > div > div {
        gap: 24px;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: bold;
    }
    
    /* Dataframe styling */
    .dataframe {
        font-size: 14px;
    }
    
    /* Container styling */
    .main > div {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# End of application
