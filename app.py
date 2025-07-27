import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Page config
st.set_page_config(
    page_title="Crypto Trade Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Crypto Trade Analyzer")

# Initialize session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    symbols = st.multiselect(
        "Select Cryptocurrencies",
        ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"],
        default=["BTCUSDT", "ETHUSDT"]
    )
    
    leverage = st.slider("Leverage", 1, 20, 5)
    refresh_rate = st.slider("Refresh Rate (seconds)", 10, 60, 30)
    
    if st.button("🔄 Refresh Now"):
        st.session_state.last_update = datetime.now()
        st.rerun()

# Function to get crypto data
@st.cache_data(ttl=30)
def get_crypto_data(symbol):
    try:
        # Binance API
        url = f"https://fapi.binance.com/fapi/v1/ticker/24hr"
        params = {"symbol": symbol}
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "symbol": symbol,
                "price": float(data.get("lastPrice", 0)),
                "change_24h": float(data.get("priceChangePercent", 0)),
                "volume": float(data.get("volume", 0)),
                "high_24h": float(data.get("highPrice", 0)),
                "low_24h": float(data.get("lowPrice", 0))
            }
    except:
        pass
    
    # Return demo data if API fails
    return {
        "symbol": symbol,
        "price": 50000 if "BTC" in symbol else 3000,
        "change_24h": np.random.uniform(-5, 5),
        "volume": 1000000,
        "high_24h": 51000 if "BTC" in symbol else 3100,
        "low_24h": 49000 if "BTC" in symbol else 2900
    }

# Function to calculate trading signals
def calculate_signals(data, leverage):
    price = data["price"]
    change = data["change_24h"]
    
    # Simple signal calculation
    base_score = 50
    
    # Price momentum
    if change > 3:
        base_score += 15
    elif change > 1:
        base_score += 10
    elif change < -3:
        base_score -= 15
    elif change < -1:
        base_score -= 10
    
    # Risk calculation
    risk = min(100, abs(change) * leverage * 2)
    
    # Probabilities
    long_prob = max(10, min(90, base_score))
    short_prob = max(10, min(90, 100 - base_score))
    
    # Recommendation
    if long_prob > 65:
        recommendation = "STRONG BUY 🚀"
    elif long_prob > 55:
        recommendation = "BUY 📈"
    elif short_prob > 65:
        recommendation = "STRONG SELL 📉"
    elif short_prob > 55:
        recommendation = "SELL 📉"
    else:
        recommendation = "NEUTRAL ⚖️"
    
    return {
        "long_prob": long_prob,
        "short_prob": short_prob,
        "risk": risk,
        "recommendation": recommendation
    }

# Main app
st.header("📊 Market Overview")

if not symbols:
    st.warning("Please select at least one cryptocurrency from the sidebar")
else:
    # Auto refresh
    current_time = datetime.now()
    time_since_update = (current_time - st.session_state.last_update).seconds
    
    if time_since_update >= refresh_rate:
        st.session_state.last_update = current_time
        st.rerun()
    
    # Time until next refresh
    time_until_refresh = refresh_rate - time_since_update
    st.caption(f"Auto-refresh in {time_until_refresh} seconds")
    
    # Display crypto cards
    cols = st.columns(len(symbols) if len(symbols) <= 3 else 3)
    
    for idx, symbol in enumerate(symbols):
        col_idx = idx % len(cols)
        
        with cols[col_idx]:
            # Get data
            data = get_crypto_data(symbol)
            signals = calculate_signals(data, leverage)
            
            # Display card
            st.subheader(f"{symbol.replace('USDT', '')}")
            
            # Price info
            price_delta = f"{data['change_24h']:+.2f}%"
            st.metric("Price", f"${data['price']:,.2f}", delta=price_delta)
            
            # Volume
            st.metric("24h Volume", f"${data['volume']:,.0f}")
            
            # Trading signals
            st.write("**Trading Signals**")
            
            # Progress bars for probabilities
            st.write(f"Long: {signals['long_prob']}%")
            st.progress(signals['long_prob'] / 100)
            
            st.write(f"Short: {signals['short_prob']}%")
            st.progress(signals['short_prob'] / 100)
            
            # Risk level
            risk_color = "🟢" if signals['risk'] < 30 else "🟡" if signals['risk'] < 60 else "🔴"
            st.write(f"**Risk Level**: {risk_color} {signals['risk']:.0f}%")
            
            # Recommendation
            st.info(f"**Signal**: {signals['recommendation']}")
            
            st.divider()

# Chart section
st.header("📈 Price Charts")

if symbols:
    selected_symbol = st.selectbox("Select symbol for chart", symbols)
    
    # Create sample data for chart
    hours = 24
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(hours, 0, -1)]
    
    # Get current price
    current_data = get_crypto_data(selected_symbol)
    base_price = current_data['price']
    
    # Generate price history
    price_history = []
    for i in range(hours):
        variation = np.random.uniform(-2, 2) / 100
        price = base_price * (1 + variation)
        price_history.append(price)
    
    # Create chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=price_history,
        mode='lines',
        name='Price',
        line=dict(color='cyan', width=2)
    ))
    
    fig.update_layout(
        title=f"{selected_symbol} Price Chart (24h)",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Tips section
with st.expander("💡 Trading Tips"):
    st.markdown("""
    ### Risk Management Guidelines
    
    - **Never risk more than 2% per trade**
    - **Use stop losses always**
    - **Don't overleverage**
    - **Diversify your portfolio**
    
    ### Signal Interpretation
    
    - **70%+ Probability**: Strong signal
    - **60-70% Probability**: Moderate signal
    - **50-60% Probability**: Weak signal
    - **Below 50%**: Consider opposite direction
    
    ### Risk Levels
    
    - 🟢 **Low Risk (0-30%)**: Safe to trade
    - 🟡 **Medium Risk (30-60%)**: Trade with caution
    - 🔴 **High Risk (60%+)**: Reduce position size
    """)

# Disclaimer
st.caption("⚠️ This tool is for educational purposes only. Not financial advice.")
