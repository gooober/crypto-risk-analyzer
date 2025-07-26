import streamlit as st
import requests
import time
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Perpetual Crypto Trade Risk Analyzer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Perpetual Crypto Trade Risk Analyzer")

# Initialize session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    symbols = st.multiselect(
        "Select symbols to analyze:", 
        ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT"], 
        default=["BTCUSDT", "ETHUSDT"]
    )
    leverage = st.slider("Set your leverage (X):", 1, 50, 10)
    
    st.header("🔄 Auto-Refresh Settings")
    auto_refresh = st.checkbox("Enable Auto-Refresh", value=True)
    refresh_rate = st.slider("Refresh every (seconds):", 10, 120, 30)
    
    if st.button("🔄 Manual Refresh"):
        st.session_state.last_update = datetime.now() - timedelta(seconds=refresh_rate)
        st.rerun()

# Function to fetch Binance Futures data with better error handling
@st.cache_data(ttl=10)  # Cache for 10 seconds to avoid rate limiting
def get_futures_data(symbol):
    """Fetch futures data from Binance API with error handling"""
    base_url = "https://fapi.binance.com"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; TradingApp/1.0)"}
    
    try:
        # Funding rate
        funding_response = requests.get(
            f"{base_url}/fapi/v1/fundingRate", 
            params={"symbol": symbol, "limit": 1}, 
            headers=headers,
            timeout=5
        )
        funding_response.raise_for_status()
        funding = funding_response.json()
        funding_rate = float(funding[0]['fundingRate']) * 100 if funding else 0.0

        # 24h stats
        stats_response = requests.get(
            f"{base_url}/fapi/v1/ticker/24hr", 
            params={"symbol": symbol}, 
            headers=headers,
            timeout=5
        )
        stats_response.raise_for_status()
        stats = stats_response.json()
        price_change_percent = float(stats.get("priceChangePercent", 0))
        current_price = float(stats.get("lastPrice", 0))
        volume = float(stats.get("volume", 0))

        # Order book depth for market sentiment
        depth_response = requests.get(
            f"{base_url}/fapi/v1/depth", 
            params={"symbol": symbol, "limit": 5}, 
            headers=headers,
            timeout=5
        )
        depth_response.raise_for_status()
        depth = depth_response.json()
        
        bid_qty = sum([float(bid[1]) for bid in depth.get('bids', [])])
        ask_qty = sum([float(ask[1]) for ask in depth.get('asks', [])])
        long_short_ratio = bid_qty / (ask_qty + 1e-9)

        return {
            "funding_rate": funding_rate,
            "volatility": abs(price_change_percent),
            "price_change_24h": price_change_percent,
            "current_price": current_price,
            "volume": volume,
            "long_short_ratio": long_short_ratio,
            "last_updated": datetime.now().strftime("%H:%M:%S")
        }
    
    except requests.exceptions.RequestException as e:
        st.error(f"Network error for {symbol}: {str(e)}")
        return None
    except (KeyError, ValueError, IndexError) as e:
        st.error(f"Data parsing error for {symbol}: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error for {symbol}: {str(e)}")
        return None

# Function to calculate risk score
def calculate_risk_score(data, leverage):
    """Calculate a comprehensive risk score"""
    if not data:
        return 0
    
    # Base risk from volatility and leverage
    volatility_risk = data['volatility'] * leverage * 0.1
    
    # Funding rate risk (high funding = more risk)
    funding_risk = abs(data['funding_rate']) * 10
    
    # Market sentiment risk (extreme ratios = more risk)
    sentiment_risk = abs(data['long_short_ratio'] - 1) * 5
    
    # Volume risk (low volume = higher risk)
    volume_risk = max(0, (10000000 - data['volume']) / 1000000) * 2
    
    total_risk = volatility_risk + funding_risk + sentiment_risk + volume_risk
    return min(round(total_risk, 1), 100)  # Cap at 100

# Function to get risk level and color
def get_risk_level(score):
    """Return risk level and corresponding color"""
    if score < 20:
        return "🟢 LOW", "green"
    elif score < 40:
        return "🟡 MEDIUM", "orange"
    elif score < 60:
        return "🟠 HIGH", "red"
    else:
        return "🔴 EXTREME", "darkred"

# Trade analysis section
st.header("💼 Your Trade Analysis")
with st.expander("📝 Enter Your Active Trade", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_symbol = st.selectbox("Trade Symbol:", symbols if symbols else ["BTCUSDT"])
    with col2:
        entry_price = st.number_input("Entry Price:", min_value=0.0, format="%.4f", value=0.0)
    with col3:
        trade_direction = st.radio("Direction:", ["Long", "Short"])
    with col4:
        position_size = st.number_input("Position Size (USDT):", min_value=0.0, value=100.0)

# Auto-refresh logic
if auto_refresh:
    current_time = datetime.now()
    time_diff = (current_time - st.session_state.last_update).total_seconds()
    
    if time_diff >= refresh_rate:
        st.session_state.last_update = current_time
        st.rerun()

# Main dashboard
if symbols:
    st.header("📊 Market Analysis Dashboard")
    
    # Display last update time
    st.caption(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')} | Next refresh in: {max(0, refresh_rate - int((datetime.now() - st.session_state.last_update).total_seconds()))}s")
    
    # Create columns for better layout
    cols = st.columns(min(len(symbols), 3))
    
    for idx, symbol in enumerate(symbols):
        with cols[idx % len(cols)]:
            with st.container():
                data = get_futures_data(symbol)
                
                if data:
                    # Calculate risk score
                    risk_score = calculate_risk_score(data, leverage)
                    risk_level, risk_color = get_risk_level(risk_score)
                    
                    # Display symbol header
                    st.subheader(f"🔸 {symbol}")
                    
                    # Price metrics
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric(
                            "Current Price", 
                            f"${data['current_price']:,.4f}",
                            delta=f"{data['price_change_24h']:+.2f}%"
                        )
                    with col_b:
                        st.metric("24h Volume", f"${data['volume']:,.0f}")
                    
                    # Risk assessment
                    st.markdown(f"**Risk Level:** :{risk_color}[{risk_level}] (Score: {risk_score}/100)")
                    
                    # Detailed metrics
                    st.markdown("**Market Details:**")
                    st.write(f"• Funding Rate: {data['funding_rate']:+.4f}%")
                    st.write(f"• Volatility (24h): {data['volatility']:.2f}%")
                    st.write(f"• Market Sentiment: {'Bullish 📈' if data['long_short_ratio'] > 1.2 else 'Bearish 📉' if data['long_short_ratio'] < 0.8 else 'Neutral ⚖️'}")
                    st.write(f"• Long/Short Ratio: {data['long_short_ratio']:.2f}")
                    
                    # Trade PnL calculation if applicable
                    if entry_price > 0 and symbol == selected_symbol:
                        price_diff = data['current_price'] - entry_price
                        pnl_percent = (price_diff / entry_price) * 100
                        
                        if trade_direction == "Short":
                            pnl_percent = -pnl_percent
                        
                        pnl_usd = (pnl_percent / 100) * position_size * leverage
                        
                        st.markdown("---")
                        st.markdown("**Your Trade Performance:**")
                        col_x, col_y = st.columns(2)
                        with col_x:
                            st.metric("Unrealized PnL", f"${pnl_usd:+.2f}", f"{pnl_percent:+.2f}%")
                        with col_y:
                            liquidation_price = entry_price * (1 - (1/leverage)) if trade_direction == "Long" else entry_price * (1 + (1/leverage))
                            st.metric("Est. Liquidation", f"${liquidation_price:.4f}")
                    
                    st.markdown("---")
                else:
                    st.error(f"❌ Failed to load data for {symbol}")
                    st.markdown("---")

else:
    st.warning("⚠️ Please select at least one symbol to analyze.")

# Footer with tips
st.markdown("---")
st.markdown("""
### 💡 Trading Tips:
- **Risk Score < 20**: Generally safer for higher leverage
- **Risk Score 20-40**: Moderate risk, consider lower leverage  
- **Risk Score 40-60**: High risk, use minimal leverage
- **Risk Score > 60**: Extreme risk, avoid or use very low leverage
- **Funding Rate**: Positive = Longs pay Shorts, Negative = Shorts pay Longs
- Always use proper risk management and never risk more than you can afford to lose!
""")

# Performance info
with st.expander("ℹ️ About This Tool"):
    st.markdown("""
    This tool analyzes perpetual futures trading risks by combining:
    - Real-time price volatility
    - Current funding rates  
    - Market sentiment (order book analysis)
    - Trading volume
    - Your leverage multiplier
    
    **Data Source:** Binance Futures API  
    **Update Frequency:** Configurable (10-120 seconds)  
    **Risk Algorithm:** Proprietary multi-factor model
    """)
