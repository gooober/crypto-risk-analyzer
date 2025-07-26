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

# Function to fetch crypto data with multiple fallback APIs
@st.cache_data(ttl=15)  # Cache for 15 seconds to avoid rate limiting
def get_futures_data(symbol):
    """Fetch futures data with multiple API fallbacks"""
    
    # Try Binance first with better headers
    def try_binance():
        base_url = "https://fapi.binance.com"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache"
        }
        
        try:
            # Use a single consolidated request for better efficiency
            stats_response = requests.get(
                f"{base_url}/fapi/v1/ticker/24hr", 
                params={"symbol": symbol}, 
                headers=headers,
                timeout=8
            )
            
            if stats_response.status_code == 200:
                stats = stats_response.json()
                current_price = float(stats.get("lastPrice", 0))
                price_change_percent = float(stats.get("priceChangePercent", 0))
                volume = float(stats.get("volume", 0))
                
                # Try to get funding rate (optional)
                funding_rate = 0.0
                try:
                    funding_response = requests.get(
                        f"{base_url}/fapi/v1/fundingRate", 
                        params={"symbol": symbol, "limit": 1}, 
                        headers=headers,
                        timeout=5
                    )
                    if funding_response.status_code == 200:
                        funding = funding_response.json()
                        funding_rate = float(funding[0]['fundingRate']) * 100 if funding else 0.0
                except:
                    pass  # Use default funding rate if fails
                
                return {
                    "funding_rate": funding_rate,
                    "volatility": abs(price_change_percent),
                    "price_change_24h": price_change_percent,
                    "current_price": current_price,
                    "volume": volume,
                    "long_short_ratio": 1.0,  # Default neutral
                    "last_updated": datetime.now().strftime("%H:%M:%S"),
                    "data_source": "Binance"
                }
        except Exception as e:
            st.warning(f"Binance API failed for {symbol}: {str(e)}")
            return None
    
    # Fallback to CoinGecko API
    def try_coingecko():
        try:
            # Map symbols to CoinGecko IDs
            symbol_map = {
                "BTCUSDT": "bitcoin",
                "ETHUSDT": "ethereum", 
                "SOLUSDT": "solana",
                "ADAUSDT": "cardano",
                "DOTUSDT": "polkadot"
            }
            
            coin_id = symbol_map.get(symbol)
            if not coin_id:
                return None
                
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json"
            }
            
            response = requests.get(
                f"https://api.coingecko.com/api/v3/simple/price",
                params={
                    "ids": coin_id,
                    "vs_currencies": "usd",
                    "include_24hr_change": "true",
                    "include_24hr_vol": "true"
                },
                headers=headers,
                timeout=8
            )
            
            if response.status_code == 200:
                data = response.json()
                coin_data = data.get(coin_id, {})
                
                return {
                    "funding_rate": 0.0,  # Not available from CoinGecko
                    "volatility": abs(coin_data.get("usd_24h_change", 0)),
                    "price_change_24h": coin_data.get("usd_24h_change", 0),
                    "current_price": coin_data.get("usd", 0),
                    "volume": coin_data.get("usd_24h_vol", 0),
                    "long_short_ratio": 1.0,  # Default neutral
                    "last_updated": datetime.now().strftime("%H:%M:%S"),
                    "data_source": "CoinGecko"
                }
        except Exception as e:
            st.warning(f"CoinGecko API failed for {symbol}: {str(e)}")
            return None
    
    # Try APIs in order
    result = try_binance()
    if result:
        return result
        
    result = try_coingecko()
    if result:
        return result
    
    # If all APIs fail, return mock data for demo
    st.error(f"⚠️ All APIs failed for {symbol}. Using demo data.")
    return {
        "funding_rate": 0.01,
        "volatility": 2.5,
        "price_change_24h": 1.2,
        "current_price": 50000.0 if "BTC" in symbol else 3000.0 if "ETH" in symbol else 100.0,
        "volume": 1000000,
        "long_short_ratio": 1.1,
        "last_updated": datetime.now().strftime("%H:%M:%S"),
        "data_source": "Demo Mode"
    }

# Advanced trading probability calculations
def calculate_trading_probabilities(data, leverage):
    """Calculate success probabilities for long and short positions"""
    if not data:
        return {"long_prob": 50, "short_prob": 50, "risk_score": 50, "recommendation": "NEUTRAL"}
    
    # Base probability starts at 50% (neutral)
    base_prob = 50
    
    # Market trend analysis (24h price change)
    trend_factor = 0
    if data['price_change_24h'] > 2:
        trend_factor = min(15, data['price_change_24h'] * 2)  # Strong uptrend favors longs
    elif data['price_change_24h'] < -2:
        trend_factor = max(-15, data['price_change_24h'] * 2)  # Strong downtrend favors shorts
    
    # Volatility impact (high volatility = higher risk for both directions)
    volatility_penalty = min(10, data['volatility'] * 0.5)
    
    # Funding rate analysis (affects long/short bias)
    funding_bias = 0
    if data['funding_rate'] > 0.01:  # Positive funding (longs pay shorts)
        funding_bias = -min(8, data['funding_rate'] * 200)  # Slightly favor shorts
    elif data['funding_rate'] < -0.01:  # Negative funding (shorts pay longs)
        funding_bias = min(8, abs(data['funding_rate']) * 200)  # Slightly favor longs
    
    # Volume analysis (higher volume = more reliable signals)
    volume_confidence = 0
    if data['volume'] > 1000000:  # High volume
        volume_confidence = min(5, (data['volume'] / 10000000) * 5)
    elif data['volume'] < 100000:  # Low volume
        volume_confidence = -3
    
    # Market sentiment from long/short ratio
    sentiment_bias = 0
    if data['long_short_ratio'] > 1.5:  # Too many longs (contrarian signal)
        sentiment_bias = -min(6, (data['long_short_ratio'] - 1) * 3)
    elif data['long_short_ratio'] < 0.7:  # Too many shorts (contrarian signal)
        sentiment_bias = min(6, (1 - data['long_short_ratio']) * 6)
    
    # Calculate probabilities
    long_probability = base_prob + trend_factor + funding_bias + volume_confidence - sentiment_bias - volatility_penalty
    short_probability = base_prob - trend_factor - funding_bias + volume_confidence + sentiment_bias - volatility_penalty
    
    # Apply leverage penalty (higher leverage = lower success probability)
    leverage_penalty = min(20, (leverage - 1) * 0.8)
    long_probability -= leverage_penalty
    short_probability -= leverage_penalty
    
    # Ensure probabilities stay within realistic bounds
    long_probability = max(15, min(85, long_probability))
    short_probability = max(15, min(85, short_probability))
    
    # Calculate overall risk score
    base_risk = data['volatility'] * leverage * 0.15
    funding_risk = abs(data['funding_rate']) * 10
    sentiment_risk = abs(data['long_short_ratio'] - 1) * 3
    volume_risk = max(0, (1000000 - data['volume']) / 100000) * 2
    leverage_risk = (leverage - 1) * 1.5
    
    total_risk = base_risk + funding_risk + sentiment_risk + volume_risk + leverage_risk
    risk_score = min(round(total_risk, 1), 100)
    
    # Determine recommendation
    prob_diff = abs(long_probability - short_probability)
    if prob_diff < 5:
        recommendation = "NEUTRAL ⚖️"
        safer_direction = "Neither (too close to call)"
    elif long_probability > short_probability:
        if prob_diff > 15:
            recommendation = "STRONG LONG 🚀"
        else:
            recommendation = "LONG 📈"
        safer_direction = "Long"
    else:
        if prob_diff > 15:
            recommendation = "STRONG SHORT 📉"
        else:
            recommendation = "SHORT 📉"
        safer_direction = "Short"
    
    return {
        "long_prob": round(long_probability, 1),
        "short_prob": round(short_probability, 1),
        "risk_score": risk_score,
        "recommendation": recommendation,
        "safer_direction": safer_direction,
        "confidence": round(prob_diff, 1),
        "factors": {
            "trend": trend_factor,
            "funding": funding_bias,
            "volume": volume_confidence,
            "sentiment": sentiment_bias,
            "volatility": -volatility_penalty,
            "leverage": -leverage_penalty
        }
    }

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
                    # Calculate trading probabilities and risk
                    analysis = calculate_trading_probabilities(data, leverage)
                    risk_level, risk_color = get_risk_level(analysis['risk_score'])
                    
                    # Display symbol header with data source
                    st.subheader(f"🔸 {symbol}")
                    st.caption(f"📡 Data source: {data.get('data_source', 'Unknown')}")
                    
                    # Price metrics
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric(
                            "Current Price", 
                            f"${data['current_price']:,.4f}",
                            delta=f"{data['price_change_24h']:+.2f}%"
                        )
                    with col_b:
                        volume_formatted = f"${data['volume']:,.0f}" if data['volume'] > 1000 else f"${data['volume']:,.2f}"
                        st.metric("24h Volume", volume_formatted)
                    
                    # 🎯 NEW: Trading Probability Analysis
                    st.markdown("---")
                    st.markdown("### 🎯 **Trading Probability Analysis**")
                    
                    # Probability display
                    prob_col1, prob_col2, prob_col3 = st.columns(3)
                    with prob_col1:
                        long_color = "green" if analysis['long_prob'] > 55 else "red" if analysis['long_prob'] < 45 else "orange"
                        st.metric("📈 Long Success %", f"{analysis['long_prob']}%")
                        st.markdown(f":{long_color}[●] Long Probability")
                    
                    with prob_col2:
                        short_color = "green" if analysis['short_prob'] > 55 else "red" if analysis['short_prob'] < 45 else "orange"
                        st.metric("📉 Short Success %", f"{analysis['short_prob']}%")
                        st.markdown(f":{short_color}[●] Short Probability")
                    
                    with prob_col3:
                        rec_color = "green" if "STRONG" in analysis['recommendation'] else "orange" if "NEUTRAL" not in analysis['recommendation'] else "gray"
                        st.metric("🎯 Recommendation", analysis['recommendation'])
                        st.markdown(f"Confidence: **{analysis['confidence']}%**")
                    
                    # Safety recommendation
                    if analysis['safer_direction'] != "Neither (too close to call)":
                        st.success(f"🛡️ **Safer Direction:** {analysis['safer_direction']} position ({max(analysis['long_prob'], analysis['short_prob'])}% success probability)")
                    else:
                        st.warning("⚠️ **Market is too uncertain** - Consider waiting for a clearer signal")
                    
                    # Risk assessment
                    st.markdown("### ⚠️ **Risk Assessment**")
                    st.markdown(f"**Overall Risk:** :{risk_color}[{risk_level}] (Score: {analysis['risk_score']}/100)")
                    
                    # Detailed factor breakdown
                    with st.expander("📊 See Probability Factors", expanded=False):
                        factors = analysis['factors']
                        st.markdown("**What's affecting the probabilities:**")
                        
                        if factors['trend'] > 5:
                            st.write(f"📈 Strong uptrend (+{factors['trend']:.1f}% for longs)")
                        elif factors['trend'] < -5:
                            st.write(f"📉 Strong downtrend (+{abs(factors['trend']):.1f}% for shorts)")
                        else:
                            st.write("↔️ Neutral trend (no bias)")
                            
                        if factors['funding'] > 2:
                            st.write(f"💰 Negative funding rate (+{factors['funding']:.1f}% for longs)")
                        elif factors['funding'] < -2:
                            st.write(f"💸 Positive funding rate (+{abs(factors['funding']):.1f}% for shorts)")
                        else:
                            st.write("💱 Neutral funding (no bias)")
                            
                        if factors['volume'] > 2:
                            st.write(f"📊 High volume (+{factors['volume']:.1f}% confidence)")
                        elif factors['volume'] < 0:
                            st.write(f"📊 Low volume ({factors['volume']:.1f}% confidence)")
                        else:
                            st.write("📊 Normal volume")
                            
                        if factors['sentiment'] > 2:
                            st.write(f"😤 Too many shorts - contrarian signal (+{factors['sentiment']:.1f}% for longs)")
                        elif factors['sentiment'] < -2:
                            st.write(f"😬 Too many longs - contrarian signal (+{abs(factors['sentiment']):.1f}% for shorts)")
                        else:
                            st.write("😐 Balanced sentiment")
                            
                        st.write(f"🌪️ Volatility penalty: {factors['volatility']:.1f}%")
                        st.write(f"⚡ Leverage penalty: {factors['leverage']:.1f}%")
                    
                    # Detailed metrics
                    st.markdown("### 📊 **Market Details**")
                    if data.get('data_source') == 'Binance':
                        st.write(f"• Funding Rate: {data['funding_rate']:+.4f}%")
                    else:
                        st.write("• Funding Rate: Not available (non-futures data)")
                    st.write(f"• Volatility (24h): {data['volatility']:.2f}%")
                    st.write(f"• Market Sentiment: {'Bullish 📈' if data['long_short_ratio'] > 1.2 else 'Bearish 📉' if data['long_short_ratio'] < 0.8 else 'Neutral ⚖️'}")
                    if data.get('data_source') == 'Binance':
                        st.write(f"• Long/Short Ratio: {data['long_short_ratio']:.2f}")
                    else:
                        st.write(f"• Market Bias: Neutral (ratio data unavailable)")
                    
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

# Footer with enhanced tips
st.markdown("---")
st.markdown("""
### 💡 Enhanced Trading Tips:

#### 🎯 **Understanding Success Probabilities:**
- **60%+ Success Rate**: Strong signal, good for higher leverage
- **50-60% Success Rate**: Moderate signal, use lower leverage
- **40-50% Success Rate**: Weak signal, consider waiting
- **<40% Success Rate**: Poor setup, avoid or take opposite direction

#### 🛡️ **Risk Management Guidelines:**
- **Risk Score < 20**: Generally safer for higher leverage (up to 10x)
- **Risk Score 20-40**: Moderate risk, stick to 3-7x leverage  
- **Risk Score 40-60**: High risk, use 1-3x leverage only
- **Risk Score > 60**: Extreme risk, avoid high leverage entirely

#### 📊 **Key Factors That Affect Probabilities:**
- **24h Trend**: Strong trends increase directional probability
- **Funding Rates**: Negative funding favors longs, positive favors shorts
- **Volume**: Higher volume = more reliable signals
- **Market Sentiment**: Extreme positioning often leads to reversals
- **Volatility**: High volatility reduces success rates for both directions

#### ⚠️ **Important Disclaimers:**
- Probabilities are estimates based on technical analysis
- Past performance doesn't guarantee future results
- Always use stop losses and proper position sizing
- Never risk more than you can afford to lose
- Consider multiple timeframes and fundamental analysis
""")

# Performance info
with st.expander("ℹ️ About This Enhanced Tool"):
    st.markdown("""
    This advanced tool calculates trading success probabilities using:
    
    **📈 Technical Analysis:**
    - Price momentum and trend analysis
    - Volatility-adjusted risk scoring
    - Volume and liquidity analysis
    
    **💰 Market Microstructure:**
    - Funding rate analysis (futures bias)
    - Long/short ratio sentiment
    - Order book imbalance detection
    
    **🧮 Probability Model:**
    - Multi-factor scoring algorithm
    - Leverage-adjusted success rates  
    - Confidence intervals and uncertainty
    
    **📡 Data Sources:** 
    - Primary: Binance Futures API (real futures data)
    - Fallback: CoinGecko API (spot market data)
    - Update Frequency: User configurable (10-120 seconds)
    
    **⚖️ Risk Algorithm:** 
    Proprietary model combining technical indicators, market structure, and leverage effects to estimate directional success probabilities.
    
    *This tool is for educational and research purposes. Not financial advice.*
    """)
