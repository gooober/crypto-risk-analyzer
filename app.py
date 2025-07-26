import streamlit as st
import requests
import time

st.set_page_config(page_title="Perpetual Crypto Trade Risk Analyzer", layout="wide")

st.title("📊 Perpetual Crypto Trade Risk Analyzer")

symbols = st.multiselect("Select symbols to analyze:", ["BTCUSDT", "ETHUSDT", "SOLUSDT"], default=["BTCUSDT"])
leverage = st.slider("Set your leverage (X):", 1, 50, 10)
auto_refresh = st.checkbox("🔄 Enable Auto-Refresh")
refresh_rate = st.slider("Auto-refresh every (seconds):", 5, 60, 15)

st.markdown("---")
st.header("📉 Risk Analysis Results:")

# Function to fetch Binance Futures data
def get_futures_data(symbol):
    base_url = "https://fapi.binance.com"
    try:
        funding = requests.get(f"{base_url}/fapi/v1/fundingRate", params={"symbol": symbol, "limit": 1}, headers={"User-Agent": "Mozilla/5.0"}).json()
        funding_rate = float(funding[0]['fundingRate']) * 100 if funding else 0.0

        stats = requests.get(f"{base_url}/fapi/v1/ticker/24hr", params={"symbol": symbol}, headers={"User-Agent": "Mozilla/5.0"}).json()
        price_change_percent = float(stats.get("priceChangePercent", 0))

        klines = requests.get(f"{base_url}/fapi/v1/klines", params={"symbol": symbol, "interval": "1h", "limit": 2}, headers={"User-Agent": "Mozilla/5.0"}).json()
        price_1h_ago = float(klines[0][4])
        price_now = float(klines[1][4])
        price_change_1h = ((price_now - price_1h_ago) / price_1h_ago) * 100

        depth = requests.get(f"{base_url}/fapi/v1/depth", params={"symbol": symbol, "limit": 5}, headers={"User-Agent": "Mozilla/5.0"}).json()
        bid_qty = sum([float(bid[1]) for bid in depth['bids']])
        ask_qty = sum([float(ask[1]) for ask in depth['asks']])
        long_short_ratio = bid_qty / (ask_qty + 1e-9)

        return {
            "funding_rate": funding_rate,
            "volatility": abs(price_change_percent),
            "trend_1h": price_change_1h,
            "price_now": price_now,
            "long_short_ratio": long_short_ratio
        }
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {e}")
        return None

# Function to calculate simple risk score
def calculate_risk(data, leverage):
    trend_penalty = -abs(data['trend_1h']) if data['trend_1h'] < 0 else 0
    risk = data['volatility'] * leverage + abs(data['funding_rate']) + trend_penalty
    return round(risk, 2)

# User trade input section
st.markdown("## 🧾 Enter Your Trade")
with st.form("trade_form"):
    selected_trade = st.selectbox("Your active trade symbol:", symbols)
    entry_price = st.number_input("Your entry price:", min_value=0.0, format="%.2f")
    direction = st.radio("Direction:", ["Long", "Short"])
    submitted = st.form_submit_button("Analyze My Trade")

# Main logic
placeholder = st.empty()
while True:
    with placeholder.container():
        for symbol in symbols:
            data = get_futures_data(symbol)
            if data:
                risk_score = calculate_risk(data, leverage)
                delta = data['price_now'] - entry_price if submitted and symbol == selected_trade else 0
                pnl_color = "green" if (delta > 0 and direction == "Long") or (delta < 0 and direction == "Short") else "red"

                with st.container():
                    st.subheader(f"{symbol}")
                    st.metric("Current Price", f"{data['price_now']:.2f}", delta=f"{delta:.2f}" if submitted and symbol == selected_trade else "")
                    st.write(f"📌 **Funding Rate:** {data['funding_rate']:.4f}%")
                    st.write(f"📈 **24h Volatility:** {data['volatility']:.2f}%")
                    st.write(f"🕐 **1h Trend:** {data['trend_1h']:.2f}%")
                    st.write(f"📊 **Long/Short Bias:** {'Bullish' if data['long_short_ratio'] > 1 else 'Bearish'} ({data['long_short_ratio']:.2f}x)")
                    st.warning(f"⚠️ Risk Score: {risk_score}")

        if not auto_refresh:
            break
        time.sleep(refresh_rate)
        st.experimental_rerun()
