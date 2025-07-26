import streamlit as st
import requests
import time

# --- Binance API Functions ---

def get_funding_rate(symbol='BTCUSDT'):
    url = f'https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1'
    res = requests.get(url).json()
    return float(res[0]['fundingRate'])

def get_24h_stats(symbol='BTCUSDT'):
    url = f'https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={symbol}'
    res = requests.get(url).json()
    return float(res['priceChangePercent']), float(res['lastPrice'])

def get_1h_price_change(symbol='BTCUSDT'):
    url = f'https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1h&limit=2'
    res = requests.get(url).json()
    open_price = float(res[0][1])
    close_price = float(res[1][4])
    price_change_pct = ((close_price - open_price) / open_price) * 100
    return price_change_pct

def get_position_risk_ratio(symbol='BTCUSDT'):
    url = f'https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=5m&limit=1'
    res = requests.get(url).json()
    if res:
        long_pct = float(res[0]['longAccount']) * 100
        short_pct = float(res[0]['shortAccount']) * 100
        return round(long_pct, 2), round(short_pct, 2)
    return None, None

# --- Risk Calculation ---

def calculate_risk(funding_rate, volatility_percent, leverage):
    funding_risk = min(abs(funding_rate) * 1000, 100)
    volatility_risk = min(abs(volatility_percent), 100)
    leverage_risk = min(leverage * 5, 100)
    total_risk = 0.4 * funding_risk + 0.4 * volatility_risk + 0.2 * leverage_risk
    return round(total_risk, 2)

def estimate_directional_risk(price_change_1h):
    if price_change_1h < -2:
        drop_risk = min(abs(price_change_1h) * 5, 100)
        rise_chance = 100 - drop_risk
    elif price_change_1h > 2:
        rise_chance = min(price_change_1h * 5, 100)
        drop_risk = 100 - rise_chance
    else:
        drop_risk = 50
        rise_chance = 50
    return round(drop_risk, 2), round(rise_chance, 2)

def analyze_symbol(symbol='BTCUSDT', leverage=10):
    funding = get_funding_rate(symbol)
    vol, price = get_24h_stats(symbol)
    price_change_1h = get_1h_price_change(symbol)
    risk = calculate_risk(funding, vol, leverage)
    drop_risk, rise_chance = estimate_directional_risk(price_change_1h)
    long_pct, short_pct = get_position_risk_ratio(symbol)

    return {
        'Symbol': symbol,
        'Funding Rate': f"{funding:.5f}",
        'Volatility (24h)': f"{vol:.2f}%",
        'Price': f"${price:,.2f}",
        'Leverage': f"{leverage}x",
        'Risk Level': f"{risk}%" if risk < 70 else f"⚠️ {risk}%",
        '1h Price Change': f"{price_change_1h:.2f}%",
        'Price Drop Risk': f"{drop_risk}%",
        'Price Rise Chance': f"{rise_chance}%",
        'Long Positions': f"{long_pct}%" if long_pct is not None else "N/A",
        'Short Positions': f"{short_pct}%" if short_pct is not None else "N/A"
    }

# --- Streamlit UI ---

st.set_page_config(page_title="Perpetual Trade Risk Analyzer", layout="wide")
st.title("📊 Perpetual Crypto Trade Risk Analyzer")

symbols = st.multiselect(
    "Select symbols to analyze:",
    ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'ADAUSDT', 'LTCUSDT'],
    default=['BTCUSDT', 'ETHUSDT']
)

leverage = st.slider("Set your leverage (X):", min_value=1, max_value=50, value=10)

refresh_sec = st.slider("Auto-refresh every (seconds):", 5, 60, 15)
auto_refresh = st.checkbox("🔄 Enable Auto-Refresh", value=False)

st.subheader("📈 Risk Analysis Results:")

for symbol in symbols:
    try:
        data = analyze_symbol(symbol, leverage)
        st.write("---")
        st.markdown(f"### {data['Symbol']}")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📉 Funding Rate", data['Funding Rate'])
            st.metric("📊 Volatility (24h)", data['Volatility (24h)'])
            st.metric("💰 Price", data['Price'])

        with col2:
            st.metric("⚖️ Leverage", data['Leverage'])
            st.metric("☢️ Risk Level", data['Risk Level'])
            st.metric("📈 1h Price Change", data['1h Price Change'])

        with col3:
            st.metric("🔻 Price Drop Risk", data['Price Drop Risk'])
            st.metric("🚀 Price Rise Chance", data['Price Rise Chance'])
            st.metric("📈 Long vs Short", f"🟢 {data['Long Positions']} / 🔴 {data['Short Positions']}")

    except Exception as e:
        st.error(f"Error loading data for {symbol}: {e}")

if auto_refresh:
    st.info(f"Auto-refreshing every {refresh_sec} seconds...")
    time.sleep(refresh_sec)
    st.experimental_rerun()
