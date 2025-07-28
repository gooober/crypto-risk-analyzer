@st.cache_data(ttl=5)
def get_enhanced_data(symbol):
    """
    Fetch enhanced data with technical indicators, VWAP, and order flow using Bybit Perpetuals API.
    Plug-and-play: No other code changes needed.
    """
    try:
        # Bybit Perpetual Ticker
        url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}"
        resp = requests.get(url, timeout=5)
        ticker_data = resp.json()['result']['list'][0]

        # Bybit Perpetual Kline for 1m candles (for VWAP etc)
        kline_url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval=1&limit=100"
        klines = requests.get(kline_url, timeout=5).json()['result']['list']

        # Bybit recent trades (for order flow)
        trades_url = f"https://api.bybit.com/v5/market/trade?category=linear&symbol={symbol}&limit=100"
        trades = requests.get(trades_url, timeout=5).json()['result']['list']

        # Compute indicators
        closes = [float(k[4]) for k in klines]
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = gains[-14:].mean()
        avg_loss = losses[-14:].mean() if losses[-14:].mean() > 0 else 1
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        if len(closes) >= 26:
            ema12 = np.mean(closes[-12:])
            ema26 = np.mean(closes[-26:])
            macd = (ema12 - ema26) / ema26 * 100
        else:
            macd = 0

        vwap_calc, vwap_up, vwap_low = calculate_vwap(klines)
        vwap = vwap_calc if not np.isnan(vwap_calc) else float(ticker_data['lastPrice'])
        vwap_upper = vwap_up
        vwap_lower = vwap_low

        imbalance, buy_vol, sell_vol = calculate_order_flow_imbalance(trades)

        return {
            'data_source': 'Bybit Perpetuals',
            'price': float(ticker_data['lastPrice']),
            'price_change_24h': float(ticker_data['price24hPcnt']) * 100,
            'volume': float(ticker_data['turnover24h']),
            'high_24h': float(ticker_data['highPrice24h']),
            'low_24h': float(ticker_data['lowPrice24h']),
            'rsi': round(rsi, 2),
            'macd': round(macd, 2),
            'vwap': round(vwap, 2),
            'vwap_upper': round(vwap_upper, 2),
            'vwap_lower': round(vwap_lower, 2),
            'price_vs_vwap': round((float(ticker_data['lastPrice']) - vwap) / vwap * 100, 2),
            'order_flow_imbalance': imbalance,
            'aggressive_buy_volume': buy_vol,
            'aggressive_sell_volume': sell_vol,
            'last_updated': datetime.now().strftime('%H:%M:%S')
        }
    except Exception as e:
        return {
            'data_source': f'Demo Mode (Bybit error: {str(e)[:50]})',
            'price': np.nan,
            'price_change_24h': np.nan,
            'volume': np.nan,
            'high_24h': np.nan,
            'low_24h': np.nan,
            'rsi': np.nan,
            'macd': np.nan,
            'vwap': np.nan,
            'vwap_upper': np.nan,
            'vwap_lower': np.nan,
            'price_vs_vwap': np.nan,
            'order_flow_imbalance': np.nan,
            'aggressive_buy_volume': 0,
            'aggressive_sell_volume': 0,
            'last_updated': datetime.now().strftime('%H:%M:%S')
        }
