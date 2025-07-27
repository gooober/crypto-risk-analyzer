st.plotly_chart(fig, use_container_width=True)
            
            with chart_col2:
                # Signal strength gauge
                fig2 = go.Figure()
                
                # Long probability gauge
                fig2.add_trace(go.Indicator(
                    mode = "gauge+number+delta",
                    value = analysis['long_prob'],
                    domain = {'x': [0, 1], 'y': [0.55, 1]},
                    title = {'text': "Long Signal Strength", 'font': {'size': 16}},
                    delta = {'reference': 50, 'increasing': {'color': "green"}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1},
                    'bar': {'color': "green" if avg_sentiment > 60 else "red" if avg_sentiment < 40 else "yellow"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 20], 'color': 'darkred'},
                        {'range': [20, 40], 'color': 'red'},
                        {'range': [40, 60], 'color': 'yellow'},
                        {'range': [60, 80], 'color': 'lightgreen'},
                        {'range': [80, 100], 'color': 'green'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig_sentiment.update_layout(
                height=300,
                template="plotly_dark"
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
            
            # Sentiment breakdown
            st.markdown("### 📊 Sentiment Factors")
            
            if symbols:
                symbol = symbols[0]  # Use first symbol for detailed sentiment
                data = get_enhanced_data(symbol)
                if data and 'sentiment_factors' in data:
                    for factor in data['sentiment_factors']:
                        if "+" in factor:
                            st.success(f"✅ {factor}")
                        elif "-" in factor:
                            st.error(f"❌ {factor}")
                        else:
                            st.info(f"ℹ️ {factor}")
        
        # Social media sentiment (simulated)
        st.markdown("---")
        st.subheader("💬 Social Media Sentiment")
        
        social_cols = st.columns(4)
        
        with social_cols[0]:
            twitter_sentiment = np.random.randint(40, 80)
            st.metric("Twitter", f"{twitter_sentiment}%", 
                     "Bullish" if twitter_sentiment > 50 else "Bearish")
        
        with social_cols[1]:
            reddit_sentiment = np.random.randint(35, 75)
            st.metric("Reddit", f"{reddit_sentiment}%",
                     "Bullish" if reddit_sentiment > 50 else "Bearish")
        
        with social_cols[2]:
            telegram_sentiment = np.random.randint(45, 85)
            st.metric("Telegram", f"{telegram_sentiment}%",
                     "Bullish" if telegram_sentiment > 50 else "Bearish")
        
        with social_cols[3]:
            overall_social = (twitter_sentiment + reddit_sentiment + telegram_sentiment) / 3
            st.metric("Overall", f"{overall_social:.0f}%",
                     "Bullish" if overall_social > 50 else "Bearish")

with tab6:
    st.header("💼 Paper Trading Simulator")
    
    if enable_paper_trading:
        # Trading interface
        trade_col1, trade_col2 = st.columns([1, 2])
        
        with trade_col1:
            st.subheader("🎯 Place New Trade")
            
            # Trade form
            with st.form("paper_trade_form"):
                pt_symbol = st.selectbox("Symbol:", symbols if symbols else ["BTCUSDT"])
                pt_direction = st.radio("Direction:", ["Long", "Short"])
                pt_size = st.number_input("Position Size ($):", 100, 10000, 1000)
                pt_leverage = st.slider("Leverage:", 1, 50, leverage)
                
                # Get current data for selected symbol
                pt_data = get_enhanced_data(pt_symbol)
                if pt_data:
                    st.info(f"Current Price: ${pt_data['current_price']:,.4f}")
                    analysis = calculate_ai_probabilities(pt_data, pt_leverage, selected_strategy)
                    
                    # Show recommendations
                    if analysis['direction'] == pt_direction:
                        st.success(f"✅ Trade aligns with signal: {analysis['recommendation']}")
                    else:
                        st.warning(f"⚠️ Trade against signal: {analysis['recommendation']}")
                    
                    st.write(f"Suggested SL: ${analysis['stop_loss']:,.4f}")
                    st.write(f"Suggested TP: ${analysis['take_profit']:,.4f}")
                
                submitted = st.form_submit_button("Execute Trade", type="primary", use_container_width=True)
                
                if submitted and pt_data:
                    trade = execute_paper_trade(
                        pt_symbol, pt_direction, pt_size, pt_leverage, 
                        pt_data['current_price']
                    )
                    st.success(f"✅ Trade #{trade['id']} executed!")
                    st.balloons()
        
        with trade_col2:
            st.subheader("📊 Active Positions")
            
            # Display active trades
            if st.session_state.trade_history:
                # Update P&L with current prices
                update_paper_trades(current_prices if 'current_prices' in locals() else {})
                
                # Portfolio summary
                total_pnl = sum(t['pnl'] for t in st.session_state.trade_history)
                open_trades = [t for t in st.session_state.trade_history if t['status'] == 'open']
                closed_trades = [t for t in st.session_state.trade_history if t['status'] == 'closed']
                
                summary_cols = st.columns(4)
                with summary_cols[0]:
                    st.metric("Total P&L", f"${total_pnl:+,.2f}", 
                             f"{total_pnl/sum(t['size'] for t in st.session_state.trade_history)*100:+.1f}%" if st.session_state.trade_history else "0%")
                with summary_cols[1]:
                    st.metric("Open Positions", len(open_trades))
                with summary_cols[2]:
                    winning_trades = sum(1 for t in st.session_state.trade_history if t['pnl'] > 0)
                    st.metric("Win Rate", 
                             f"{winning_trades/len(st.session_state.trade_history)*100:.1f}%" if st.session_state.trade_history else "0%")
                with summary_cols[3]:
                    st.metric("Total Volume", f"${sum(t['size'] for t in st.session_state.trade_history):,.0f}")
                
                # Open positions table
                if open_trades:
                    st.markdown("### 📈 Open Positions")
                    
                    positions_data = []
                    for trade in open_trades:
                        positions_data.append({
                            'ID': trade['id'],
                            'Symbol': trade['symbol'],
                            'Direction': trade['direction'],
                            'Entry': trade['entry_price'],
                            'Current': trade.get('current_price', trade['entry_price']),
                            'Size': trade['size'],
                            'Leverage': f"{trade['leverage']}x",
                            'P&L': trade['pnl'],
                            'P&L %': trade['pnl_percent'],
                            'Time': trade['entry_time'].strftime('%H:%M:%S')
                        })
                    
                    df_positions = pd.DataFrame(positions_data)
                    
                    # Custom styling
                    st.dataframe(
                        df_positions.style.format({
                            'Entry': '${:.4f}',
                            'Current': '${:.4f}',
                            'Size': '${:.0f}',
                            'P&L': '${:+.2f}',
                            'P&L %': '{:+.2f}%'
                        }).apply(lambda x: ['background-color: #1a4d2e' if x['P&L'] > 0 else 'background-color: #4d1a1a' 
                                           for _ in range(len(x))], axis=1, subset=['P&L', 'P&L %']),
                        use_container_width=True
                    )
                    
                    # Close position buttons
                    st.markdown("### 🎯 Manage Positions")
                    
                    close_col1, close_col2, close_col3 = st.columns(3)
                    
                    with close_col1:
                        position_to_close = st.selectbox("Select Position:", 
                                                        [f"#{t['id']} - {t['symbol']} {t['direction']}" for t in open_trades])
                    
                    with close_col2:
                        if st.button("Close Position", type="secondary", use_container_width=True):
                            # Extract position ID
                            position_id = int(position_to_close.split('#')[1].split(' ')[0])
                            
                            # Find and close the position
                            for trade in st.session_state.trade_history:
                                if trade['id'] == position_id and trade['status'] == 'open':
                                    trade['status'] = 'closed'
                                    trade['exit_time'] = datetime.now()
                                    trade['exit_price'] = trade.get('current_price', trade['entry_price'])
                                    st.success(f"✅ Position #{position_id} closed! P&L: ${trade['pnl']:+.2f}")
                                    st.rerun()
                    
                    with close_col3:
                        if st.button("Close All Positions", type="secondary", use_container_width=True):
                            for trade in st.session_state.trade_history:
                                if trade['status'] == 'open':
                                    trade['status'] = 'closed'
                                    trade['exit_time'] = datetime.now()
                                    trade['exit_price'] = trade.get('current_price', trade['entry_price'])
                            st.success("✅ All positions closed!")
                            st.rerun()
                
                # Performance chart
                if len(st.session_state.trade_history) > 1:
                    st.markdown("### 📈 Performance Chart")
                    
                    # Calculate cumulative P&L
                    cumulative_pnl = []
                    running_total = 0
                    times = []
                    
                    for trade in sorted(st.session_state.trade_history, key=lambda x: x['entry_time']):
                        running_total += trade['pnl']
                        cumulative_pnl.append(running_total)
                        times.append(trade['entry_time'])
                    
                    fig_performance = go.Figure()
                    
                    fig_performance.add_trace(go.Scatter(
                        x=times,
                        y=cumulative_pnl,
                        mode='lines+markers',
                        name='Cumulative P&L',
                        line=dict(color='cyan', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(0, 255, 255, 0.1)'
                    ))
                    
                    # Add zero line
                    fig_performance.add_hline(y=0, line_dash="dash", line_color="gray")
                    
                    fig_performance.update_layout(
                        title="Portfolio Performance",
                        xaxis_title="Time",
                        yaxis_title="Cumulative P&L ($)",
                        template="plotly_dark",
                        height=400
                    )
                    st.plotly_chart(fig_performance, use_container_width=True)
            else:
                st.info("📌 No trades yet. Place your first trade to start tracking performance!")
    else:
        st.info("📌 Enable Paper Trading in the sidebar to access this feature.")

# Footer with tips and information
st.markdown("---")

# Tips section
with st.expander("💡 Pro Trading Tips", expanded=False):
    tips_tabs = st.tabs(["🎯 Entry", "📊 Management", "🚪 Exit", "🧠 Psychology"])
    
    with tips_tabs[0]:
        st.markdown("""
        ### 🎯 Entry Strategies
        
        **1. Wait for Confluence**
        - Multiple timeframes align
        - Technical indicators confirm
        - Volume supports the move
        - Risk/Reward > 2:1
        
        **2. Best Entry Signals**
        - RSI divergence + Support/Resistance
        - MACD cross + Volume spike
        - Bollinger Band squeeze breakout
        - Multi-timeframe alignment
        
        **3. Avoid These Mistakes**
        - FOMO entries on big green candles
        - Catching falling knives
        - Trading against the trend
        - Overleveraging on weak signals
        """)
    
    with tips_tabs[1]:
        st.markdown("""
        ### 📊 Position Management
        
        **1. Position Sizing**
        - Risk 1-2% per trade maximum
        - Reduce size in high volatility
        - Scale in/out of positions
        - Adjust for correlation
        
        **2. Leverage Guidelines**
        - <20 Risk Score: Max 10x
        - 20-40 Risk Score: Max 5x
        - 40-60 Risk Score: Max 3x
        - >60 Risk Score: No leverage
        
        **3. Active Management**
        - Move stop to breakeven at 1R profit
        - Take partial profits at targets
        - Add to winners, not losers
        - Monitor funding rates
        """)
    
    with tips_tabs[2]:
        st.markdown("""
        ### 🚪 Exit Strategies
        
        **1. Take Profit Methods**
        - Fixed targets (2R, 3R, 5R)
        - Resistance/Support levels
        - Trailing stop (ATR-based)
        - Partial exits at milestones
        
        **2. Stop Loss Discipline**
        - Always use stops, no exceptions
        - ATR-based stops (2-3x ATR)
        - Below support/above resistance
        - Time-based stops for scalping
        
        **3. Exit Signals**
        - RSI overbought/oversold
        - Divergence forming
        - Volume declining
        - Sentiment shifting
        """)
    
    with tips_tabs[3]:
        st.markdown("""
        ### 🧠 Trading Psychology
        
        **1. Emotional Control**
        - Never revenge trade
        - Accept losses as business cost
        - Don't move stops (except to profit)
        - Take breaks after big wins/losses
        
        **2. Discipline Rules**
        - Follow your system
        - Journal every trade
        - Review weekly performance
        - Continuous education
        
        **3. Success Mindset**
        - Think in probabilities
        - Focus on process, not outcomes
        - Preserve capital first
        - Compound gains slowly
        """)

# Disclaimer
with st.expander("⚠️ Risk Disclaimer & Terms"):
    st.markdown("""
    ### ⚠️ IMPORTANT RISK DISCLAIMER
    
    **CRYPTOCURRENCY TRADING RISK WARNING:**
    
    Trading cryptocurrencies carries a substantial risk of loss and is not suitable for every investor. 
    The valuation of cryptocurrencies may fluctuate, and, as a result, you may lose more than your original investment.
    
    **KEY RISKS:**
    - **Market Risk**: Extreme price volatility
    - **Leverage Risk**: Amplified losses with leveraged trading
    - **Liquidity Risk**: Inability to exit positions
    - **Technical Risk**: Platform failures, hacks
    - **Regulatory Risk**: Changing regulations
    - **Counterparty Risk**: Exchange insolvency
    
    **DISCLAIMER:**
    - This tool is for educational and informational purposes only
    - Not financial, investment, or trading advice
    - Past performance does not indicate future results
    - We are not responsible for any losses incurred
    - Always do your own research (DYOR)
    - Never invest more than you can afford to lose
    
    **TERMS OF USE:**
    - By using this tool, you accept all risks
    - You are solely responsible for your trading decisions
    - This tool does not guarantee profits
    - Technical indicators are not infallible
    - Backtesting results are hypothetical
    
    **RECOMMENDATIONS:**
    - Start with a demo account
    - Educate yourself thoroughly
    - Use proper risk management
    - Consider seeking professional advice
    - Keep detailed records for taxes
    
    **If you do not fully understand and accept these risks, do not use this tool or trade cryptocurrencies.**
    
    ---
    
    *Version 2.0 Pro | Data provided by public APIs | Not affiliated with any exchange*
    """)

# Performance metrics in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    
    session_duration = (datetime.now() - st.session_state.get('session_start', datetime.now())).seconds // 60
    st.write(f"Session Duration: {session_duration} min")
    st.write(f"Alerts Generated: {len(st.session_state.alerts)}")
    st.write(f"Trades Executed: {len(st.session_state.trade_history)}")
    
    # Quick links
    st.markdown("### 🔗 Quick Links")
    st.markdown("[📚 Documentation](https://example.com)")
    st.markdown("[💬 Community](https://example.com)")
    st.markdown("[🐛 Report Bug](https://example.com)")
    st.markdown("[✨ Feature Request](https://example.com)")

# Auto-save session state
if 'session_start' not in st.session_state:
    st.session_state.session_start = datetime.now()

# Hidden debug info (uncomment for debugging)
# with st.expander("🔧 Debug Info"):
#     st.json({
#         'session_state_keys': list(st.session_state.keys()),
#         'active_symbols': symbols,
#         'cache_size': len(st.session_state.data_cache),
#         'alert_count': len(st.session_state.alerts),
#         'trade_count': len(st.session_state.trade_history)
#     }) 100], 'tickwidth': 1},
                        'bar': {'color': "darkgreen" if analysis['long_prob'] > 60 else "orange" if analysis['long_prob'] > 40 else "darkred"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 40], 'color': '#ffebee'},
                            {'range': [40, 60], 'color': '#fff3e0'},
                            {'range': [60, 100], 'color': '#e8f5e9'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                # Risk gauge
                fig2.add_trace(go.Indicator(
                    mode = "gauge+number+delta",
                    value = analysis['risk_score'],
                    domain = {'x': [0, 1], 'y': [0, 0.45]},
                    title = {'text': "Risk Level", 'font': {'size': 16}},
                    delta = {'reference': 50, 'decreasing': {'color': "green"}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1},
                        'bar': {'color': "darkred" if analysis['risk_score'] > 60 else "orange" if analysis['risk_score'] > 40 else "darkgreen"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 20], 'color': '#e8f5e9'},
                            {'range': [20, 40], 'color': '#fff9c4'},
                            {'range': [40, 60], 'color': '#ffccbc'},
                            {'range': [60, 80], 'color': '#ffab91'},
                            {'range': [80, 100], 'color': '#ef5350'}
                        ]
                    }
                ))
                
                fig2.update_layout(
                    height=400,
                    template="plotly_dark",
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # ML Predictions
                if use_ai_predictions:
                    st.markdown("### 🤖 AI Price Predictions")
                    predictions = get_ml_prediction(selected_symbol, data)
                    if predictions:
                        pred_cols = st.columns(3)
                        with pred_cols[0]:
                            st.metric("1H Target", 
                                     f"${predictions['1h']:.2f}",
                                     f"{((predictions['1h']/data['current_price'])-1)*100:.2f}%")
                        with pred_cols[1]:
                            st.metric("4H Target", 
                                     f"${predictions['4h']:.2f}",
                                     f"{((predictions['4h']/data['current_price'])-1)*100:.2f}%")
                        with pred_cols[2]:
                            st.metric("24H Target", 
                                     f"${predictions['24h']:.2f}",
                                     f"{((predictions['24h']/data['current_price'])-1)*100:.2f}%")
                        
                        st.progress(predictions['confidence'] / 100)
                        st.caption(f"AI Confidence: {predictions['confidence']:.0f}%")
            
            # Detailed Analysis Sections
            st.markdown("---")
            
            # Trading signals breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Signal Analysis")
                
                # Create signal breakdown chart
                if 'signals' in analysis:
                    signal_names = list(analysis['signals'].keys())
                    signal_values = list(analysis['signals'].values())
                    colors = ['green' if v > 0 else 'red' for v in signal_values]
                    
                    fig_signals = go.Figure(data=[
                        go.Bar(
                            x=signal_values,
                            y=signal_names,
                            orientation='h',
                            marker_color=colors,
                            text=[f"{v:+.1f}" for v in signal_values],
                            textposition='auto',
                        )
                    ])
                    
                    fig_signals.update_layout(
                        title="Signal Contributions",
                        xaxis_title="Signal Strength",
                        template="plotly_dark",
                        height=300
                    )
                    st.plotly_chart(fig_signals, use_container_width=True)
                
                # Risk factors
                if 'risk_factors' in analysis:
                    st.markdown("### ⚠️ Risk Breakdown")
                    for factor in analysis['risk_factors']:
                        st.write(f"• {factor}")
            
            with col2:
                st.markdown("### 🎯 Trade Setup")
                
                # Recommended trade parameters
                trade_params = st.container()
                with trade_params:
                    st.markdown(f"""
                    <div style="background: rgba(0,100,200,0.1); padding: 20px; border-radius: 10px; border: 1px solid #0066cc;">
                        <h4>Recommended Parameters</h4>
                        <table style="width: 100%;">
                            <tr><td><b>Direction:</b></td><td>{analysis['direction']}</td></tr>
                            <tr><td><b>Entry:</b></td><td>${analysis['entry_price']:,.4f}</td></tr>
                            <tr><td><b>Stop Loss:</b></td><td>${analysis['stop_loss']:,.4f}</td></tr>
                            <tr><td><b>Take Profit:</b></td><td>${analysis['take_profit']:,.4f}</td></tr>
                            <tr><td><b>Position Size:</b></td><td>{analysis['position_size_percent']}%</td></tr>
                            <tr><td><b>Risk/Reward:</b></td><td>1:{abs((analysis['take_profit']-analysis['entry_price'])/(analysis['entry_price']-analysis['stop_loss'])):.1f}</td></tr>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Market context
                st.markdown("### 🌍 Market Context")
                if 'sentiment_factors' in data:
                    for factor in data['sentiment_factors'][:5]:
                        st.write(f"• {factor}")
            
            # Order book visualization
            if show_order_book and 'order_book_levels' in data and data['order_book_levels']:
                st.markdown("---")
                st.markdown("### 📖 Order Book Analysis")
                
                ob_cols = st.columns(3)
                with ob_cols[0]:
                    st.metric("Order Imbalance", 
                             f"{data['order_book_imbalance']:.1f}%",
                             "Bullish" if data['order_book_imbalance'] > 0 else "Bearish")
                with ob_cols[1]:
                    st.metric("Bid Liquidity", f"${data['bid_liquidity']/1e6:.2f}M")
                with ob_cols[2]:
                    st.metric("Ask Liquidity", f"${data['ask_liquidity']/1e6:.2f}M")
                
                # Order book visualization
                if data['order_book_levels'].get('bid_wall_price'):
                    st.info(f"🟢 Major Bid Wall: ${data['order_book_levels']['bid_wall_price']:.2f} "
                           f"(Size: {data['order_book_levels']['bid_wall_size']:.0f})")
                if data['order_book_levels'].get('ask_wall_price'):
                    st.warning(f"🔴 Major Ask Wall: ${data['order_book_levels']['ask_wall_price']:.2f} "
                              f"(Size: {data['order_book_levels']['ask_wall_size']:.0f})")
            
            # Support & Resistance Levels
            if 'support_resistance' in data:
                st.markdown("---")
                st.markdown("### 💪 Support & Resistance Levels")
                
                sr_data = data['support_resistance']
                
                # Pivot points
                pivot_cols = st.columns(5)
                with pivot_cols[0]:
                    st.metric("S3", f"${sr_data.get('s3', 0):.2f}")
                with pivot_cols[1]:
                    st.metric("S2", f"${sr_data.get('s2', 0):.2f}")
                with pivot_cols[2]:
                    st.metric("Pivot", f"${sr_data.get('pivot', 0):.2f}")
                with pivot_cols[3]:
                    st.metric("R1", f"${sr_data.get('r1', 0):.2f}")
                with pivot_cols[4]:
                    st.metric("R2", f"${sr_data.get('r2', 0):.2f}")
                
                # Fibonacci levels
                st.markdown("**Fibonacci Retracements**")
                fib_cols = st.columns(4)
                with fib_cols[0]:
                    st.write(f"23.6%: ${sr_data.get('fib_236', 0):.2f}")
                with fib_cols[1]:
                    st.write(f"38.2%: ${sr_data.get('fib_382', 0):.2f}")
                with fib_cols[2]:
                    st.write(f"50.0%: ${sr_data.get('fib_500', 0):.2f}")
                with fib_cols[3]:
                    st.write(f"61.8%: ${sr_data.get('fib_618', 0):.2f}")

with tab3:
    st.header("📈 Market Overview & Correlation")
    
    if symbols:
        # Market summary metrics
        st.subheader("🌍 Market Summary")
        
        market_data = []
        current_prices = {}
        
        for symbol in symbols:
            data = get_enhanced_data(symbol)
            if data:
                analysis = calculate_ai_probabilities(data, leverage, selected_strategy)
                current_prices[symbol] = data['current_price']
                
                market_data.append({
                    'Symbol': symbol.replace('USDT', ''),
                    'Price': data['current_price'],
                    '24h %': data['price_change_24h'],
                    'Volume': data['volume'],
                    'RSI': data['rsi'],
                    'Signal': analysis['recommendation'].split()[0],
                    'Long %': analysis['long_prob'],
                    'Risk': analysis['risk_score'],
                    'Sentiment': data.get('sentiment_score', 50),
                    'Whale Activity': len(data.get('whale_trades', []))
                })
        
        # Update paper trades with current prices
        if enable_paper_trading:
            update_paper_trades(current_prices)
        
        if market_data:
            # Convert to DataFrame for display
            df_market = pd.DataFrame(market_data)
            
            # Display with custom formatting
            st.dataframe(
                df_market.style.format({
                    'Price': '${:.4f}',
                    '24h %': '{:+.2f}%',
                    'Volume': '${:,.0f}',
                    'RSI': '{:.1f}',
                    'Long %': '{:.1f}%',
                    'Risk': '{:.0f}',
                    'Sentiment': '{:.0f}',
                    'Whale Activity': '{:.0f}'
                }).background_gradient(subset=['24h %'], cmap='RdYlGn', vmin=-5, vmax=5)
                  .background_gradient(subset=['Risk'], cmap='RdYlGn_r', vmin=0, vmax=100),
                use_container_width=True,
                height=400
            )
        
        # Market statistics
        st.markdown("---")
        stat_cols = st.columns(5)
        
        with stat_cols[0]:
            avg_change = df_market['24h %'].mean()
            st.metric("Avg 24h Change", f"{avg_change:+.2f}%")
        
        with stat_cols[1]:
            total_volume = df_market['Volume'].sum()
            st.metric("Total Volume", f"${total_volume/1e9:.2f}B")
        
        with stat_cols[2]:
            bullish_count = len(df_market[df_market['24h %'] > 0])
            st.metric("Bullish/Total", f"{bullish_count}/{len(df_market)}")
        
        with stat_cols[3]:
            avg_risk = df_market['Risk'].mean()
            st.metric("Avg Risk Score", f"{avg_risk:.1f}")
        
        with stat_cols[4]:
            whale_activity = df_market['Whale Activity'].sum()
            st.metric("Whale Trades", f"{whale_activity}")
        
        # Correlation analysis
        if len(symbols) > 1:
            st.markdown("---")
            st.subheader("🔗 Correlation Analysis")
            
            corr_col1, corr_col2 = st.columns([2, 1])
            
            with corr_col1:
                # Correlation heatmap
                correlation_data = {}
                for symbol in symbols:
                    data = get_enhanced_data(symbol)
                    if data and 'price_history' in data:
                        correlation_data[symbol.replace('USDT', '')] = data['price_history']
                
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
                        textfont={"size": 12},
                        hoverongaps=False,
                        reversescale=False
                    ))
                    
                    fig_corr.update_layout(
                        title="24-Hour Price Correlation Matrix",
                        height=500,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
            
            with corr_col2:
                # Correlation insights
                st.markdown("### 📊 Correlation Insights")
                
                # Find highest correlations
                corr_values = []
                for i in range(len(correlation_matrix)):
                    for j in range(i+1, len(correlation_matrix)):
                        corr_values.append({
                            'pair': f"{correlation_matrix.index[i]}/{correlation_matrix.columns[j]}",
                            'correlation': correlation_matrix.iloc[i, j]
                        })
                
                corr_values.sort(key=lambda x: abs(x['correlation']), reverse=True)
                
                st.markdown("**Strongest Correlations:**")
                for item in corr_values[:3]:
                    emoji = "🟢" if item['correlation'] > 0 else "🔴"
                    st.write(f"{emoji} {item['pair']}: {item['correlation']:.3f}")
                
                # Diversification score
                avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
                diversification_score = (1 - abs(avg_correlation)) * 100
                
                st.markdown("**Portfolio Diversification:**")
                st.progress(diversification_score / 100)
                st.caption(f"Score: {diversification_score:.1f}/100")
                
                if diversification_score > 70:
                    st.success("✅ Well diversified portfolio")
                elif diversification_score > 40:
                    st.warning("⚠️ Moderate diversification")
                else:
                    st.error("❌ Low diversification - high risk")
        
        # Market sentiment overview
        st.markdown("---")
        st.subheader("🎭 Market Sentiment Analysis")
        
        sent_col1, sent_col2 = st.columns([1, 2])
        
        with sent_col1:
            # Overall sentiment pie chart
            bullish = len(df_market[df_market['Sentiment'] > 60])
            bearish = len(df_market[df_market['Sentiment'] < 40])
            neutral = len(df_market) - bullish - bearish
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Bullish', 'Neutral', 'Bearish'],
                values=[bullish, neutral, bearish],
                hole=.4,
                marker_colors=['#00ff88', '#ffaa00', '#ff4444'],
                textfont=dict(size=16)
            )])
            
            fig_pie.update_layout(
                title="Overall Market Sentiment",
                template="plotly_dark",
                height=300,
                showlegend=True
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with sent_col2:
            # Sentiment by symbol
            fig_sent = go.Figure()
            
            symbols_clean = [s.replace('USDT', '') for s in symbols]
            sentiments = [d['Sentiment'] for d in market_data]
            colors = ['green' if s > 60 else 'red' if s < 40 else 'yellow' for s in sentiments]
            
            fig_sent.add_trace(go.Bar(
                x=symbols_clean,
                y=sentiments,
                marker_color=colors,
                text=[f"{s}%" for s in sentiments],
                textposition='auto'
            ))
            
            fig_sent.add_hline(y=50, line_dash="dash", line_color="gray")
            
            fig_sent.update_layout(
                title="Sentiment Score by Symbol",
                xaxis_title="Symbol",
                yaxis_title="Sentiment Score",
                template="plotly_dark",
                height=300
            )
            st.plotly_chart(fig_sent, use_container_width=True)

with tab4:
    st.header("🧪 Strategy Backtesting Laboratory")
    
    if enable_backtesting:
        st.markdown("Test your trading strategies on historical data before risking real money!")
        
        # Strategy configuration
        config_col1, config_col2, config_col3 = st.columns(3)
        
        with config_col1:
            bt_symbol = st.selectbox("Select Symbol:", symbols if symbols else ["BTCUSDT"])
            bt_strategy = st.selectbox("Strategy Type:", 
                                      ["MA Crossover", "RSI Reversal", "Bollinger Bands", 
                                       "MACD Cross", "Supertrend", "Custom"])
        
        with config_col2:
            bt_leverage = st.slider("Backtest Leverage:", 1, 50, 10)
            bt_capital = st.number_input("Starting Capital ($):", 1000, 1000000, 10000)
        
        with config_col3:
            bt_commission = st.slider("Commission (%):", 0.0, 1.0, 0.1)
            bt_slippage = st.slider("Slippage (%):", 0.0, 1.0, 0.05)
        
        # Strategy-specific parameters
        st.markdown("### ⚙️ Strategy Parameters")
        
        param_cols = st.columns(4)
        
        if bt_strategy == "MA Crossover":
            with param_cols[0]:
                short_ma = st.number_input("Short MA:", 3, 50, 9)
            with param_cols[1]:
                long_ma = st.number_input("Long MA:", 10, 200, 21)
            strategy_params = {'short_ma': short_ma, 'long_ma': long_ma}
            
        elif bt_strategy == "RSI Reversal":
            with param_cols[0]:
                rsi_oversold = st.number_input("RSI Oversold:", 10, 40, 30)
            with param_cols[1]:
                rsi_overbought = st.number_input("RSI Overbought:", 60, 90, 70)
            strategy_params = {'rsi_oversold': rsi_oversold, 'rsi_overbought': rsi_overbought}
        
        else:
            strategy_params = {}
        
        # Run backtest button
        if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
            with st.spinner("Running backtest simulation..."):
                # Get data and run backtest
                data = get_enhanced_data(bt_symbol)
                if data:
                    results = run_advanced_backtest(bt_symbol, data, bt_strategy, strategy_params)
                    
                    if results:
                        st.session_state.backtest_results[bt_symbol] = results
                        st.success("✅ Backtest completed successfully!")
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("📊 Backtest Results")
                        
                        # Key metrics
                        metric_cols = st.columns(6)
                        
                        with metric_cols[0]:
                            roi_color = "green" if results['total_return'] > 0 else "red"
                            st.metric("Total Return", 
                                     f"{results['total_return']:+.2f}%",
                                     f"${bt_capital * results['total_return'] / 100:+.2f}")
                        
                        with metric_cols[1]:
                            st.metric("Win Rate", 
                                     f"{results['win_rate']:.1f}%",
                                     f"{results['win_trades']}/{results['total_trades']}")
                        
                        with metric_cols[2]:
                            st.metric("Profit Factor", 
                                     f"{results['profit_factor']:.2f}",
                                     "Profitable" if results['profit_factor'] > 1 else "Unprofitable")
                        
                        with metric_cols[3]:
                            st.metric("Max Drawdown", 
                                     f"{results['max_drawdown']:.1f}%")
                        
                        with metric_cols[4]:
                            st.metric("Sharpe Ratio", 
                                     f"{results['sharpe_ratio']:.2f}")
                        
                        with metric_cols[5]:
                            st.metric("Total Trades", 
                                     results['total_trades'])
                        
                        # Equity curve
                        st.markdown("### 📈 Equity Curve")
                        
                        fig_equity = go.Figure()
                        
                        fig_equity.add_trace(go.Scatter(
                            x=list(range(len(results['equity_curve']))),
                            y=results['equity_curve'],
                            mode='lines',
                            name='Portfolio Value',
                            line=dict(color='cyan', width=2)
                        ))
                        
                        # Add starting capital line
                        fig_equity.add_hline(y=bt_capital, line_dash="dash", 
                                           line_color="gray", annotation_text="Starting Capital")
                        
                        fig_equity.update_layout(
                            title=f"{bt_symbol} - {bt_strategy} Strategy Performance",
                            xaxis_title="Time Period",
                            yaxis_title="Portfolio Value ($)",
                            template="plotly_dark",
                            height=400
                        )
                        st.plotly_chart(fig_equity, use_container_width=True)
                        
                        # Trade visualization
                        if results['trades']:
                            st.markdown("### 🎯 Trade Visualization")
                            
                            prices = data['price_history']
                            timestamps = data['timestamps']
                            
                            fig_trades = go.Figure()
                            
                            # Price line
                            fig_trades.add_trace(go.Scatter(
                                x=timestamps,
                                y=prices,
                                mode='lines',
                                name='Price',
                                line=dict(color='white', width=1)
                            ))
                            
                            # Add trade markers
                            buy_trades = [t for t in results['trades'] if t['type'] in ['open_long', 'close_short']]
                            sell_trades = [t for t in results['trades'] if t['type'] in ['close_long', 'open_short']]
                            
                            if buy_trades:
                                fig_trades.add_trace(go.Scatter(
                                    x=[timestamps[t['index']] for t in buy_trades],
                                    y=[t['price'] for t in buy_trades],
                                    mode='markers',
                                    marker=dict(symbol='triangle-up', size=12, color='green'),
                                    name='Buy'
                                ))
                            
                            if sell_trades:
                                fig_trades.add_trace(go.Scatter(
                                    x=[timestamps[t['index']] for t in sell_trades],
                                    y=[t['price'] for t in sell_trades],
                                    mode='markers',
                                    marker=dict(symbol='triangle-down', size=12, color='red'),
                                    name='Sell'
                                ))
                            
                            fig_trades.update_layout(
                                title="Entry and Exit Points",
                                xaxis_title="Time",
                                yaxis_title="Price",
                                template="plotly_dark",
                                height=400
                            )
                            st.plotly_chart(fig_trades, use_container_width=True)
                        
                        # Trade analysis
                        st.markdown("### 📋 Trade Analysis")
                        
                        trade_analysis_cols = st.columns(3)
                        
                        with trade_analysis_cols[0]:
                            st.markdown("**Winning Trades**")
                            st.write(f"Count: {results['win_trades']}")
                            st.write(f"Avg Win: ${results['avg_win']:.2f}")
                            st.write(f"Win Rate: {results['win_rate']:.1f}%")
                        
                        with trade_analysis_cols[1]:
                            st.markdown("**Losing Trades**")
                            st.write(f"Count: {results['loss_trades']}")
                            st.write(f"Avg Loss: ${results['avg_loss']:.2f}")
                            st.write(f"Loss Rate: {100-results['win_rate']:.1f}%")
                        
                        with trade_analysis_cols[2]:
                            st.markdown("**Risk Metrics**")
                            st.write(f"Max Drawdown: {results['max_drawdown']:.1f}%")
                            st.write(f"Risk/Reward: 1:{results['avg_win']/results['avg_loss']:.1f}" if results['avg_loss'] > 0 else "Risk/Reward: N/A")
                            st.write(f"Expectancy: ${(results['win_rate']/100 * results['avg_win']) - ((100-results['win_rate'])/100 * results['avg_loss']):.2f}")
                    else:
                        st.error("❌ Insufficient data for backtesting. Need at least 20 data points.")
                else:
                    st.error("❌ Failed to fetch data for backtesting.")
    else:
        st.info("📌 Enable backtesting in the sidebar to access this feature.")

with tab5:
    st.header("📰 Market News & Sentiment Analysis")
    
    if show_news:
        news_col1, news_col2 = st.columns([2, 1])
        
        with news_col1:
            st.subheader("📰 Latest Market News")
            
            # Fetch news for selected symbols
            for symbol in symbols[:3]:  # Show news for first 3 symbols
                with st.expander(f"📰 {symbol} News", expanded=True):
                    news_items = fetch_market_news(symbol)
                    
                    for news in news_items:
                        sentiment_emoji = "🟢" if news['sentiment'] == 'positive' else "🔴" if news['sentiment'] == 'negative' else "🟡"
                        
                        st.markdown(f"""
                        <div style="padding: 15px; margin: 10px 0; background: rgba(255,255,255,0.05); 
                                    border-radius: 10px; border-left: 3px solid {'green' if news['sentiment'] == 'positive' else 'red' if news['sentiment'] == 'negative' else 'yellow'};">
                            <h4>{sentiment_emoji} {news['title']}</h4>
                            <p>{news['summary']}</p>
                            <small>📅 {news['time'].strftime('%Y-%m-%d %H:%M')}</small>
                        </div>
                        """, unsafe_allow_html=True)
        
        with news_col2:
            st.subheader("🎭 Sentiment Overview")
            
            # Overall market sentiment
            sentiment_scores = []
            for symbol in symbols:
                data = get_enhanced_data(symbol)
                if data:
                    sentiment_scores.append(data.get('sentiment_score', 50))
            
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 50
            
            # Sentiment gauge
            fig_sentiment = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = avg_sentiment,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Market Sentiment", 'font': {'size': 20}},
                gauge = {
                    'axis': {'range': [None,                alert_msg = f"🎯 **{symbol}**: Price below ${alert['price']:.2f}"
                st.session_state.alerts.append({
                    'time': current_time,
                    'symbol': symbol,
                    'type': 'price_alert',
                    'message': alert_msg,
                    'data': alert
                })
                st.session_state.price_alerts[symbol].remove(alert)

# Risk level function with emoji
def get_risk_level(score):
    if score < 20:
        return "🟢 VERY LOW", "green", "✅"
    elif score < 40:
        return "🟡 LOW", "yellow", "⚠️"
    elif score < 60:
        return "🟠 MEDIUM", "orange", "⚡"
    elif score < 80:
        return "🔴 HIGH", "red", "🚨"
    else:
        return "⚫ EXTREME", "darkred", "☠️"

# Enhanced backtesting with multiple strategies
def run_advanced_backtest(symbol, data, strategy_type, params):
    """Run advanced backtesting with multiple strategies"""
    if not data or 'price_history' not in data:
        return None
    
    prices = np.array(data['price_history'])
    if len(prices) < 20:
        return None
    
    trades = []
    equity_curve = [10000]  # Starting capital
    position = None
    position_size = 0
    
    # Strategy implementation
    if strategy_type == "MA Crossover":
        short_period = params.get('short_ma', 5)
        long_period = params.get('long_ma', 15)
        
        for i in range(long_period, len(prices)):
            short_ma = np.mean(prices[i-short_period:i])
            long_ma = np.mean(prices[i-long_period:i])
            
            if position is None and short_ma > long_ma:
                # Open long
                position = 'long'
                position_size = equity_curve[-1] * 0.95  # Use 95% of capital
                entry_price = prices[i]
                trades.append({
                    'type': 'open_long',
                    'price': entry_price,
                    'index': i,
                    'size': position_size
                })
            elif position == 'long' and short_ma < long_ma:
                # Close long
                exit_price = prices[i]
                pnl = (exit_price - entry_price) / entry_price * position_size
                equity_curve.append(equity_curve[-1] + pnl)
                trades.append({
                    'type': 'close_long',
                    'price': exit_price,
                    'index': i,
                    'pnl': pnl
                })
                position = None
            elif position is None and short_ma < long_ma:
                # Open short
                position = 'short'
                position_size = equity_curve[-1] * 0.95
                entry_price = prices[i]
                trades.append({
                    'type': 'open_short',
                    'price': entry_price,
                    'index': i,
                    'size': position_size
                })
            elif position == 'short' and short_ma > long_ma:
                # Close short
                exit_price = prices[i]
                pnl = (entry_price - exit_price) / entry_price * position_size
                equity_curve.append(equity_curve[-1] + pnl)
                trades.append({
                    'type': 'close_short',
                    'price': exit_price,
                    'index': i,
                    'pnl': pnl
                })
                position = None
            else:
                equity_curve.append(equity_curve[-1])
    
    elif strategy_type == "RSI Reversal":
        rsi_oversold = params.get('rsi_oversold', 30)
        rsi_overbought = params.get('rsi_overbought', 70)
        
        # Calculate RSI for historical data
        rsi_values = []
        for i in range(14, len(prices)):
            gains = []
            losses = []
            for j in range(1, 14):
                change = prices[i-j+1] - prices[i-j]
                if change > 0:
                    gains.append(change)
                else:
                    losses.append(abs(change))
            
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)
        
        for i in range(len(rsi_values)):
            idx = i + 14
            
            if position is None and rsi_values[i] < rsi_oversold:
                # Open long (oversold bounce)
                position = 'long'
                position_size = equity_curve[-1] * 0.95
                entry_price = prices[idx]
                trades.append({
                    'type': 'open_long',
                    'price': entry_price,
                    'index': idx,
                    'size': position_size
                })
            elif position == 'long' and rsi_values[i] > rsi_overbought:
                # Close long
                exit_price = prices[idx]
                pnl = (exit_price - entry_price) / entry_price * position_size
                equity_curve.append(equity_curve[-1] + pnl)
                trades.append({
                    'type': 'close_long',
                    'price': exit_price,
                    'index': idx,
                    'pnl': pnl
                })
                position = None
            else:
                equity_curve.append(equity_curve[-1])
    
    # Calculate performance metrics
    if len(trades) < 2:
        return None
    
    total_return = ((equity_curve[-1] - equity_curve[0]) / equity_curve[0]) * 100
    
    # Count wins and losses
    win_trades = 0
    loss_trades = 0
    profits = []
    losses = []
    
    for trade in trades:
        if 'pnl' in trade:
            if trade['pnl'] > 0:
                win_trades += 1
                profits.append(trade['pnl'])
            else:
                loss_trades += 1
                losses.append(abs(trade['pnl']))
    
    total_trades = win_trades + loss_trades
    win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Calculate additional metrics
    profit_factor = sum(profits) / sum(losses) if losses else float('inf')
    avg_win = np.mean(profits) if profits else 0
    avg_loss = np.mean(losses) if losses else 0
    
    # Maximum drawdown
    peak = equity_curve[0]
    max_drawdown = 0
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # Sharpe ratio (simplified)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 and np.std(returns) > 0 else 0
    
    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'win_trades': win_trades,
        'loss_trades': loss_trades,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'equity_curve': equity_curve,
        'trades': trades
    }

# Fetch market news (simulated)
def fetch_market_news(symbol):
    """Fetch relevant market news for the symbol"""
    # In production, this would connect to a news API
    # For now, return simulated news
    base_news = [
        {
            'title': f"{symbol} Shows Strong Momentum Amid Market Rally",
            'summary': "Technical indicators suggest continued bullish trend...",
            'sentiment': 'positive',
            'time': datetime.now() - timedelta(hours=2)
        },
        {
            'title': f"Whale Activity Detected in {symbol} Trading",
            'summary': "Large transactions observed in the past 24 hours...",
            'sentiment': 'neutral',
            'time': datetime.now() - timedelta(hours=5)
        },
        {
            'title': f"Market Analysis: {symbol} Approaching Key Resistance",
            'summary': "Traders watching critical levels as volume increases...",
            'sentiment': 'neutral',
            'time': datetime.now() - timedelta(hours=8)
        }
    ]
    
    return base_news

# Machine Learning Prediction (simulated)
def get_ml_prediction(symbol, data):
    """Generate ML-based price predictions"""
    if not data:
        return None
    
    current_price = data['current_price']
    
    # Simulate ML predictions based on multiple factors
    factors = {
        'trend': 1.0 if data.get('ma_20', 0) > data.get('ma_50', 0) else -1.0,
        'momentum': 1.0 if data.get('rsi', 50) > 50 else -1.0,
        'volume': 1.0 if data.get('volume_ratio', 1) > 1 else -1.0,
        'sentiment': (data.get('sentiment_score', 50) - 50) / 50
    }
    
    # Weighted prediction
    prediction_factor = (
        factors['trend'] * 0.3 +
        factors['momentum'] * 0.25 +
        factors['volume'] * 0.25 +
        factors['sentiment'] * 0.2
    )
    
    # Generate predictions
    predictions = {
        '1h': current_price * (1 + prediction_factor * 0.005),
        '4h': current_price * (1 + prediction_factor * 0.02),
        '24h': current_price * (1 + prediction_factor * 0.05),
        'confidence': min(95, max(30, 50 + abs(prediction_factor) * 30)),
        'direction': 'Bullish' if prediction_factor > 0 else 'Bearish'
    }
    
    return predictions

# Paper trading functions
def execute_paper_trade(symbol, direction, size, leverage, entry_price):
    """Execute a paper trade"""
    trade = {
        'id': len(st.session_state.trade_history) + 1,
        'symbol': symbol,
        'direction': direction,
        'size': size,
        'leverage': leverage,
        'entry_price': entry_price,
        'entry_time': datetime.now(),
        'status': 'open',
        'pnl': 0,
        'pnl_percent': 0
    }
    
    st.session_state.trade_history.append(trade)
    return trade

def update_paper_trades(current_prices):
    """Update P&L for all open paper trades"""
    for trade in st.session_state.trade_history:
        if trade['status'] == 'open' and trade['symbol'] in current_prices:
            current_price = current_prices[trade['symbol']]
            
            if trade['direction'] == 'Long':
                pnl_percent = ((current_price - trade['entry_price']) / trade['entry_price']) * 100
            else:  # Short
                pnl_percent = ((trade['entry_price'] - current_price) / trade['entry_price']) * 100
            
            trade['pnl_percent'] = pnl_percent * trade['leverage']
            trade['pnl'] = (trade['pnl_percent'] / 100) * trade['size']
            trade['current_price'] = current_price

# Main app interface with tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard", "🔬 Advanced Analysis", "📈 Market Overview", 
    "🧪 Backtesting", "📰 News & Sentiment", "💼 Paper Trading"
])

with tab1:
    st.header("📊 Crypto Trading Dashboard")
    
    # Quick stats row
    if symbols:
        quick_stats_cols = st.columns(4)
        total_market_change = 0
        total_volume = 0
        bullish_count = 0
        
        for symbol in symbols[:4]:  # Show first 4 symbols
            data = get_enhanced_data(symbol)
            if data:
                total_market_change += data['price_change_24h']
                total_volume += data['volume']
                if data['price_change_24h'] > 0:
                    bullish_count += 1
        
        with quick_stats_cols[0]:
            st.metric("📈 Market Trend", 
                     "Bullish" if total_market_change > 0 else "Bearish",
                     f"{total_market_change/len(symbols):.2f}%")
        with quick_stats_cols[1]:
            st.metric("💰 Total Volume", 
                     f"${total_volume/1e6:.1f}M")
        with quick_stats_cols[2]:
            st.metric("🎯 Bullish Ratio", 
                     f"{bullish_count}/{len(symbols)}")
        with quick_stats_cols[3]:
            st.metric("⏰ Last Update", 
                     st.session_state.last_update.strftime("%H:%M:%S"))
    
    # Main dashboard
    if symbols:
        # Alerts panel
        if st.session_state.alerts:
            with st.expander(f"🔔 Active Alerts ({len(st.session_state.alerts[-10:])})", expanded=True):
                for alert in reversed(st.session_state.alerts[-10:]):
                    alert_color = {
                        'signal': 'green',
                        'risk': 'blue',
                        'volume': 'orange',
                        'whale': 'red',
                        'pattern': 'violet',
                        'sr_level': 'yellow'
                    }.get(alert['type'], 'gray')
                    
                    st.markdown(f"""
                    <div style="padding: 10px; margin: 5px 0; border-left: 4px solid {alert_color}; 
                                background-color: rgba(255,255,255,0.05); border-radius: 5px;">
                        <b>{alert['time'].strftime('%H:%M:%S')}</b> - {alert['message']}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Symbol cards
        st.subheader("🪙 Symbol Analysis")
        
        # Create responsive grid
        num_cols = min(3, len(symbols))
        cols = st.columns(num_cols)
        
        for idx, symbol in enumerate(symbols):
            with cols[idx % num_cols]:
                data = get_enhanced_data(symbol)
                if data:
                    analysis = calculate_ai_probabilities(data, leverage, selected_strategy)
                    check_alerts(symbol, data, analysis)
                    
                    # Create card with custom styling
                    with st.container():
                        # Header with symbol and source
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                                    padding: 15px; border-radius: 15px 15px 0 0; text-align: center;">
                            <h2 style="margin: 0; color: white;">{symbol.replace('USDT', '')}</h2>
                            <p style="margin: 0; color: #ccc; font-size: 12px;">{data.get('data_source', 'Unknown')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Price section
                        price_col1, price_col2 = st.columns([2, 1])
                        with price_col1:
                            price_color = "green" if data['price_change_24h'] > 0 else "red"
                            st.metric(
                                "Price", 
                                f"${data['current_price']:,.4f}",
                                delta=f"{data['price_change_24h']:+.2f}%"
                            )
                        with price_col2:
                            vol_millions = data['volume'] / 1e6
                            st.metric(
                                "Volume", 
                                f"${vol_millions:.1f}M",
                                delta=f"{(data['volume_ratio']-1)*100:+.0f}%"
                            )
                        
                        # Sentiment meter
                        sentiment = data.get('sentiment_score', 50)
                        sentiment_color = (
                            "green" if sentiment > 70 else 
                            "yellow" if sentiment > 50 else 
                            "orange" if sentiment > 30 else "red"
                        )
                        
                        st.markdown(f"""
                        <div style="text-align: center; margin: 10px 0;">
                            <b>Market Sentiment</b>
                            <div style="background: #333; border-radius: 20px; height: 20px; margin: 5px 0;">
                                <div style="background: {sentiment_color}; width: {sentiment}%; 
                                          height: 100%; border-radius: 20px; transition: width 0.5s;">
                                </div>
                            </div>
                            <small>{sentiment}% Bullish</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Trading signals with visual bars
                        st.markdown("**📊 Trading Signals**")
                        
                        # Long probability bar
                        long_color = "#00ff88" if analysis['long_prob'] > 60 else "#ff4444" if analysis['long_prob'] < 40 else "#ffaa00"
                        st.markdown(f"""
                        <div style="margin: 5px 0;">
                            <div style="display: flex; justify-content: space-between;">
                                <span>📈 Long</span>
                                <span>{analysis['long_prob']}%</span>
                            </div>
                            <div style="background: #333; border-radius: 10px; height: 10px;">
                                <div style="background: {long_color}; width: {analysis['long_prob']}%; 
                                          height: 100%; border-radius: 10px;">
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Short probability bar
                        short_color = "#00ff88" if analysis['short_prob'] > 60 else "#ff4444" if analysis['short_prob'] < 40 else "#ffaa00"
                        st.markdown(f"""
                        <div style="margin: 5px 0;">
                            <div style="display: flex; justify-content: space-between;">
                                <span>📉 Short</span>
                                <span>{analysis['short_prob']}%</span>
                            </div>
                            <div style="background: #333; border-radius: 10px; height: 10px;">
                                <div style="background: {short_color}; width: {analysis['short_prob']}%; 
                                          height: 100%; border-radius: 10px;">
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Recommendation badge
                        rec_style = {
                            "STRONG LONG": "background: #00ff88; color: black;",
                            "LONG": "background: #00cc66; color: white;",
                            "WEAK LONG": "background: #339933; color: white;",
                            "NEUTRAL": "background: #666666; color: white;",
                            "WEAK SHORT": "background: #cc6666; color: white;",
                            "SHORT": "background: #ff6666; color: white;",
                            "STRONG SHORT": "background: #ff0000; color: white;"
                        }
                        
                        style = rec_style.get(analysis['recommendation'].split()[0] + " " + analysis['recommendation'].split()[1] if len(analysis['recommendation'].split()) > 1 else analysis['recommendation'].split()[0], "background: #666666; color: white;")
                        
                        st.markdown(f"""
                        <div style="text-align: center; margin: 15px 0;">
                            <div style="{style} padding: 10px; border-radius: 25px; 
                                      font-weight: bold; font-size: 16px;">
                                {analysis['recommendation']}
                            </div>
                            <small>Confidence: {analysis['confidence']}%</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Risk meter
                        risk_level, risk_color, risk_emoji = get_risk_level(analysis['risk_score'])
                        st.markdown(f"""
                        <div style="background: {risk_color}; padding: 10px; border-radius: 10px; 
                                  text-align: center; margin: 10px 0;">
                            <b>{risk_emoji} Risk: {risk_level}</b>
                            <div>{analysis['risk_score']}/100</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Key indicators
                        st.markdown("**📈 Key Indicators**")
                        ind_col1, ind_col2 = st.columns(2)
                        
                        with ind_col1:
                            rsi_emoji = "🔴" if data['rsi'] > 70 else "🟢" if data['rsi'] < 30 else "🟡"
                            st.write(f"{rsi_emoji} RSI: {data['rsi']:.1f}")
                            
                            macd_emoji = "🟢" if data['macd'] > 0 else "🔴"
                            st.write(f"{macd_emoji} MACD: {data['macd']:.2f}")
                        
                        with ind_col2:
                            bb_emoji = "🔴" if data['bb_position'] > 80 else "🟢" if data['bb_position'] < 20 else "🟡"
                            st.write(f"{bb_emoji} BB: {data['bb_position']:.0f}%")
                            
                            trend_emoji = "🟢" if data.get('supertrend_bullish', True) else "🔴"
                            st.write(f"{trend_emoji} Trend: {'Bull' if data.get('supertrend_bullish', True) else 'Bear'}")
                        
                        # Patterns detected
                        if data.get('patterns'):
                            st.markdown("**🎯 Patterns Detected**")
                            for pattern in data['patterns'][:2]:  # Show max 2 patterns
                                st.write(f"• {pattern}")
                        
                        # Multi-timeframe trend
                        if 'mtf_analysis' in data:
                            st.markdown("**⏱️ Multi-Timeframe**")
                            mtf_summary = []
                            for tf, analysis in data['mtf_analysis'].items():
                                emoji = "🟢" if analysis['trend'] == 'Bullish' else "🔴"
                                mtf_summary.append(f"{emoji} {tf}")
                            st.write(" | ".join(mtf_summary))
                        
                        # Action buttons
                        st.markdown("---")
                        action_col1, action_col2 = st.columns(2)
                        
                        with action_col1:
                            if st.button(f"📊 Analyze", key=f"analyze_{symbol}", use_container_width=True):
                                st.session_state['selected_symbol'] = symbol
                                st.session_state['active_tab'] = 1
                                st.rerun()
                        
                        with action_col2:
                            if st.button(f"💰 Trade", key=f"trade_{symbol}", use_container_width=True):
                                st.session_state['selected_symbol'] = symbol
                                st.session_state['active_tab'] = 5
                                st.rerun()
                        
                        st.markdown("---")

with tab2:
    st.header("🔬 Advanced Technical Analysis")
    
    if symbols:
        # Symbol selector
        selected_symbol = st.selectbox(
            "Select Symbol for Deep Analysis",
            symbols,
            index=symbols.index(st.session_state.get('selected_symbol', symbols[0])) if st.session_state.get('selected_symbol', symbols[0]) in symbols else 0
        )
        
        data = get_enhanced_data(selected_symbol)
        if data:
            analysis = calculate_ai_probabilities(data, leverage, selected_strategy)
            
            # Header metrics
            met_cols = st.columns(6)
            with met_cols[0]:
                st.metric("Current Price", f"${data['current_price']:,.4f}", 
                         f"{data['price_change_24h']:+.2f}%")
            with met_cols[1]:
                st.metric("24h High", f"${data['high_24h']:,.4f}")
            with met_cols[2]:
                st.metric("24h Low", f"${data['low_24h']:,.4f}")
            with met_cols[3]:
                st.metric("Volume", f"${data['volume']/1e6:.1f}M")
            with met_cols[4]:
                st.metric("ATR", f"${data['atr']:.2f}")
            with met_cols[5]:
                risk_level, _, _ = get_risk_level(analysis['risk_score'])
                st.metric("Risk", risk_level)
            
            # Advanced charts
            chart_col1, chart_col2 = st.columns([2, 1])
            
            with chart_col1:
                # Main price chart with multiple indicators
                fig = make_subplots(
                    rows=4, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.5, 0.15, 0.15, 0.2],
                    subplot_titles=('Price Action', 'RSI', 'MACD', 'Volume')
                )
                
                # Price and moving averages
                fig.add_trace(
                    go.Candlestick(
                        x=data['timestamps'],
                        open=[p - np.random.uniform(0, 50) for p in data['price_history']],
                        high=[p + np.random.uniform(0, 50) for p in data['price_history']],
                        low=[p - np.random.uniform(0, 50) for p in data['price_history']],
                        close=data['price_history'],
                        name='Price'
                    ),
                    row=1, col=1
                )
                
                # Add moving averages
                if len(data['price_history']) >= 7:
                    ma7 = pd.Series(data['price_history']).rolling(window=7).mean()
                    fig.add_trace(
                        go.Scatter(x=data['timestamps'], y=ma7, name='MA7', 
                                 line=dict(color='orange', width=1)),
                        row=1, col=1
                    )
                
                if len(data['price_history']) >= 20:
                    ma20 = pd.Series(data['price_history']).rolling(window=20).mean()
                    fig.add_trace(
                        go.Scatter(x=data['timestamps'], y=ma20, name='MA20', 
                                 line=dict(color='blue', width=1)),
                        row=1, col=1
                    )
                
                # Bollinger Bands
                if 'bb_upper' in data and 'bb_lower' in data:
                    fig.add_trace(
                        go.Scatter(x=data['timestamps'], 
                                 y=[data['bb_upper']] * len(data['timestamps']),
                                 name='BB Upper', line=dict(color='gray', dash='dash')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=data['timestamps'], 
                                 y=[data['bb_lower']] * len(data['timestamps']),
                                 name='BB Lower', line=dict(color='gray', dash='dash')),
                        row=1, col=1
                    )
                
                # RSI
                rsi_values = [data['rsi'] + np.random.uniform(-5, 5) for _ in data['timestamps']]
                fig.add_trace(
                    go.Scatter(x=data['timestamps'], y=rsi_values, name='RSI',
                             line=dict(color='purple')),
                    row=2, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # MACD
                macd_values = [data['macd'] + np.random.uniform(-10, 10) for _ in data['timestamps']]
                signal_values = [data['macd_signal'] + np.random.uniform(-10, 10) for _ in data['timestamps']]
                fig.add_trace(
                    go.Scatter(x=data['timestamps'], y=macd_values, name='MACD',
                             line=dict(color='blue')),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(x=data['timestamps'], y=signal_values, name='Signal',
                             line=dict(color='red')),
                    row=3, col=1
                )
                
                # Volume
                colors = ['green' if i % 2 == 0 else 'red' for i in range(len(data['volume_history']))]
                fig.add_trace(
                    go.Bar(x=data['timestamps'], y=data['volume_history'],
                         name='Volume', marker_color=colors),
                    row=4, col=1
                )
                
                fig.update_layout(
                    title=f"{selected_symbol} Technical Analysis",
                    xaxis_title="Time",
                    height=800,
                    template="plotly_dark",
                    showlegend=True
                )
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
import hashlib
import hmac
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Crypto Trade Analyzer Pro", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Advanced Crypto Trading Analyzer - Professional Edition"
    }
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    /* Dark theme enhancements */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Metric styling */
    .stMetric {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    
    .stMetric > label {
        font-size: 14px !important;
        font-weight: bold !important;
        color: #888 !important;
    }
    
    .stMetric > div {
        font-size: 24px !important;
        font-weight: bold !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #00897b;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #00695c;
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
    }
    
    /* Tab styling */
    .stTabs > div > div {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 10px;
    }
    
    /* Alert styling */
    .stAlert {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
    }
    
    /* Expander styling */
    .stExpander {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 10px;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #1b5e20;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    
    .stError {
        background-color: #c62828;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Animated gradient title */
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main-title {
        background: linear-gradient(90deg, #00897b, #4fc3f7, #00897b);
        background-size: 200% 200%;
        animation: gradient 3s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Animated title
st.markdown('<h1 class="main-title">🚀 Advanced Crypto Trade Analyzer Pro</h1>', unsafe_allow_html=True)

# Initialize enhanced session state
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
if 'price_alerts' not in st.session_state:
    st.session_state.price_alerts = {}
if 'strategy_performance' not in st.session_state:
    st.session_state.strategy_performance = {}
if 'whale_alerts' not in st.session_state:
    st.session_state.whale_alerts = []
if 'news_cache' not in st.session_state:
    st.session_state.news_cache = {}
if 'ml_predictions' not in st.session_state:
    st.session_state.ml_predictions = {}
if 'user_settings' not in st.session_state:
    st.session_state.user_settings = {
        'theme': 'dark',
        'notifications': True,
        'sound_alerts': False,
        'risk_tolerance': 'medium'
    }

# Enhanced sidebar with more features
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Add search functionality
    search_symbol = st.text_input("🔍 Search Symbol", placeholder="Type to search...")
    
    # Enhanced symbol list with categories
    crypto_categories = {
        "🏆 Top Coins": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"],
        "🚀 DeFi": ["UNIUSDT", "LINKUSDT", "AAVEUSDT", "SUSHIUSDT", "CRVUSDT"],
        "🎮 Gaming": ["AXSUSDT", "SANDUSDT", "MANAUSDT", "ENJUSDT", "GALAUSDT"],
        "⚡ Layer 2": ["MATICUSDT", "ARBUSDT", "OPUSDT", "IMXUSDT"],
        "🌟 AI Coins": ["FETUSDT", "AGIXUSDT", "OCEANUSDT", "RENDERUSDT"],
        "💎 Meme Coins": ["DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "FLOKIUSDT"]
    }
    
    # Flatten all symbols for the multiselect
    all_symbols = []
    for category, coins in crypto_categories.items():
        all_symbols.extend(coins)
    
    # Filter symbols based on search
    if search_symbol:
        filtered_symbols = [s for s in all_symbols if search_symbol.upper() in s]
    else:
        filtered_symbols = all_symbols
    
    symbols = st.multiselect(
        "Select symbols to analyze:", 
        filtered_symbols,
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    )
    
    # Quick select buttons
    st.markdown("### 🎯 Quick Select")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Top 5", use_container_width=True):
            symbols = crypto_categories["🏆 Top Coins"]
    with col2:
        if st.button("DeFi", use_container_width=True):
            symbols = crypto_categories["🚀 DeFi"]
    with col3:
        if st.button("AI", use_container_width=True):
            symbols = crypto_categories["🌟 AI Coins"]
    
    leverage = st.slider("Set your leverage (X):", 1, 125, 10)
    
    st.header("🔄 Auto-Refresh Settings")
    auto_refresh = st.checkbox("Enable Auto-Refresh", value=True)
    refresh_rate = st.slider("Refresh every (seconds):", 5, 120, 15)
    
    # Enhanced Alert Settings
    st.header("🔔 Alert Settings")
    enable_alerts = st.checkbox("Enable Smart Alerts", value=True)
    if enable_alerts:
        alert_prob_threshold = st.slider("Signal Strength >", 60, 95, 75)
        alert_risk_threshold = st.slider("Risk Score <", 10, 50, 30)
        alert_volume_spike = st.slider("Volume Spike >", 150, 1000, 300)
        whale_alert_threshold = st.slider("Whale Alert ($K)", 100, 10000, 1000)
        enable_sound = st.checkbox("Enable Sound Alerts", value=False)
    
    # Advanced Settings
    st.header("🔧 Advanced Settings")
    use_ai_predictions = st.checkbox("Enable AI Predictions", value=True)
    show_order_book = st.checkbox("Show Order Book Analysis", value=True)
    enable_backtesting = st.checkbox("Enable Backtesting", value=True)
    show_news = st.checkbox("Show Market News", value=True)
    show_whale_tracking = st.checkbox("Whale Movement Tracking", value=True)
    enable_paper_trading = st.checkbox("Paper Trading Mode", value=True)
    
    # Risk Management Settings
    st.header("🛡️ Risk Management")
    risk_tolerance = st.select_slider(
        "Risk Tolerance",
        options=["Conservative", "Moderate", "Aggressive"],
        value="Moderate"
    )
    max_position_size = st.number_input("Max Position Size ($)", 100, 100000, 1000)
    stop_loss_percent = st.slider("Default Stop Loss %", 1.0, 10.0, 3.0)
    take_profit_percent = st.slider("Default Take Profit %", 2.0, 50.0, 10.0)
    
    # Trading Strategy Selection
    st.header("📊 Trading Strategy")
    selected_strategy = st.selectbox(
        "Select Strategy",
        ["Scalping", "Day Trading", "Swing Trading", "Position Trading", "Custom"]
    )
    
    # Quick Actions
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.session_state.last_update = datetime.now() - timedelta(seconds=refresh_rate)
            st.rerun()
    with col2:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.alerts = []
            st.session_state.trade_history = []
            st.success("Cleared all data!")
    
    # Export/Import Settings
    st.markdown("---")
    if st.button("📊 Export All Data", use_container_width=True):
        export_data = {
            'trades': st.session_state.trade_history,
            'alerts': st.session_state.alerts,
            'settings': st.session_state.user_settings,
            'timestamp': datetime.now().isoformat()
        }
        st.download_button(
            label="💾 Download JSON",
            data=json.dumps(export_data, indent=2, default=str),
            file_name=f"crypto_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

# Auto-refresh logic
if auto_refresh:
    current_time = datetime.now()
    time_diff = (current_time - st.session_state.last_update).total_seconds()
    
    if time_diff >= refresh_rate:
        st.session_state.last_update = current_time
        st.rerun()

# Enhanced data fetching with multiple sources
@st.cache_data(ttl=10)
def get_enhanced_data(symbol):
    """Fetch enhanced data with multiple indicators and sources"""
    
    def try_binance_advanced():
        base_url = "https://fapi.binance.com"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json"
        }
        
        try:
            # Multiple timeframe data
            timeframes = ['5m', '15m', '1h', '4h', '1d']
            multi_timeframe_data = {}
            
            for tf in timeframes:
                klines_response = requests.get(
                    f"{base_url}/fapi/v1/klines", 
                    params={"symbol": symbol, "interval": tf, "limit": 100}, 
                    headers=headers, timeout=8
                )
                if klines_response.status_code == 200:
                    multi_timeframe_data[tf] = klines_response.json()
            
            # Get main timeframe data (1h)
            klines = multi_timeframe_data.get('1h', [])
            
            # Get 24h stats
            stats_response = requests.get(
                f"{base_url}/fapi/v1/ticker/24hr", 
                params={"symbol": symbol}, 
                headers=headers, timeout=8
            )
            
            # Get order book depth
            depth_response = requests.get(
                f"{base_url}/fapi/v1/depth",
                params={"symbol": symbol, "limit": 100},
                headers=headers, timeout=5
            )
            
            # Get recent trades for whale detection
            trades_response = requests.get(
                f"{base_url}/fapi/v1/aggTrades",
                params={"symbol": symbol, "limit": 100},
                headers=headers, timeout=5
            )
            
            if klines and stats_response.status_code == 200:
                stats = stats_response.json()
                depth = depth_response.json() if depth_response.status_code == 200 else None
                trades = trades_response.json() if trades_response.status_code == 200 else []
                
                # Process kline data
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                df = df.astype({
                    'open': float, 'high': float, 'low': float, 
                    'close': float, 'volume': float, 'quote_volume': float
                })
                
                # Calculate advanced technical indicators
                closes = df['close'].values
                highs = df['high'].values
                lows = df['low'].values
                volumes = df['volume'].values
                
                # Price action patterns
                def detect_patterns(df):
                    patterns = []
                    
                    # Bullish patterns
                    if len(df) >= 3:
                        # Hammer
                        last_candle = df.iloc[-1]
                        body = abs(last_candle['close'] - last_candle['open'])
                        lower_shadow = last_candle['open'] - last_candle['low'] if last_candle['close'] > last_candle['open'] else last_candle['close'] - last_candle['low']
                        if lower_shadow > body * 2 and last_candle['high'] - max(last_candle['close'], last_candle['open']) < body * 0.3:
                            patterns.append("🔨 Hammer (Bullish)")
                        
                        # Three White Soldiers
                        if (df['close'].iloc[-3:] > df['open'].iloc[-3:]).all():
                            patterns.append("⚔️ Three White Soldiers (Strong Bullish)")
                    
                    # Bearish patterns
                    if len(df) >= 3:
                        # Shooting Star
                        last_candle = df.iloc[-1]
                        body = abs(last_candle['close'] - last_candle['open'])
                        upper_shadow = last_candle['high'] - last_candle['close'] if last_candle['close'] > last_candle['open'] else last_candle['high'] - last_candle['open']
                        if upper_shadow > body * 2:
                            patterns.append("🌠 Shooting Star (Bearish)")
                    
                    return patterns
                
                patterns = detect_patterns(df)
                
                # Enhanced RSI with divergence detection
                def calculate_rsi_with_divergence(prices, period=14):
                    deltas = np.diff(prices)
                    gains = np.where(deltas > 0, deltas, 0)
                    losses = np.where(deltas < 0, -deltas, 0)
                    
                    avg_gains = pd.Series(gains).rolling(window=period).mean()
                    avg_losses = pd.Series(losses).rolling(window=period).mean()
                    
                    rs = avg_gains / avg_losses
                    rsi = 100 - (100 / (1 + rs))
                    
                    # Detect divergence
                    divergence = "None"
                    if len(rsi) > 20:
                        recent_rsi = rsi.iloc[-10:].values
                        recent_prices = prices[-10:]
                        
                        # Bullish divergence: price making lower lows, RSI making higher lows
                        if recent_prices[-1] < recent_prices[-5] and recent_rsi[-1] > recent_rsi[-5]:
                            divergence = "Bullish Divergence"
                        # Bearish divergence: price making higher highs, RSI making lower highs
                        elif recent_prices[-1] > recent_prices[-5] and recent_rsi[-1] < recent_rsi[-5]:
                            divergence = "Bearish Divergence"
                    
                    return rsi.iloc[-1] if len(rsi) > 0 and not np.isnan(rsi.iloc[-1]) else 50, divergence
                
                current_rsi, rsi_divergence = calculate_rsi_with_divergence(closes)
                
                # Advanced indicators
                def calculate_advanced_indicators(df):
                    indicators = {}
                    
                    # Ichimoku Cloud
                    high_9 = df['high'].rolling(window=9).max()
                    low_9 = df['low'].rolling(window=9).min()
                    indicators['tenkan_sen'] = (high_9 + low_9) / 2
                    
                    high_26 = df['high'].rolling(window=26).max()
                    low_26 = df['low'].rolling(window=26).min()
                    indicators['kijun_sen'] = (high_26 + low_26) / 2
                    
                    # Supertrend
                    atr_period = 10
                    multiplier = 3
                    atr = calculate_atr(highs, lows, closes, atr_period)
                    hl_avg = (df['high'] + df['low']) / 2
                    upper_band = hl_avg + (multiplier * atr)
                    lower_band = hl_avg - (multiplier * atr)
                    
                    supertrend = closes[-1] > upper_band if len(closes) > 0 else False
                    indicators['supertrend_bullish'] = supertrend
                    
                    # VWAP
                    typical_price = (df['high'] + df['low'] + df['close']) / 3
                    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
                    indicators['vwap'] = vwap.iloc[-1] if len(vwap) > 0 else closes[-1]
                    
                    # Market Profile Value Area
                    price_levels = np.linspace(lows.min(), highs.max(), 50)
                    volume_profile = []
                    for price in price_levels:
                        vol_at_price = df[(df['low'] <= price) & (df['high'] >= price)]['volume'].sum()
                        volume_profile.append(vol_at_price)
                    
                    poc_idx = np.argmax(volume_profile)
                    indicators['point_of_control'] = price_levels[poc_idx]
                    
                    return indicators
                
                advanced_indicators = calculate_advanced_indicators(df)
                
                # Stochastic RSI
                def calculate_stoch_rsi(rsi_values, period=14):
                    if len(rsi_values) < period:
                        return 50
                    min_rsi = min(rsi_values[-period:])
                    max_rsi = max(rsi_values[-period:])
                    if max_rsi - min_rsi == 0:
                        return 50
                    return ((rsi_values[-1] - min_rsi) / (max_rsi - min_rsi)) * 100
                
                # Multiple Moving Averages
                ma_periods = [7, 20, 50, 100, 200]
                mas = {}
                for period in ma_periods:
                    mas[f'ma_{period}'] = np.mean(closes[-period:]) if len(closes) >= period else closes[-1]
                
                # MACD with histogram
                ema_12 = pd.Series(closes).ewm(span=12).mean().iloc[-1]
                ema_26 = pd.Series(closes).ewm(span=26).mean().iloc[-1]
                macd = ema_12 - ema_26
                signal_line = pd.Series(closes).ewm(span=9).mean().iloc[-1]
                macd_histogram = macd - signal_line
                
                # Bollinger Bands with squeeze detection
                bb_period = 20
                if len(closes) >= bb_period:
                    bb_sma = np.mean(closes[-bb_period:])
                    bb_std = np.std(closes[-bb_period:])
                    bb_upper = bb_sma + (bb_std * 2)
                    bb_lower = bb_sma - (bb_std * 2)
                    bb_position = (closes[-1] - bb_lower) / (bb_upper - bb_lower) * 100
                    bb_squeeze = bb_std < np.percentile(pd.Series(closes).rolling(window=bb_period).std().dropna(), 20)
                else:
                    bb_position = 50
                    bb_upper = bb_lower = closes[-1]
                    bb_squeeze = False
                
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
                
                # OBV (On Balance Volume)
                obv = []
                obv_value = 0
                for i in range(1, len(closes)):
                    if closes[i] > closes[i-1]:
                        obv_value += volumes[i]
                    elif closes[i] < closes[i-1]:
                        obv_value -= volumes[i]
                    obv.append(obv_value)
                obv_trend = "Bullish" if len(obv) > 10 and obv[-1] > obv[-10] else "Bearish"
                
                # Support and Resistance levels (multiple methods)
                def calculate_support_resistance(highs, lows, closes, method='pivot'):
                    levels = {}
                    
                    # Pivot Points
                    if method == 'pivot':
                        pivot = (highs[-1] + lows[-1] + closes[-1]) / 3
                        levels['pivot'] = pivot
                        levels['r1'] = 2 * pivot - lows[-1]
                        levels['r2'] = pivot + (highs[-1] - lows[-1])
                        levels['r3'] = highs[-1] + 2 * (pivot - lows[-1])
                        levels['s1'] = 2 * pivot - highs[-1]
                        levels['s2'] = pivot - (highs[-1] - lows[-1])
                        levels['s3'] = lows[-1] - 2 * (highs[-1] - pivot)
                    
                    # Fibonacci levels
                    recent_high = max(highs[-20:]) if len(highs) >= 20 else highs[-1]
                    recent_low = min(lows[-20:]) if len(lows) >= 20 else lows[-1]
                    diff = recent_high - recent_low
                    
                    levels['fib_0'] = recent_low
                    levels['fib_236'] = recent_low + 0.236 * diff
                    levels['fib_382'] = recent_low + 0.382 * diff
                    levels['fib_500'] = recent_low + 0.500 * diff
                    levels['fib_618'] = recent_low + 0.618 * diff
                    levels['fib_786'] = recent_low + 0.786 * diff
                    levels['fib_1000'] = recent_high
                    
                    return levels
                
                support_resistance = calculate_support_resistance(highs, lows, closes)
                
                # Order book analysis
                order_book_imbalance = 0
                bid_liquidity = 0
                ask_liquidity = 0
                order_book_levels = {}
                
                if depth:
                    bids = depth.get('bids', [])
                    asks = depth.get('asks', [])
                    
                    # Calculate liquidity
                    for i, bid in enumerate(bids[:20]):
                        bid_liquidity += float(bid[0]) * float(bid[1])
                    for i, ask in enumerate(asks[:20]):
                        ask_liquidity += float(ask[0]) * float(ask[1])
                    
                    total_liquidity = bid_liquidity + ask_liquidity
                    if total_liquidity > 0:
                        order_book_imbalance = (bid_liquidity - ask_liquidity) / total_liquidity * 100
                    
                    # Find significant levels
                    bid_levels = {}
                    ask_levels = {}
                    
                    for bid in bids[:50]:
                        price = float(bid[0])
                        size = float(bid[1])
                        rounded_price = round(price, -int(np.log10(price)) + 2)
                        bid_levels[rounded_price] = bid_levels.get(rounded_price, 0) + size
                    
                    for ask in asks[:50]:
                        price = float(ask[0])
                        size = float(ask[1])
                        rounded_price = round(price, -int(np.log10(price)) + 2)
                        ask_levels[rounded_price] = ask_levels.get(rounded_price, 0) + size
                    
                    # Find walls
                    bid_wall = max(bid_levels.items(), key=lambda x: x[1]) if bid_levels else (0, 0)
                    ask_wall = max(ask_levels.items(), key=lambda x: x[1]) if ask_levels else (0, 0)
                    
                    order_book_levels = {
                        'bid_wall_price': bid_wall[0],
                        'bid_wall_size': bid_wall[1],
                        'ask_wall_price': ask_wall[0],
                        'ask_wall_size': ask_wall[1]
                    }
                
                # Whale detection
                whale_trades = []
                if trades:
                    current_price = closes[-1]
                    for trade in trades:
                        trade_value = float(trade['p']) * float(trade['q'])
                        if trade_value > whale_alert_threshold * 1000:  # Convert to USD
                            whale_trades.append({
                                'price': float(trade['p']),
                                'quantity': float(trade['q']),
                                'value': trade_value,
                                'is_buyer': trade['m'],
                                'time': datetime.fromtimestamp(trade['T']/1000)
                            })
                
                # Get funding rate
                funding_rate = 0.0
                next_funding_time = None
                try:
                    funding_response = requests.get(
                        f"{base_url}/fapi/v1/fundingRate", 
                        params={"symbol": symbol, "limit": 1}, 
                        headers=headers, timeout=5
                    )
                    if funding_response.status_code == 200:
                        funding = funding_response.json()
                        if funding:
                            funding_rate = float(funding[0]['fundingRate']) * 100
                            next_funding_time = datetime.fromtimestamp(funding[0]['fundingTime']/1000)
                except:
                    pass
                
                # Market sentiment calculation
                sentiment_score = 50  # Neutral
                sentiment_factors = []
                
                # Price action sentiment
                if closes[-1] > mas['ma_20']:
                    sentiment_score += 10
                    sentiment_factors.append("Strong ask pressure (-10)")
                
                # Whale sentiment
                if whale_trades:
                    buyer_volume = sum(t['value'] for t in whale_trades if t['is_buyer'])
                    seller_volume = sum(t['value'] for t in whale_trades if not t['is_buyer'])
                    if buyer_volume > seller_volume * 1.5:
                        sentiment_score += 15
                        sentiment_factors.append("Whale accumulation (+15)")
                    elif seller_volume > buyer_volume * 1.5:
                        sentiment_score -= 15
                        sentiment_factors.append("Whale distribution (-15)")
                
                # Normalize sentiment score
                sentiment_score = max(0, min(100, sentiment_score))
                
                # Multi-timeframe analysis
                mtf_analysis = {}
                for tf, tf_data in multi_timeframe_data.items():
                    if tf_data:
                        tf_df = pd.DataFrame(tf_data[:20], columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                            'taker_buy_quote', 'ignore'
                        ])
                        tf_df = tf_df.astype({'close': float, 'open': float})
                        
                        tf_trend = "Bullish" if tf_df['close'].iloc[-1] > tf_df['close'].iloc[0] else "Bearish"
                        tf_strength = abs((tf_df['close'].iloc[-1] - tf_df['close'].iloc[0]) / tf_df['close'].iloc[0] * 100)
                        
                        mtf_analysis[tf] = {
                            'trend': tf_trend,
                            'strength': tf_strength
                        }
                
                # Calculate RSI values for Stoch RSI
                rsi_values = []
                for i in range(14, len(closes)):
                    rsi_val, _ = calculate_rsi_with_divergence(closes[:i+1])
                    rsi_values.append(rsi_val)
                
                stoch_rsi = calculate_stoch_rsi(rsi_values) if rsi_values else 50
                
                # Compile all data
                return {
                    # Basic data
                    "current_price": float(stats.get("lastPrice", 0)),
                    "price_change_24h": float(stats.get("priceChangePercent", 0)),
                    "high_24h": float(stats.get("highPrice", 0)),
                    "low_24h": float(stats.get("lowPrice", 0)),
                    "volume": float(stats.get("volume", 0)),
                    "quote_volume": float(stats.get("quoteVolume", 0)),
                    "volume_ratio": volume_ratio,
                    "volatility": abs(float(stats.get("priceChangePercent", 0))),
                    
                    # Advanced indicators
                    "rsi": current_rsi,
                    "rsi_divergence": rsi_divergence,
                    "stoch_rsi": stoch_rsi,
                    "macd": macd,
                    "macd_signal": signal_line,
                    "macd_histogram": macd_histogram,
                    "bb_position": bb_position,
                    "bb_upper": bb_upper,
                    "bb_lower": bb_lower,
                    "bb_squeeze": bb_squeeze,
                    "atr": atr,
                    "obv_trend": obv_trend,
                    
                    # Moving averages
                    **mas,
                    
                    # Support/Resistance
                    "support_resistance": support_resistance,
                    
                    # Advanced indicators
                    "vwap": advanced_indicators['vwap'],
                    "point_of_control": advanced_indicators['point_of_control'],
                    "supertrend_bullish": advanced_indicators['supertrend_bullish'],
                    
                    # Order book data
                    "order_book_imbalance": order_book_imbalance,
                    "bid_liquidity": bid_liquidity,
                    "ask_liquidity": ask_liquidity,
                    "order_book_levels": order_book_levels,
                    
                    # Whale tracking
                    "whale_trades": whale_trades,
                    
                    # Market data
                    "funding_rate": funding_rate,
                    "next_funding_time": next_funding_time,
                    "long_short_ratio": 1.0,  # Would need separate API
                    
                    # Sentiment
                    "sentiment_score": sentiment_score,
                    "sentiment_factors": sentiment_factors,
                    
                    # Patterns
                    "patterns": patterns,
                    
                    # Multi-timeframe
                    "mtf_analysis": mtf_analysis,
                    
                    # Metadata
                    "last_updated": datetime.now().strftime("%H:%M:%S"),
                    "data_source": "Binance Advanced",
                    "price_history": closes[-24:].tolist(),
                    "volume_history": volumes[-24:].tolist(),
                    "timestamps": [datetime.fromtimestamp(int(k[0])/1000).strftime("%H:%M") for k in klines[-24:]]
                }
        except Exception as e:
            st.warning(f"Advanced API failed for {symbol}: {str(e)}")
            return None
    
    # Try advanced Binance first
    result = try_binance_advanced()
    if result:
        return result
    
    # Fallback to demo data with realistic values
    return {
        "current_price": 50000.0 if "BTC" in symbol else 3000.0 if "ETH" in symbol else 100.0,
        "price_change_24h": np.random.uniform(-5, 5),
        "high_24h": 51000 if "BTC" in symbol else 3100 if "ETH" in symbol else 105,
        "low_24h": 49000 if "BTC" in symbol else 2900 if "ETH" in symbol else 95,
        "volume": np.random.uniform(500000, 2000000),
        "quote_volume": np.random.uniform(25000000, 100000000),
        "volume_ratio": np.random.uniform(0.8, 1.5),
        "volatility": np.random.uniform(1, 5),
        "rsi": np.random.uniform(30, 70),
        "rsi_divergence": "None",
        "stoch_rsi": np.random.uniform(20, 80),
        "ma_7": 50000,
        "ma_20": 50000,
        "ma_50": 49000,
        "ma_100": 48000,
        "ma_200": 47000,
        "macd": np.random.uniform(-100, 100),
        "macd_signal": np.random.uniform(-100, 100),
        "macd_histogram": np.random.uniform(-50, 50),
        "bb_position": np.random.uniform(20, 80),
        "bb_upper": 51000,
        "bb_lower": 49000,
        "bb_squeeze": False,
        "atr": 1000,
        "obv_trend": "Neutral",
        "support_resistance": {
            'pivot': 50000,
            'r1': 51000,
            'r2': 52000,
            's1': 49000,
            's2': 48000
        },
        "vwap": 50000,
        "point_of_control": 50000,
        "supertrend_bullish": True,
        "order_book_imbalance": 0,
        "bid_liquidity": 1000000,
        "ask_liquidity": 1000000,
        "order_book_levels": {},
        "whale_trades": [],
        "funding_rate": 0.01,
        "next_funding_time": datetime.now() + timedelta(hours=4),
        "long_short_ratio": 1.0,
        "sentiment_score": 50,
        "sentiment_factors": ["Demo mode - neutral sentiment"],
        "patterns": ["Demo Pattern"],
        "mtf_analysis": {
            '5m': {'trend': 'Neutral', 'strength': 0.5},
            '15m': {'trend': 'Neutral', 'strength': 0.5},
            '1h': {'trend': 'Neutral', 'strength': 0.5},
            '4h': {'trend': 'Neutral', 'strength': 0.5},
            '1d': {'trend': 'Neutral', 'strength': 0.5}
        },
        "last_updated": datetime.now().strftime("%H:%M:%S"),
        "data_source": "Demo Mode",
        "price_history": [50000] * 24,
        "volume_history": [1000000] * 24,
        "timestamps": [(datetime.now() - timedelta(hours=i)).strftime("%H:%M") for i in range(24, 0, -1)]
    }

# Enhanced AI-powered probability calculation
def calculate_ai_probabilities(data, leverage, strategy="Balanced"):
    """Advanced probability calculation with AI/ML simulation"""
    if not data:
        return {
            "long_prob": 50, "short_prob": 50, "risk_score": 50, 
            "recommendation": "NEUTRAL", "confidence": 0,
            "stop_loss": 0, "take_profit": 0
        }
    
    base_prob = 50
    signals = {}
    weights = {}
    
    # Strategy-specific weights
    if strategy == "Scalping":
        weights = {
            'rsi': 0.15, 'stoch_rsi': 0.15, 'macd': 0.10,
            'bb': 0.15, 'volume': 0.20, 'orderbook': 0.15,
            'mtf': 0.05, 'patterns': 0.05
        }
    elif strategy == "Day Trading":
        weights = {
            'rsi': 0.10, 'stoch_rsi': 0.10, 'macd': 0.15,
            'bb': 0.10, 'volume': 0.15, 'orderbook': 0.10,
            'mtf': 0.15, 'patterns': 0.15
        }
    elif strategy == "Swing Trading":
        weights = {
            'rsi': 0.10, 'stoch_rsi': 0.05, 'macd': 0.20,
            'bb': 0.10, 'volume': 0.10, 'orderbook': 0.05,
            'mtf': 0.25, 'patterns': 0.15
        }
    else:  # Balanced
        weights = {
            'rsi': 0.125, 'stoch_rsi': 0.125, 'macd': 0.125,
            'bb': 0.125, 'volume': 0.125, 'orderbook': 0.125,
            'mtf': 0.125, 'patterns': 0.125
        }
    
    # RSI signal with divergence
    rsi_signal = 0
    if data['rsi'] > 80:
        rsi_signal = -15
    elif data['rsi'] > 70:
        rsi_signal = -10
    elif data['rsi'] < 20:
        rsi_signal = 15
    elif data['rsi'] < 30:
        rsi_signal = 10
    elif data['rsi'] > 60:
        rsi_signal = -5
    elif data['rsi'] < 40:
        rsi_signal = 5
    
    # Add divergence bonus
    if data['rsi_divergence'] == "Bullish Divergence":
        rsi_signal += 10
    elif data['rsi_divergence'] == "Bearish Divergence":
        rsi_signal -= 10
    
    signals['rsi'] = rsi_signal * weights['rsi']
    
    # Stochastic RSI signal
    stoch_signal = 0
    if data['stoch_rsi'] > 80:
        stoch_signal = -10
    elif data['stoch_rsi'] < 20:
        stoch_signal = 10
    signals['stoch_rsi'] = stoch_signal * weights['stoch_rsi']
    
    # MACD signal
    macd_signal = 0
    if data['macd_histogram'] > 0 and data['macd'] > data['macd_signal']:
        macd_signal = 10
        if data['macd_histogram'] > 50:
            macd_signal = 15
    elif data['macd_histogram'] < 0 and data['macd'] < data['macd_signal']:
        macd_signal = -10
        if data['macd_histogram'] < -50:
            macd_signal = -15
    signals['macd'] = macd_signal * weights['macd']
    
    # Bollinger Bands signal
    bb_signal = 0
    if data['bb_position'] > 90:
        bb_signal = -12
    elif data['bb_position'] < 10:
        bb_signal = 12
    elif data['bb_position'] > 75:
        bb_signal = -6
    elif data['bb_position'] < 25:
        bb_signal = 6
    
    # Bollinger squeeze bonus
    if data['bb_squeeze']:
        bb_signal = bb_signal * 1.5  # Amplify signal during squeeze
    
    signals['bb'] = bb_signal * weights['bb']
    
    # Volume analysis
    volume_signal = 0
    if data['volume_ratio'] > 3:
        volume_signal = 12 if data['price_change_24h'] > 0 else -12
    elif data['volume_ratio'] > 2:
        volume_signal = 8 if data['price_change_24h'] > 0 else -8
    elif data['volume_ratio'] > 1.5:
        volume_signal = 5 if data['price_change_24h'] > 0 else -5
    elif data['volume_ratio'] < 0.5:
        volume_signal = -5
    
    # OBV confirmation
    if data['obv_trend'] == "Bullish" and volume_signal > 0:
        volume_signal *= 1.2
    elif data['obv_trend'] == "Bearish" and volume_signal < 0:
        volume_signal *= 1.2
    
    signals['volume'] = volume_signal * weights['volume']
    
    # Order book analysis
    orderbook_signal = 0
    if data['order_book_imbalance'] > 30:
        orderbook_signal = 10
    elif data['order_book_imbalance'] > 15:
        orderbook_signal = 5
    elif data['order_book_imbalance'] < -30:
        orderbook_signal = -10
    elif data['order_book_imbalance'] < -15:
        orderbook_signal = -5
    
    # Check for walls
    if 'order_book_levels' in data and data['order_book_levels']:
        current_price = data['current_price']
        if data['order_book_levels'].get('bid_wall_size', 0) > data['order_book_levels'].get('ask_wall_size', 0) * 2:
            orderbook_signal += 5
        elif data['order_book_levels'].get('ask_wall_size', 0) > data['order_book_levels'].get('bid_wall_size', 0) * 2:
            orderbook_signal -= 5
    
    signals['orderbook'] = orderbook_signal * weights['orderbook']
    
    # Multi-timeframe analysis
    mtf_signal = 0
    if 'mtf_analysis' in data:
        bullish_count = sum(1 for tf, analysis in data['mtf_analysis'].items() if analysis['trend'] == 'Bullish')
        bearish_count = sum(1 for tf, analysis in data['mtf_analysis'].items() if analysis['trend'] == 'Bearish')
        
        if bullish_count >= 4:
            mtf_signal = 15
        elif bullish_count >= 3:
            mtf_signal = 8
        elif bearish_count >= 4:
            mtf_signal = -15
        elif bearish_count >= 3:
            mtf_signal = -8
    
    signals['mtf'] = mtf_signal * weights['mtf']
    
    # Pattern recognition
    pattern_signal = 0
    if 'patterns' in data and data['patterns']:
        for pattern in data['patterns']:
            if "Bullish" in pattern:
                pattern_signal += 10
            elif "Bearish" in pattern:
                pattern_signal -= 10
    
    signals['patterns'] = pattern_signal * weights['patterns']
    
    # AI Enhancement (simulated ML predictions)
    ai_boost = 0
    if use_ai_predictions:
        # Momentum analysis
        if len(data.get('price_history', [])) >= 3:
            recent_prices = data['price_history'][-3:]
            momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100
            
            if abs(momentum) > 2:
                ai_boost += 5 if momentum > 0 else -5
        
        # Sentiment integration
        sentiment = data.get('sentiment_score', 50)
        if sentiment > 70:
            ai_boost += 8
        elif sentiment < 30:
            ai_boost -= 8
        
        # Whale activity
        if data.get('whale_trades'):
            buyer_whales = sum(1 for t in data['whale_trades'] if t['is_buyer'])
            seller_whales = len(data['whale_trades']) - buyer_whales
            if buyer_whales > seller_whales:
                ai_boost += 7
            elif seller_whales > buyer_whales:
                ai_boost -= 7
        
        signals['ai'] = ai_boost
    
    # Support/Resistance proximity
    sr_signal = 0
    if 'support_resistance' in data:
        current_price = data['current_price']
        sr = data['support_resistance']
        
        # Check proximity to levels
        for level_name, level_price in sr.items():
            if 'r' in level_name:  # Resistance
                if abs(current_price - level_price) / current_price < 0.005:  # Within 0.5%
                    sr_signal -= 5
            elif 's' in level_name:  # Support
                if abs(current_price - level_price) / current_price < 0.005:
                    sr_signal += 5
    
    signals['support_resistance'] = sr_signal
    
    # Funding rate bias
    funding_signal = 0
    if data['funding_rate'] > 0.1:
        funding_signal = -10
    elif data['funding_rate'] > 0.05:
        funding_signal = -5
    elif data['funding_rate'] < -0.1:
        funding_signal = 10
    elif data['funding_rate'] < -0.05:
        funding_signal = 5
    signals['funding'] = funding_signal
    
    # Calculate total signal
    total_signal = sum(signals.values())
    
    # Apply leverage penalty
    leverage_penalty = min(25, (leverage - 1) * 0.5)
    
    # Calculate probabilities
    long_probability = base_prob + total_signal - leverage_penalty
    short_probability = base_prob - total_signal - leverage_penalty
    
    # Normalize probabilities
    long_probability = max(5, min(95, long_probability))
    short_probability = max(5, min(95, short_probability))
    
    # Risk calculation
    current_price = data['current_price']
    atr_risk = (data['atr'] / current_price) * 100 * leverage if current_price > 0 else 0
    
    risk_factors = []
    risk_score = 0
    
    # Volatility risk
    volatility_risk = data['volatility'] * leverage * 0.15
    risk_score += volatility_risk
    risk_factors.append(f"Volatility: {volatility_risk:.1f}")
    
    # Leverage risk
    leverage_risk = (leverage - 1) * 1.5
    risk_score += leverage_risk
    risk_factors.append(f"Leverage: {leverage_risk:.1f}")
    
    # ATR risk
    atr_risk_score = atr_risk * 0.5
    risk_score += atr_risk_score
    risk_factors.append(f"ATR: {atr_risk_score:.1f}")
    
    # Funding risk
    funding_risk = abs(data['funding_rate']) * 10
    risk_score += funding_risk
    risk_factors.append(f"Funding: {funding_risk:.1f}")
    
    # Liquidity risk
    volume_risk = max(0, (1000000 - data['volume']) / 100000) if data['volume'] < 1000000 else 0
    risk_score += volume_risk
    risk_factors.append(f"Liquidity: {volume_risk:.1f}")
    
    risk_score = min(100, round(risk_score, 1))
    
    # Generate recommendation
    prob_diff = abs(long_probability - short_probability)
    confidence = round(prob_diff, 1)
    
    if prob_diff < 5:
        recommendation = "NEUTRAL ⚖️"
        direction = "Wait"
    elif long_probability > short_probability:
        if prob_diff > 30:
            recommendation = "STRONG LONG 🚀🚀🚀"
        elif prob_diff > 20:
            recommendation = "LONG 🚀🚀"
        elif prob_diff > 10:
            recommendation = "WEAK LONG 🚀"
        else:
            recommendation = "NEUTRAL LONG 📈"
        direction = "Long"
    else:
        if prob_diff > 30:
            recommendation = "STRONG SHORT 📉📉📉"
        elif prob_diff > 20:
            recommendation = "SHORT 📉📉"
        elif prob_diff > 10:
            recommendation = "WEAK SHORT 📉"
        else:
            recommendation = "NEUTRAL SHORT 📉"
        direction = "Short"
    
    # Calculate dynamic stop loss and take profit
    atr = data.get('atr', current_price * 0.01)
    
    if direction == "Long":
        stop_loss = current_price - (atr * 2.5)
        take_profit = current_price + (atr * 4)
    elif direction == "Short":
        stop_loss = current_price + (atr * 2.5)
        take_profit = current_price - (atr * 4)
    else:
        stop_loss = current_price * 0.97
        take_profit = current_price * 1.03
    
    # Risk-adjusted position sizing
    if risk_score < 30:
        position_size_percent = 100
    elif risk_score < 50:
        position_size_percent = 75
    elif risk_score < 70:
        position_size_percent = 50
    else:
        position_size_percent = 25
    
    return {
        "long_prob": round(long_probability, 1),
        "short_prob": round(short_probability, 1),
        "risk_score": risk_score,
        "risk_factors": risk_factors,
        "recommendation": recommendation,
        "direction": direction,
        "confidence": confidence,
        "signals": signals,
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "position_size_percent": position_size_percent,
        "entry_price": current_price
    }

# Enhanced alert system
def check_alerts(symbol, data, analysis):
    """Advanced alert system with multiple alert types"""
    if not enable_alerts:
        return
    
    current_time = datetime.now()
    
    # Signal strength alert
    max_prob = max(analysis['long_prob'], analysis['short_prob'])
    if max_prob >= alert_prob_threshold:
        direction = "Long" if analysis['long_prob'] > analysis['short_prob'] else "Short"
        alert_msg = f"🎯 **{symbol}**: Strong {direction} signal detected!"
        alert_data = {
            'time': current_time,
            'symbol': symbol,
            'type': 'signal',
            'message': alert_msg,
            'data': {
                'probability': max_prob,
                'direction': direction,
                'price': data['current_price'],
                'confidence': analysis['confidence']
            }
        }
        
        # Check if similar alert was already sent recently
        recent_alerts = [a for a in st.session_state.alerts[-10:] 
                        if a['symbol'] == symbol and a['type'] == 'signal' 
                        and (current_time - a['time']).seconds < 300]
        
        if not recent_alerts:
            st.session_state.alerts.append(alert_data)
            if enable_sound:
                st.balloons()  # Visual alert
    
    # Low risk alert
    if analysis['risk_score'] <= alert_risk_threshold:
        alert_msg = f"🛡️ **{symbol}**: Low risk opportunity detected!"
        alert_data = {
            'time': current_time,
            'symbol': symbol,
            'type': 'risk',
            'message': alert_msg,
            'data': {
                'risk_score': analysis['risk_score'],
                'price': data['current_price']
            }
        }
        
        recent_alerts = [a for a in st.session_state.alerts[-10:] 
                        if a['symbol'] == symbol and a['type'] == 'risk' 
                        and (current_time - a['time']).seconds < 300]
        
        if not recent_alerts:
            st.session_state.alerts.append(alert_data)
    
    # Volume spike alert
    if data['volume_ratio'] * 100 >= alert_volume_spike:
        alert_msg = f"📊 **{symbol}**: Massive volume spike detected!"
        alert_data = {
            'time': current_time,
            'symbol': symbol,
            'type': 'volume',
            'message': alert_msg,
            'data': {
                'volume_ratio': data['volume_ratio'],
                'volume': data['volume'],
                'price': data['current_price']
            }
        }
        
        recent_alerts = [a for a in st.session_state.alerts[-10:] 
                        if a['symbol'] == symbol and a['type'] == 'volume' 
                        and (current_time - a['time']).seconds < 600]
        
        if not recent_alerts:
            st.session_state.alerts.append(alert_data)
    
    # Whale alert
    if data.get('whale_trades') and show_whale_tracking:
        for whale in data['whale_trades'][-5:]:  # Check last 5 whale trades
            if whale['value'] >= whale_alert_threshold * 1000:
                trade_type = "BUY" if whale['is_buyer'] else "SELL"
                alert_msg = f"🐋 **{symbol}**: Whale {trade_type} ${whale['value']:,.0f}"
                alert_data = {
                    'time': current_time,
                    'symbol': symbol,
                    'type': 'whale',
                    'message': alert_msg,
                    'data': whale
                }
                
                # Check if this specific whale trade was already alerted
                recent_whale_alerts = [a for a in st.session_state.whale_alerts 
                                     if a['data']['time'] == whale['time']]
                
                if not recent_whale_alerts:
                    st.session_state.whale_alerts.append(alert_data)
                    st.session_state.alerts.append(alert_data)
                    if enable_sound:
                        st.snow()  # Special effect for whale alerts
    
    # Pattern alert
    if data.get('patterns'):
        for pattern in data['patterns']:
            alert_msg = f"📊 **{symbol}**: Pattern detected - {pattern}"
            alert_data = {
                'time': current_time,
                'symbol': symbol,
                'type': 'pattern',
                'message': alert_msg,
                'data': {
                    'pattern': pattern,
                    'price': data['current_price']
                }
            }
            
            recent_alerts = [a for a in st.session_state.alerts[-20:] 
                           if a['symbol'] == symbol and a['type'] == 'pattern' 
                           and a['data']['pattern'] == pattern
                           and (current_time - a['time']).seconds < 3600]
            
            if not recent_alerts:
                st.session_state.alerts.append(alert_data)
    
    # Support/Resistance alert
    if 'support_resistance' in data:
        current_price = data['current_price']
        sr = data['support_resistance']
        
        for level_name, level_price in sr.items():
            if abs(current_price - level_price) / current_price < 0.01:  # Within 1%
                level_type = "Resistance" if 'r' in level_name else "Support"
                alert_msg = f"💪 **{symbol}**: Approaching {level_type} at ${level_price:.2f}"
                alert_data = {
                    'time': current_time,
                    'symbol': symbol,
                    'type': 'sr_level',
                    'message': alert_msg,
                    'data': {
                        'level_type': level_type,
                        'level_price': level_price,
                        'current_price': current_price
                    }
                }
                
                recent_alerts = [a for a in st.session_state.alerts[-10:] 
                               if a['symbol'] == symbol and a['type'] == 'sr_level' 
                               and abs(a['data']['level_price'] - level_price) < 10
                               and (current_time - a['time']).seconds < 1800]
                
                if not recent_alerts:
                    st.session_state.alerts.append(alert_data)
    
    # Price alerts (user-defined)
    if symbol in st.session_state.price_alerts:
        for alert in st.session_state.price_alerts[symbol]:
            if alert['type'] == 'above' and data['current_price'] > alert['price']:
                alert_msg = f"🎯 **{symbol}**: Price above ${alert['price']:.2f}"
                st.session_state.alerts.append({
                    'time': current_time,
                    'symbol': symbol,
                    'type': 'price_alert',
                    'message': alert_msg,
                    'data': alert
                })
                st.session_state.price_alerts[symbol].remove(alert)
            elif alert['type'] == 'below' and data['current_price'] < alert['price']:
                alert_msg = f"append("Price above MA20 (+10)")
                else:
                    sentiment_score -= 10
                    sentiment_factors.append("Price below MA20 (-10)")
                
                # RSI sentiment
                if current_rsi < 30:
                    sentiment_score += 15
                    sentiment_factors.append("Oversold RSI (+15)")
                elif current_rsi > 70:
                    sentiment_score -= 15
                    sentiment_factors.append("Overbought RSI (-15)")
                
                # Volume sentiment
                if volume_ratio > 1.5:
                    if closes[-1] > closes[-2]:
                        sentiment_score += 10
                        sentiment_factors.append("High volume on green candle (+10)")
                    else:
                        sentiment_score -= 10
                        sentiment_factors.append("High volume on red candle (-10)")
                
                # Order book sentiment
                if order_book_imbalance > 20:
                    sentiment_score += 10
                    sentiment_factors.append("Strong bid pressure (+10)")
                elif order_book_imbalance < -20:
                    sentiment_score -= 10
                    sentiment_factors.
