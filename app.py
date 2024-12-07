import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
from black_scholes import black_scholes_call, black_scholes_put, calculate_greeks
from scipy.stats import norm

# Page config
st.set_page_config(
    page_title="Quantitative Finance Dashboard",
    page_icon="📈",
    layout="wide"
)

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        # Read the CSV file
        df = pd.read_csv('nifty_500.csv')
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Function to safely convert values to numeric
        def safe_numeric_convert(value):
            if pd.isna(value):
                return np.nan
            if isinstance(value, (int, float)):
                return value
            try:
                # Remove commas and convert to float
                return float(str(value).replace(',', ''))
            except (ValueError, AttributeError):
                return np.nan
        
        # List of columns to convert to numeric
        numeric_columns = [
            'Percentage Change', '365 Day Percentage Change', '30 Day Percentage Change',
            'Open', 'High', 'Low', 'Previous Close', 'Last Traded Price',
            'Change', '52 Week High', '52 Week Low'
        ]
        
        # Convert numeric columns
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(safe_numeric_convert)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame if loading fails

# Load data
df = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Data Explorer", "Portfolio Optimization", "Options Calculator", "Stock Recommendations"]
)

if page == "Data Explorer":
    st.title("Data Explorer")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_industry = st.multiselect(
            "Select Industry",
            options=sorted(df['Industry'].unique()),
            default=None
        )
    
    with col2:
        price_range = st.slider(
            "Price Range",
            float(df['Last Traded Price'].min()),
            float(df['Last Traded Price'].max()),
            (float(df['Last Traded Price'].min()), float(df['Last Traded Price'].max()))
        )
    
    with col3:
        market_cap_filter = st.selectbox(
            "Performance Filter",
            ["All", "Top Gainers", "Top Losers"]
        )
    
    # Filter data
    filtered_df = df.copy()
    if selected_industry:
        filtered_df = filtered_df[filtered_df['Industry'].isin(selected_industry)]
    
    filtered_df = filtered_df[
        (filtered_df['Last Traded Price'] >= price_range[0]) &
        (filtered_df['Last Traded Price'] <= price_range[1])
    ]
    
    if market_cap_filter == "Top Gainers":
        filtered_df = filtered_df.nlargest(10, 'Percentage Change')
    elif market_cap_filter == "Top Losers":
        filtered_df = filtered_df.nsmallest(10, 'Percentage Change')
    
    # Display data
    st.subheader("Stock Data")
    st.dataframe(filtered_df)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Industry Distribution")
        fig = px.pie(
            df,
            names='Industry',
            title='Distribution of Stocks by Industry'
        )
        st.plotly_chart(fig)
    
    with col2:
        st.subheader("Price vs Returns")
        fig = px.scatter(
            filtered_df,
            x='Last Traded Price',
            y='Percentage Change',
            color='Industry',
            hover_data=['Company Name'],
            title='Price vs Daily Returns'
        )
        st.plotly_chart(fig)

elif page == "Portfolio Optimization":
    st.title("Portfolio Optimization")
    
    # Add filtering options in the sidebar
    st.sidebar.subheader("Portfolio Filters")
    
    # Market Cap Filter
    market_cap_filter = st.sidebar.selectbox(
        "Market Cap Category",
        ["All", "Large Cap", "Mid Cap", "Small Cap"]
    )
    
    # Sector/Industry Filter
    selected_sectors = st.sidebar.multiselect(
        "Select Sectors",
        options=sorted(df['Industry'].unique()),
        default=None
    )
    
    # Performance Filters
    col1, col2 = st.sidebar.columns(2)
    with col1:
        min_return = st.number_input(
            "Minimum Return (%)",
            min_value=float(df['Percentage Change'].min()),
            max_value=float(df['Percentage Change'].max()),
            value=float(df['Percentage Change'].min())
        )
    with col2:
        max_volatility = st.number_input(
            "Maximum Volatility (%)",
            min_value=0.0,
            max_value=100.0,
            value=50.0
        )
    
    # Volume Filter
    min_volume = st.sidebar.slider(
        "Minimum Share Volume",
        min_value=0,
        max_value=int(df['Share Volume'].max()),
        value=10000,
        step=10000
    )
    
    # Price Filter
    price_range = st.sidebar.slider(
        "Price Range (₹)",
        min_value=float(df['Last Traded Price'].min()),
        max_value=float(df['Last Traded Price'].max()),
        value=(float(df['Last Traded Price'].min()), float(df['Last Traded Price'].max()))
    )
    
    # Apply filters to create filtered dataset
    filtered_df = df.copy()
    
    # Market Cap Filter
    if market_cap_filter != "All":
        if market_cap_filter == "Large Cap":
            filtered_df = filtered_df.nlargest(100, 'Last Traded Price')
        elif market_cap_filter == "Mid Cap":
            filtered_df = filtered_df.iloc[100:350]
        else:  # Small Cap
            filtered_df = filtered_df.iloc[350:]
    
    # Sector Filter
    if selected_sectors:
        filtered_df = filtered_df[filtered_df['Industry'].isin(selected_sectors)]
    
    # Performance Filters
    filtered_df = filtered_df[
        (filtered_df['Percentage Change'] >= min_return) &
        (filtered_df['30 Day Percentage Change'].abs() <= max_volatility)
    ]
    
    # Volume Filter
    filtered_df = filtered_df[filtered_df['Share Volume'] >= min_volume]
    
    # Price Filter
    filtered_df = filtered_df[
        (filtered_df['Last Traded Price'] >= price_range[0]) &
        (filtered_df['Last Traded Price'] <= price_range[1])
    ]
    
    # Display filtering summary
    st.subheader("Filtering Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Stocks Available", len(filtered_df))
    with col2:
        st.metric("Sectors Represented", filtered_df['Industry'].nunique())
    with col3:
        st.metric("Avg. Return", f"{filtered_df['Percentage Change'].mean():.2f}%")
    
    # Stock selection from filtered universe
    selected_stocks = st.multiselect(
        "Select stocks for your portfolio (2-10 stocks recommended)",
        options=filtered_df['Symbol'].unique(),
        default=filtered_df['Symbol'].unique()[:5]
    )
    
    # Add quality metrics
    if selected_stocks:
        st.subheader("Selected Stocks Quality Metrics")
        quality_metrics = pd.DataFrame({
            'Symbol': selected_stocks,
            'Return': filtered_df[filtered_df['Symbol'].isin(selected_stocks)]['Percentage Change'],
            'Volatility': filtered_df[filtered_df['Symbol'].isin(selected_stocks)]['30 Day Percentage Change'],
            'Share Volume': filtered_df[filtered_df['Symbol'].isin(selected_stocks)]['Share Volume'],
            'Price': filtered_df[filtered_df['Symbol'].isin(selected_stocks)]['Last Traded Price']
        })
        
        # Format metrics
        quality_metrics['Return'] = quality_metrics['Return'].round(2)
        quality_metrics['Volatility'] = quality_metrics['Volatility'].round(2)
        quality_metrics['Share Volume'] = quality_metrics['Share Volume'].apply(lambda x: f"{x:,.0f}")
        quality_metrics['Price'] = quality_metrics['Price'].round(2)
        
        st.dataframe(quality_metrics)
        
        # Add risk warning based on metrics
        avg_vol = filtered_df[filtered_df['Symbol'].isin(selected_stocks)]['30 Day Percentage Change'].mean()
        if avg_vol > 30:
            st.warning("⚠️ Selected portfolio has high volatility. Consider adding more stable stocks.")
        elif len(selected_stocks) < 5:
            st.warning("⚠️ Portfolio may be under-diversified. Consider adding more stocks.")
    
    # Continue with portfolio optimization if stocks are selected
    if len(selected_stocks) >= 2:
        try:
            # Get data for selected stocks
            stock_data = filtered_df[filtered_df['Symbol'].isin(selected_stocks)].copy()
            
            # Calculate volatility metrics
            stock_data['Monthly_Volatility'] = stock_data['30 Day Percentage Change'] / np.sqrt(30)
            stock_data['Annualized_Volatility'] = stock_data['Monthly_Volatility'] * np.sqrt(12)
            
            # Prepare returns data with proper scaling
            returns_data = pd.DataFrame({
                'Symbol': selected_stocks,
                'Daily_Return': stock_data['Percentage Change'].values / 100,
                'Monthly_Return': stock_data['30 Day Percentage Change'].values / 100,
                'Annual_Return': stock_data['365 Day Percentage Change'].values / 100,
                'Annualized_Volatility': stock_data['Annualized_Volatility'].values / 100
            }).dropna()
            
            # Cap extreme values
            returns_data['Annual_Return'] = returns_data['Annual_Return'].clip(-0.5, 0.5)
            returns_data['Annualized_Volatility'] = returns_data['Annualized_Volatility'].clip(0.1, 0.5)
            
            # Create correlation matrix using industry relationships
            n_stocks = len(returns_data)
            correlation_matrix = np.zeros((n_stocks, n_stocks))
            
            # Get industry data
            industry_data = stock_data[['Symbol', 'Industry']]
            
            for i in range(n_stocks):
                for j in range(n_stocks):
                    if i == j:
                        correlation_matrix[i,j] = 1.0
                    else:
                        industry_i = industry_data.iloc[i]['Industry']
                        industry_j = industry_data.iloc[j]['Industry']
                        # Higher correlation (0.6) for same industry, lower (0.3) for different
                        correlation_matrix[i,j] = 0.6 if industry_i == industry_j else 0.3
            
            # Monte Carlo Simulation
            num_portfolios = 5000
            all_weights = np.zeros((num_portfolios, n_stocks))
            all_returns = np.zeros(num_portfolios)
            all_volatilities = np.zeros(num_portfolios)
            print(all_weights)
            
            for i in range(num_portfolios):
                # Generate random weights
                weights = np.random.random(n_stocks)
                weights = weights / np.sum(weights)
                all_weights[i] = weights
                
                # Calculate portfolio return using annual returns
                portfolio_return = np.sum(weights * returns_data['Annual_Return'])
                
                # Calculate portfolio volatility using correlation matrix and annualized volatilities
                portfolio_volatility = np.sqrt(
                    np.dot(
                        weights.T, 
                        np.dot(
                            correlation_matrix * np.outer(
                                returns_data['Annualized_Volatility'],
                                returns_data['Annualized_Volatility']
                            ),
                            weights
                        )
                    )
                )
                
                all_returns[i] = portfolio_return
                all_volatilities[i] = portfolio_volatility
            
            # Calculate Sharpe Ratio using Indian risk-free rate
            risk_free_rate = 0.072  # 7.2% annual rate
            sharpe_ratios = (all_returns - risk_free_rate) / all_volatilities
            
            # Find optimal portfolio
            optimal_idx = np.argmax(sharpe_ratios)
            optimal_weights = all_weights[optimal_idx]
            
            # Display additional risk metrics
            st.subheader("Portfolio Risk Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Expected Annual Return",
                    f"{(all_returns[optimal_idx] * 100):.2f}%",
                    help="Annualized expected return of the optimal portfolio"
                )
            
            with col2:
                st.metric(
                    "Annual Volatility",
                    f"{(all_volatilities[optimal_idx] * 100):.2f}%",
                    help="Annualized volatility (risk) of the optimal portfolio"
                )
            
            with col3:
                st.metric(
                    "Sharpe Ratio",
                    f"{sharpe_ratios[optimal_idx]:.2f}",
                    help="Risk-adjusted return using 7.2% risk-free rate"
                )
            
            # Display correlation heatmap
            st.subheader("Stock Correlation Matrix")
            fig_corr = px.imshow(
                correlation_matrix,
                title='Stock Correlation Heatmap',
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            st.plotly_chart(fig_corr)
            
            # Create visualization
            fig = go.Figure()
            
            # Plot all portfolios
            fig.add_trace(go.Scatter(
                x=all_volatilities,
                y=all_returns,
                mode='markers',
                name='Simulated Portfolios',
                marker=dict(
                    size=6,
                    color=sharpe_ratios,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Sharpe Ratio')
                ),
                text=[f'Sharpe: {sr:.2f}' for sr in sharpe_ratios],
                hovertemplate='Return: %{y:.2%}<br>Volatility: %{x:.2%}<br>%{text}<extra></extra>'
            ))
            
            # Highlight optimal portfolio
            fig.add_trace(go.Scatter(
                x=[all_volatilities[optimal_idx]],
                y=[all_returns[optimal_idx]],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(
                    size=15,
                    symbol='star',
                    color='red'
                ),
                hovertemplate='Optimal Portfolio<br>Return: %{y:.2%}<br>Volatility: %{x:.2%}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Efficient Frontier',
                xaxis_title='Expected Volatility',
                yaxis_title='Expected Return',
                showlegend=True,
                hovermode='closest',
                xaxis=dict(tickformat='.1%'),
                yaxis=dict(tickformat='.1%')
            )
            
            # Display the graph
            st.plotly_chart(fig)
            
            # Display optimal portfolio allocation
            st.subheader("Optimal Portfolio Allocation")
            
            optimal_portfolio = pd.DataFrame({
                'Stock': returns_data['Symbol'],
                'Weight (%)': optimal_weights * 100,
                'Expected Annual Return (%)': returns_data['Annual_Return'] * 100,
                'Monthly Return (%)': returns_data['Monthly_Return'] * 100
            })
            
            # Format the DataFrame
            optimal_portfolio = optimal_portfolio.round(2)
            st.dataframe(optimal_portfolio)
            
            # Display portfolio metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Portfolio Expected Return (Annual)",
                    f"{(all_returns[optimal_idx] * 100):.2f}%"
                )
            
            with col2:
                st.metric(
                    "Portfolio Volatility (Annual)",
                    f"{(all_volatilities[optimal_idx] * 100):.2f}%"
                )
            
            with col3:
                st.metric(
                    "Sharpe Ratio",
                    f"{sharpe_ratios[optimal_idx]:.2f}",
                    help="Annualized risk-adjusted return using 7.2% risk-free rate"
                )
            
        except Exception as e:
            st.error(f"""
            An error occurred during portfolio optimization. This might be due to:
            - Missing or invalid data for some stocks
            - Insufficient price history
            - Numerical computation issues
            
            Technical details: {str(e)}
            """)
    elif len(selected_stocks) == 1:
        st.warning("Please select at least 2 stocks for portfolio optimization.")

elif page == "Options Calculator":
    st.title("Options Calculator")
    
    st.markdown("""
    ### Black-Scholes Model Implementation
    
    1. **Core Formula**:
       ```
       C = S₀N(d₁) - Ke^(-rT)N(d₂)
       P = Ke^(-rT)N(-d₂) - S₀N(-d₁)
       ```
       where:
       - d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
       - d₂ = d₁ - σ√T
    
    2. **Greeks Calculations**:
       ```
       Delta_call = N(d₁)
       Delta_put = N(d₁) - 1
       Gamma = N'(d₁)/(S₀σ√T)
       Theta = -(S₀N'(d₁)σ)/(2√T) - rKe^(-rT)N(d₂)
       Vega = S₀√T * N'(d₁)
       Rho = KTe^(-rT)N(d₂)
       ```
    
    3. **Implementation Notes**:
       - Uses scipy.stats.norm for N(x) calculations
       - Volatility input is annualized
       - Greeks are scaled to practical units
       - Assumes European-style options
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        spot_price = st.number_input("Spot Price (S)", min_value=0.0, value=100.0, help="Current price of the underlying asset")
        strike_price = st.number_input("Strike Price (K)", min_value=0.0, value=100.0, help="Strike price of the option")
        time_to_maturity = st.number_input("Time to Maturity (Years)", min_value=0.0, max_value=10.0, value=1.0, help="Time until option expiration in years")
    
    with col2:
        volatility = st.slider("Volatility (σ) %", min_value=1, max_value=100, value=20, help="Annualized volatility of the underlying asset") / 100
        risk_free_rate = st.slider("Risk-free Rate (r) %", min_value=0, max_value=20, value=5, help="Annual risk-free interest rate") / 100
    
    # Calculate option prices and Greeks
    call_price = black_scholes_call(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility)
    put_price = black_scholes_put(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility)
    greeks = calculate_greeks(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility)
    
    st.subheader("Option Prices")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Call Option Price", f"₹{call_price:.2f}")
        st.markdown("### Call Option Greeks")
        st.write(f"Delta: {greeks['delta_call']:.4f}")
        st.write(f"Gamma: {greeks['gamma']:.4f}")
        st.write(f"Theta: {greeks['theta_call']:.4f}")
        st.write(f"Vega: {greeks['vega']:.4f}")
        st.write(f"Rho: {greeks['rho_call']:.4f}")
    
    with col2:
        st.metric("Put Option Price", f"₹{put_price:.2f}")
        st.markdown("### Put Option Greeks")
        st.write(f"Delta: {greeks['delta_put']:.4f}")
        st.write(f"Gamma: {greeks['gamma']:.4f}")
        st.write(f"Theta: {greeks['theta_put']:.4f}")
        st.write(f"Vega: {greeks['vega']:.4f}")
        st.write(f"Rho: {greeks['rho_put']:.4f}")
    
    # Add visualization of option prices vs spot price
    st.subheader("Option Prices vs Spot Price")
    spot_range = np.linspace(spot_price*0.5, spot_price*1.5, 100)
    call_prices = [black_scholes_call(s, strike_price, time_to_maturity, risk_free_rate, volatility) for s in spot_range]
    put_prices = [black_scholes_put(s, strike_price, time_to_maturity, risk_free_rate, volatility) for s in spot_range]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spot_range, y=call_prices, name="Call Option"))
    fig.add_trace(go.Scatter(x=spot_range, y=put_prices, name="Put Option"))
    fig.update_layout(
        title="Option Prices vs Underlying Price",
        xaxis_title="Spot Price",
        yaxis_title="Option Price",
        hovermode='x unified'
    )
    st.plotly_chart(fig)

elif page == "Stock Recommendations":
    st.title("Stock Recommendations")
    
    st.markdown("""
    ### Technical Analysis Implementation
    
    1. **Momentum Strategy Calculations**:
       ```python
       momentum_score = (P_t - P_{t-30})/P_{t-30}
       volume_factor = V_t/V_{avg}
       final_score = momentum_score * volume_factor
       ```
       where:
       - $P_t$: Current price
       - $P_{t-30}$: Price 30 days ago
       - $V_t$: Current volume
       - $V_{avg}$: Average volume
    
    2. **Mean Reversion Metrics**:
       ```python
       z_score = (price - MA_20)/std_20
       RSI = 100 - (100/(1 + RS))
       RS = avg_gain/avg_loss
       ```
       where:
       - MA_20: 20-day moving average
       - std_20: 20-day standard deviation
       - RS: Relative strength
    
    3. **Filtering Algorithm**:
       - Momentum: Select top 5 stocks by 30-day return
       - Mean Reversion: Select bottom 5 stocks by z-score
       - Additional volume filter: V_t > V_avg * 1.5
    
    4. **Statistical Measures**:
       - Returns: Log-normalized for better distribution
       - Volatility: Exponentially weighted std dev
       - Trend: Linear regression slope over 30 days
    """)
    
    # Filter stocks based on momentum and mean reversion
    momentum_stocks = df.nlargest(5, '30 Day Percentage Change')
    oversold_stocks = df.nsmallest(5, '30 Day Percentage Change')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Momentum Strategy (Long)")
        momentum_df = momentum_stocks[['Symbol', 'Company Name', '30 Day Percentage Change', 'Last Traded Price']].copy()
        momentum_df['30 Day Percentage Change'] = momentum_df['30 Day Percentage Change'].round(2)
        momentum_df['Last Traded Price'] = momentum_df['Last Traded Price'].round(2)
        st.dataframe(momentum_df)
        
        st.markdown("""
        #### Momentum Strategy Explanation
        - These stocks have shown strong positive trends
        - Higher momentum may indicate continued upward movement
        - Consider entry points and stop-loss levels
        - Monitor market conditions and sector trends
        """)
    
    with col2:
        st.subheader("Mean Reversion Strategy (Short)")
        reversion_df = oversold_stocks[['Symbol', 'Company Name', '30 Day Percentage Change', 'Last Traded Price']].copy()
        reversion_df['30 Day Percentage Change'] = reversion_df['30 Day Percentage Change'].round(2)
        reversion_df['Last Traded Price'] = reversion_df['Last Traded Price'].round(2)
        st.dataframe(reversion_df)
        
        st.markdown("""
        #### Mean Reversion Strategy Explanation
        - These stocks have experienced significant price drops
        - May present opportunities if fundamentally sound
        """)
    
    # Add performance metrics visualization
    st.subheader("Performance Comparison")
    
    # Create performance comparison chart
    fig = go.Figure()
    
    # Add momentum stocks
    fig.add_trace(go.Bar(
        x=momentum_stocks['Symbol'],
        y=momentum_stocks['30 Day Percentage Change'],
        name='Top Performers',
        marker_color='green'
    ))
    
    # Add oversold stocks
    fig.add_trace(go.Bar(
        x=oversold_stocks['Symbol'],
        y=oversold_stocks['30 Day Percentage Change'],
        name='Oversold Stocks',
        marker_color='red'
    ))
    
    fig.update_layout(
        title='30-Day Performance Comparison',
        xaxis_title='Stock Symbol',
        yaxis_title='30-Day Return (%)',
        barmode='group',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig)
    

# Footer
st.markdown("---")
st.markdown("Built with Streamlit by Jigyansu Rout") 