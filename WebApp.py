# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Finance Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class FinanceDashboard:
    def __init__(self):
        self.api_key = "demo"  # Using free demo API key
        self.base_url = "https://www.alphavantage.co/query"
        
    def fetch_stock_data(self, symbol):
        """Fetch stock data from Alpha Vantage API"""
        try:
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'compact'
            }
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
                time_series = data['Time Series (Daily)']
                df = pd.DataFrame.from_dict(time_series, orient='index')
                df = df.astype(float)
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                return df
            else:
                st.error(f"Error fetching data for {symbol}")
                return None
                
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return None
    
    def get_crypto_data(self):
        """Get sample cryptocurrency data"""
        crypto_data = {
            'Bitcoin': [45000, 45500, 44800, 45200, 46000, 45800, 46500],
            'Ethereum': [3200, 3250, 3180, 3220, 3300, 3280, 3350],
            'Cardano': [1.2, 1.25, 1.18, 1.22, 1.30, 1.28, 1.35]
        }
        dates = pd.date_range(start='2024-01-01', periods=7, freq='D')
        return pd.DataFrame(crypto_data, index=dates)

def main():
    # Initialize dashboard
    dashboard = FinanceDashboard()
    
    # Header
    st.markdown('<h1 class="main-header">💰 Personal Finance Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation & Controls")
        
        # Radio button for main view selection
        selected_view = st.radio(
            "Choose Dashboard View:",
            ["Stock Analysis", "Portfolio Overview", "Expense Tracking", "Market Insights"]
        )
        
        st.markdown("---")
        st.subheader("Stock Data Settings")
        
        # Selectbox for stock selection
        stock_symbol = st.selectbox(
            "Select Stock Symbol:",
            ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META"]
        )
        
        # Date input for analysis period
        start_date = st.date_input(
            "Start Date:",
            value=datetime.now() - timedelta(days=30)
        )
        
        # Color picker for charts
        chart_color = st.color_picker(
            "Choose Chart Color:",
            "#1f77b4"
        )
        
        # File uploader for personal data
        uploaded_file = st.file_uploader(
            "Upload Expense CSV:",
            type=['csv']
        )
        
        # Multiselect for crypto selection
        crypto_selection = st.multiselect(
            "Select Cryptocurrencies:",
            ["Bitcoin", "Ethereum", "Cardano", "Solana", "Polkadot"],
            default=["Bitcoin", "Ethereum"]
        )
        
        # Slider for risk tolerance
        risk_tolerance = st.slider(
            "Risk Tolerance:",
            min_value=1,
            max_value=10,
            value=5
        )
        
        # Button to refresh data
        if st.button("🔄 Refresh Data"):
            st.success("Data refreshed successfully!")
            time.sleep(1)
            st.rerun()

    # Main content area
    if selected_view == "Stock Analysis":
        display_stock_analysis(dashboard, stock_symbol, chart_color)
    elif selected_view == "Portfolio Overview":
        display_portfolio_overview(dashboard, crypto_selection)
    elif selected_view == "Expense Tracking":
        display_expense_tracking(uploaded_file)
    else:
        display_market_insights(dashboard, risk_tolerance)

def display_stock_analysis(dashboard, symbol, color):
    """Display stock analysis view"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"📈 {symbol} Stock Analysis")
        
        # Fetch and display stock data
        with st.spinner("Fetching stock data..."):
            stock_data = dashboard.fetch_stock_data(symbol)
            
        if stock_data is not None:
            # Line chart
            st.subheader("Price Movement")
            chart_data = stock_data[['4. close']].rename(columns={'4. close': 'Close Price'})
            st.line_chart(chart_data)
            
            # Area chart for volume
            st.subheader("Trading Volume")
            volume_data = stock_data[['5. volume']].rename(columns={'5. volume': 'Volume'})
            st.area_chart(volume_data)
            
    with col2:
        st.subheader("Quick Metrics")
        
        if stock_data is not None:
            latest_data = stock_data.iloc[-1]
            prev_data = stock_data.iloc[-2]
            
            price_change = latest_data['4. close'] - prev_data['4. close']
            change_percent = (price_change / prev_data['4. close']) * 100
            
            st.metric(
                label="Current Price",
                value=f"${latest_data['4. close']:.2f}",
                delta=f"{change_percent:.2f}%"
            )
            
            st.metric(
                label="Daily Volume",
                value=f"{latest_data['5. volume']:,.0f}"
            )
            
            st.metric(
                label="Daily Range",
                value=f"${latest_data['3. low']:.2f} - ${latest_data['2. high']:.2f}"
            )
        
        # Information box
        st.info("💡 Stock data is updated daily. Use the refresh button to get latest data.")

def display_portfolio_overview(dashboard, crypto_selection):
    """Display portfolio overview"""
    st.header("📊 Portfolio Overview")
    
    # Sample portfolio data
    portfolio_data = {
        'Asset': ['Stocks', 'Bonds', 'Cryptocurrency', 'Cash', 'Real Estate'],
        'Value': [50000, 25000, 15000, 10000, 75000],
        'Allocation': [30, 15, 9, 6, 45]
    }
    portfolio_df = pd.DataFrame(portfolio_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Interactive dataframe
        st.subheader("Portfolio Allocation")
        edited_df = st.data_editor(
            portfolio_df,
            use_container_width=True,
            num_rows="dynamic"
        )
        
        # Bar chart
        st.subheader("Asset Allocation")
        st.bar_chart(edited_df.set_index('Asset')['Allocation'])
    
    with col2:
        # Pie chart using plotly
        st.subheader("Portfolio Distribution")
        fig = px.pie(
            edited_df, 
            values='Value', 
            names='Asset',
            title="Portfolio Value Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Crypto performance
        if crypto_selection:
            st.subheader("Cryptocurrency Performance")
            crypto_data = dashboard.get_crypto_data()
            selected_crypto_data = crypto_data[crypto_selection]
            st.line_chart(selected_crypto_data)

def display_expense_tracking(uploaded_file):
    """Display expense tracking functionality"""
    st.header("💳 Expense Tracking")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Expense input form
        with st.form("expense_form"):
            st.subheader("Add New Expense")
            
            expense_date = st.date_input("Date")
            expense_amount = st.number_input("Amount ($)", min_value=0.0, step=0.01)
            expense_category = st.selectbox(
                "Category",
                ["Food", "Transportation", "Entertainment", "Utilities", "Shopping", "Healthcare"]
            )
            expense_description = st.text_input("Description")
            
            submitted = st.form_submit_button("Add Expense")
            if submitted:
                st.success("✅ Expense added successfully!")
    
    with col2:
        # Sample expense data
        expense_data = {
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
            'Amount': [45.50, 23.75, 120.00, 15.99],
            'Category': ['Food', 'Transportation', 'Shopping', 'Entertainment'],
            'Description': ['Lunch', 'Bus fare', 'Clothes', 'Movie']
        }
        expense_df = pd.DataFrame(expense_data)
        
        st.subheader("Recent Expenses")
        st.dataframe(expense_df, use_container_width=True)
        
        # Monthly summary
        st.subheader("Monthly Summary")
        monthly_summary = expense_df.groupby('Category')['Amount'].sum().reset_index()
        st.bar_chart(monthly_summary.set_index('Category'))

def display_market_insights(dashboard, risk_tolerance):
    """Display market insights and maps"""
    st.header("🌍 Market Insights")
    
    # Warning box for high risk
    if risk_tolerance > 7:
        st.warning("⚠️ High risk tolerance selected. Consider diversifying your portfolio.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Financial Institutions Map")
        
        # Sample financial institutions data
        financial_data = pd.DataFrame({
            'lat': [40.7128, 34.0522, 41.8781, 29.7604, 39.9526],
            'lon': [-74.0060, -118.2437, -87.6298, -95.3698, -75.1652],
            'name': ['NY Bank', 'LA Financial', 'Chicago Trust', 'Houston Capital', 'Philly Funds'],
            'size': [100, 80, 60, 40, 30]
        })
        
        # Map display
        st.map(financial_data)
        
        # 3D map with pydeck
        st.subheader("Global Financial Centers")
        try:
            import pydeck as pdk
            
            layer = pdk.Layer(
                'ScatterplotLayer',
                financial_data,
                get_position=['lon', 'lat'],
                get_color=[255, 0, 0, 160],
                get_radius='size',
                radius_scale=100,
            )
            
            view_state = pdk.ViewState(
                latitude=39.8283,
                longitude=-98.5795,
                zoom=3,
                pitch=50,
            )
            
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={'text': '{name}\nSize: {size}'}
            )
            
            st.pydeck_chart(deck)
        except ImportError:
            st.info("Install pydeck for 3D maps: pip install pydeck")
    
    with col2:
        st.subheader("Economic Indicators")
        
        # Sample economic data
        economic_data = {
            'Indicator': ['GDP Growth', 'Inflation Rate', 'Unemployment', 'Interest Rate'],
            'Current': [2.1, 3.2, 3.7, 5.25],
            'Previous': [2.3, 3.1, 3.8, 5.25],
            'Trend': ['↓', '↑', '↓', '→']
        }
        economic_df = pd.DataFrame(economic_data)
        
        st.dataframe(economic_df, use_container_width=True)
        
        # Checkbox for detailed view
        show_details = st.checkbox("Show Detailed Analysis")
        if show_details:
            st.subheader("Market Analysis")
            st.write("""
            Based on current economic indicators:
            - GDP growth shows slight moderation
            - Inflation remains above target
            - Labor market remains strong
            - Interest rates are stable
            """)
            
            # Progress bar for market sentiment
            st.subheader("Market Sentiment")
            sentiment = 65  # Example value
            st.progress(sentiment / 100)
            st.write(f"Current Market Sentiment: {sentiment}% Positive")

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.caption("Built with Streamlit")
with footer_col2:
    st.caption("Data provided by Alpha Vantage API")
with footer_col3:
    st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()