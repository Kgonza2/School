# app.py - Streamlit Finance Dashboard (No External Dependencies Needed)
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import json
from io import StringIO

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
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #1565c0;
    }
    .dataframe {
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)


class FinanceDashboard:
    def __init__(self):
        # Sample data generation - no API needed
        pass

    def generate_stock_data(self, symbol, days=30):
        """Generate realistic stock data"""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # Realistic base prices for popular stocks
        base_prices = {
            "AAPL": 185.00, "GOOGL": 142.50, "MSFT": 372.00,
            "TSLA": 238.00, "AMZN": 152.75, "META": 348.50,
            "NVDA": 495.00, "NFLX": 485.00
        }

        base_price = base_prices.get(symbol, 100.00)

        # Generate price movements with realistic volatility
        daily_returns = np.random.normal(0.001, 0.02, days)

        prices = []
        price = base_price
        for r in daily_returns:
            price = max(price * (1 + r), base_price * 0.8)  # Prevent price dropping too low
            prices.append(price)

        # Generate volume data
        avg_volume = {
            "AAPL": 50000000, "GOOGL": 25000000, "MSFT": 30000000,
            "TSLA": 100000000, "AMZN": 40000000, "META": 20000000,
            "NVDA": 50000000, "NFLX": 30000000
        }

        base_volume = avg_volume.get(symbol, 10000000)
        volumes = np.random.normal(base_volume, base_volume * 0.3, days)
        volumes = np.abs(volumes).astype(int)

        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Open': [p * 0.995 for p in prices],
            'High': [p * 1.015 for p in prices],
            'Low': [p * 0.985 for p in prices],
            'Close': prices,
            'Volume': volumes
        })

        df.set_index('Date', inplace=True)
        return df

    def generate_crypto_data(self):
        """Generate cryptocurrency data"""
        crypto_data = {
            'BTC': {'name': 'Bitcoin', 'price': 45230.50, 'change': 2.3, 'market_cap': 885000000000},
            'ETH': {'name': 'Ethereum', 'price': 2415.75, 'change': 1.8, 'market_cap': 290000000000},
            'ADA': {'name': 'Cardano', 'price': 0.52, 'change': 0.5, 'market_cap': 18500000000},
            'SOL': {'name': 'Solana', 'price': 98.25, 'change': 3.2, 'market_cap': 42000000000},
            'DOT': {'name': 'Polkadot', 'price': 7.85, 'change': -0.3, 'market_cap': 10000000000},
            'XRP': {'name': 'Ripple', 'price': 0.62, 'change': 0.8, 'market_cap': 33500000000}
        }
        return crypto_data

    def generate_portfolio_data(self):
        """Generate portfolio data"""
        portfolio = {
            'Stocks': {'value': 78500, 'allocation': 42.3, 'return': 8.2},
            'Bonds': {'value': 35200, 'allocation': 19.0, 'return': 3.1},
            'Crypto': {'value': 25150, 'allocation': 13.6, 'return': 15.5},
            'Real Estate': {'value': 45000, 'allocation': 24.3, 'return': 5.2},
            'Cash': {'value': 15830, 'allocation': 8.5, 'return': 1.5}
        }
        return portfolio

    def generate_expense_data(self):
        """Generate expense data"""
        categories = ['Food', 'Transport', 'Entertainment', 'Utilities', 'Shopping', 'Healthcare']
        expenses = []

        for i in range(20):
            date = datetime.now() - timedelta(days=np.random.randint(1, 30))
            category = np.random.choice(categories)
            amount = np.random.uniform(10, 500)

            expenses.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Category': category,
                'Description': f"{category} expense #{i + 1}",
                'Amount': round(amount, 2),
                'Payment': np.random.choice(['Credit Card', 'Debit Card', 'Cash', 'Bank Transfer'])
            })

        return pd.DataFrame(expenses)


def display_header():
    """Display application header"""
    st.markdown('<h1 class="main-header">💰 Personal Finance Dashboard</h1>',
                unsafe_allow_html=True)
    st.caption("Track, Analyze, and Optimize Your Financial Portfolio")


def create_sidebar():
    """Create sidebar with controls"""
    with st.sidebar:
        st.header("🎯 Dashboard Controls")

        # User greeting with name input
        user_name = st.text_input("Enter your name:", value="Financial Analyst")

        if user_name:
            st.success(f"Welcome back, {user_name}!")

        st.markdown("---")

        # Main navigation with radio buttons
        selected_view = st.radio(
            "**Select View:**",
            ["📊 Portfolio Overview", "📈 Stock Analysis", "💎 Crypto Tracker",
             "💳 Expense Manager", "🌍 Market Insights"],
            index=0
        )

        st.markdown("---")

        # Portfolio settings expander
        with st.expander("⚙️ Settings"):
            # Number input
            investment_amount = st.number_input(
                "Monthly Investment ($):",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100
            )

            # Select slider
            risk_level = st.select_slider(
                "Risk Tolerance:",
                options=["Very Low", "Low", "Medium", "High", "Very High"],
                value="Medium"
            )

            # Color picker
            theme_color = st.color_picker(
                "Chart Color Theme:",
                "#1f77b4"
            )

            # Checkbox
            show_details = st.checkbox("Show Detailed Analysis", value=True)

        st.markdown("---")

        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.success("Data refreshed!")
                time.sleep(1)
                st.rerun()

        with col2:
            if st.button("📊 Report", use_container_width=True):
                st.info("Generating report...")

        st.markdown("---")

        # Quick metrics
        st.subheader("Your Metrics")
        st.metric("Portfolio Value", "$185,430", "+2.3%")
        st.metric("Monthly Return", "+$2,850", "+1.5%")

        # Progress bar
        st.progress(65)
        st.caption("Annual Goal: 65% Complete")

    return selected_view, theme_color


def display_portfolio_overview(dashboard):
    """Display portfolio overview"""
    st.header("📊 Portfolio Overview")

    # Get portfolio data
    portfolio = dashboard.generate_portfolio_data()

    # Metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Value", "$185,430", "+2.3%")

    with col2:
        st.metric("Daily Change", "+$850", "+0.46%")

    with col3:
        st.metric("YTD Return", "$12,450", "+7.2%")

    with col4:
        st.metric("Dividends", "$1,250", "+5%")

    # Main content
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Asset Allocation")

        # Create portfolio DataFrame
        portfolio_df = pd.DataFrame([
            {'Asset': asset, 'Value ($)': data['value'],
             'Allocation (%)': data['allocation'], 'Return (%)': data['return']}
            for asset, data in portfolio.items()
        ])

        # Interactive data editor
        edited_df = st.data_editor(
            portfolio_df,
            use_container_width=True,
            column_config={
                "Value ($)": st.column_config.NumberColumn(
                    format="$%d"
                ),
                "Allocation (%)": st.column_config.ProgressColumn(
                    format="%.1f%%",
                    min_value=0,
                    max_value=100
                ),
                "Return (%)": st.column_config.NumberColumn(
                    format="%.1f%%"
                )
            },
            hide_index=True
        )

        # Tabs for different visualizations
        tab1, tab2 = st.tabs(["📈 Bar Chart", "📊 Data"])

        with tab1:
            # Bar chart using Streamlit
            st.subheader("Asset Returns")
            chart_data = edited_df[['Asset', 'Return (%)']].set_index('Asset')
            st.bar_chart(chart_data)

        with tab2:
            st.subheader("Detailed Allocation")
            st.dataframe(edited_df, use_container_width=True)

    with col_right:
        st.subheader("Quick Actions")

        # Action buttons
        if st.button("🔄 Rebalance Portfolio", use_container_width=True):
            st.success("Portfolio rebalanced successfully!")

        if st.button("💰 Add Funds", use_container_width=True):
            st.info("Fund transfer initiated...")

        if st.button("📋 Tax Summary", use_container_width=True):
            st.info("Generating tax summary...")

        st.markdown("---")

        # Performance chart
        st.subheader("Performance Trend")

        # Generate performance data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        performance = np.cumsum(np.random.normal(500, 200, 30)) + 180000

        perf_df = pd.DataFrame({
            'Date': dates,
            'Portfolio Value': performance
        }).set_index('Date')

        st.line_chart(perf_df)

        st.markdown("---")

        # Investment suggestions
        st.subheader("💡 Suggestions")
        st.info("Consider increasing your bond allocation for better risk management.")
        st.warning("Your crypto allocation is above recommended levels.")


def display_stock_analysis(dashboard):
    """Display stock analysis"""
    st.header("📈 Stock Analysis")

    # Stock selection
    stock_symbol = st.selectbox(
        "Select Stock:",
        ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "NFLX"],
        index=0
    )

    # Time period selection
    period = st.radio(
        "Time Period:",
        ["1 Month", "3 Months", "6 Months", "1 Year"],
        horizontal=True
    )

    # Get stock data
    days_map = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365}
    stock_data = dashboard.generate_stock_data(stock_symbol, days_map[period])

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    latest = stock_data.iloc[-1]
    prev = stock_data.iloc[-2] if len(stock_data) > 1 else latest
    change = latest['Close'] - prev['Close']
    change_pct = (change / prev['Close']) * 100

    with col1:
        st.metric(
            "Current Price",
            f"${latest['Close']:.2f}",
            f"{change_pct:+.2f}%"
        )

    with col2:
        st.metric(
            "Today's Range",
            f"${latest['Low']:.2f}-${latest['High']:.2f}"
        )

    with col3:
        st.metric(
            "Volume",
            f"{latest['Volume']:,.0f}"
        )

    with col4:
        avg_volume = stock_data['Volume'].mean()
        st.metric(
            "Avg Volume",
            f"{avg_volume:,.0f}"
        )

    # Charts
    st.subheader("Price Movement")

    # Line chart for closing prices
    price_chart = stock_data[['Close']].rename(columns={'Close': 'Price'})
    st.line_chart(price_chart)

    # Area chart for volume
    st.subheader("Trading Volume")
    volume_chart = stock_data[['Volume']]
    st.area_chart(volume_chart)

    # Additional analysis
    with st.expander("📊 Technical Analysis"):
        col_a, col_b = st.columns(2)

        with col_a:
            # Calculate simple moving averages
            stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
            stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()

            st.write("**Moving Averages**")
            st.write(f"20-Day MA: ${stock_data['MA20'].iloc[-1]:.2f}")
            st.write(f"50-Day MA: ${stock_data['MA50'].iloc[-1]:.2f}")

            if stock_data['Close'].iloc[-1] > stock_data['MA20'].iloc[-1]:
                st.success("Price above 20-Day MA (Bullish)")
            else:
                st.warning("Price below 20-Day MA (Bearish)")

        with col_b:
            # Calculate volatility
            returns = stock_data['Close'].pct_change()
            volatility = returns.std() * np.sqrt(252) * 100

            st.write("**Volatility Analysis**")
            st.write(f"Annual Volatility: {volatility:.2f}%")

            if volatility > 30:
                st.error("High volatility - High risk")
            elif volatility > 20:
                st.warning("Moderate volatility - Medium risk")
            else:
                st.success("Low volatility - Low risk")

    # Stock comparison
    st.subheader("📊 Compare Stocks")
    compare_stocks = st.multiselect(
        "Select stocks to compare:",
        ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "NFLX"],
        default=["AAPL", "GOOGL", "MSFT"]
    )

    if compare_stocks:
        comparison_data = pd.DataFrame()

        for symbol in compare_stocks:
            data = dashboard.generate_stock_data(symbol, 30)
            comparison_data[symbol] = data['Close']

        st.line_chart(comparison_data)

        # Calculate correlations
        st.write("**Correlation Matrix**")
        correlation = comparison_data.corr()
        st.dataframe(correlation.style.format("{:.2f}").background_gradient(cmap='Blues'))


def display_crypto_tracker(dashboard):
    """Display cryptocurrency tracker"""
    st.header("💎 Cryptocurrency Tracker")

    # Get crypto data
    crypto_data = dashboard.generate_crypto_data()

    # Display top cryptocurrencies
    st.subheader("Top Cryptocurrencies")

    # Create DataFrame
    crypto_list = []
    for symbol, data in crypto_data.items():
        crypto_list.append({
            'Symbol': symbol,
            'Name': data['name'],
            'Price ($)': data['price'],
            '24h Change (%)': data['change'],
            'Market Cap ($B)': data['market_cap'] / 1e9
        })

    crypto_df = pd.DataFrame(crypto_list)

    # Interactive table
    st.dataframe(
        crypto_df,
        column_config={
            "Price ($)": st.column_config.NumberColumn(format="$%.2f"),
            "24h Change (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Market Cap ($B)": st.column_config.NumberColumn(format="$%.2fB")
        },
        use_container_width=True,
        hide_index=True
    )

    # Crypto metrics
    col1, col2, col3 = st.columns(3)

    total_market_cap = sum([data['market_cap'] for data in crypto_data.values()]) / 1e12
    avg_change = np.mean([data['change'] for data in crypto_data.values()])

    with col1:
        st.metric("Total Market Cap", f"${total_market_cap:.2f}T")

    with col2:
        st.metric("24h Avg Change", f"{avg_change:+.2f}%")

    with col3:
        # Find top performer
        top_crypto = max(crypto_data.items(), key=lambda x: x[1]['change'])
        st.metric("Top Performer", top_crypto[0], f"+{top_crypto[1]['change']}%")

    # Historical chart
    st.subheader("Historical Trends")

    # Generate historical data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    historical = pd.DataFrame({
        'BTC': np.random.normal(45000, 2000, 30).cumsum(),
        'ETH': np.random.normal(2400, 150, 30).cumsum(),
        'SOL': np.random.normal(100, 10, 30).cumsum()
    }, index=dates)

    st.line_chart(historical)

    # Crypto news
    with st.expander("📰 Crypto News"):
        news = [
            "Bitcoin ETF approval expected this quarter",
            "Ethereum completes major network upgrade",
            "Regulatory clarity improves for crypto markets",
            "New DeFi platform reaches $1B in total value locked"
        ]

        for item in news:
            st.write(f"• {item}")

    # Add crypto form
    st.subheader("Add Cryptocurrency")

    with st.form("add_crypto_form"):
        col_a, col_b = st.columns(2)

        with col_a:
            crypto_name = st.text_input("Cryptocurrency Name")
            amount = st.number_input("Amount", min_value=0.0, step=0.01)

        with col_b:
            buy_price = st.number_input("Buy Price ($)", min_value=0.0, step=0.01)
            purchase_date = st.date_input("Purchase Date")

        if st.form_submit_button("Add to Portfolio"):
            st.success(f"Added {amount} {crypto_name} to your portfolio!")


def display_expense_manager(dashboard):
    """Display expense manager"""
    st.header("💳 Expense Manager")

    # Two column layout
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Expense form
        with st.form("expense_form"):
            st.subheader("Add New Expense")

            col1, col2 = st.columns(2)

            with col1:
                date = st.date_input("Date", datetime.now())
                amount = st.number_input("Amount ($)", min_value=0.01, value=50.00)
                category = st.selectbox(
                    "Category",
                    ["Food & Dining", "Transportation", "Entertainment",
                     "Utilities", "Shopping", "Healthcare", "Education"]
                )

            with col2:
                payment = st.selectbox(
                    "Payment Method",
                    ["Credit Card", "Debit Card", "Cash", "Bank Transfer", "Digital Wallet"]
                )
                description = st.text_input("Description")
                tags = st.multiselect(
                    "Tags",
                    ["Necessary", "Business", "Personal", "Recurring", "One-time"]
                )

            if st.form_submit_button("➕ Add Expense"):
                st.success("Expense added successfully!")

        # Expense history
        st.subheader("Recent Expenses")

        expenses = dashboard.generate_expense_data()
        st.dataframe(
            expenses.sort_values('Date', ascending=False).head(10),
            use_container_width=True,
            hide_index=True
        )

        # Monthly summary
        st.subheader("Monthly Summary")

        monthly_expenses = expenses.copy()
        monthly_expenses['Date'] = pd.to_datetime(monthly_expenses['Date'])
        monthly_expenses['Month'] = monthly_expenses['Date'].dt.strftime('%Y-%m')

        monthly_summary = monthly_expenses.groupby(['Month', 'Category'])['Amount'].sum().reset_index()

        # Pivot for chart
        pivot_data = monthly_summary.pivot(index='Month', columns='Category', values='Amount').fillna(0)

        if not pivot_data.empty:
            st.bar_chart(pivot_data)

    with col_right:
        # Budget overview
        st.subheader("💰 Budget Overview")

        budget_data = {
            'Category': ['Food', 'Transport', 'Entertainment', 'Utilities', 'Shopping'],
            'Budget': [400, 200, 150, 300, 250],
            'Spent': [285.50, 145.00, 85.99, 220.50, 187.30]
        }

        budget_df = pd.DataFrame(budget_data)
        budget_df['Remaining'] = budget_df['Budget'] - budget_df['Spent']
        budget_df['Percentage'] = (budget_df['Spent'] / budget_df['Budget']) * 100

        for _, row in budget_df.iterrows():
            st.write(f"**{row['Category']}**")
            st.progress(min(row['Percentage'] / 100, 1))
            st.caption(f"${row['Spent']:.2f} of ${row['Budget']} (${row['Remaining']:.2f} left)")
            st.divider()

        # Total metrics
        total_budget = budget_df['Budget'].sum()
        total_spent = budget_df['Spent'].sum()

        st.metric("Total Budget", f"${total_budget:.2f}")
        st.metric("Total Spent", f"${total_spent:.2f}")
        st.metric("Remaining", f"${total_budget - total_spent:.2f}")

        # Savings goal
        st.subheader("🎯 Savings Goal")

        goal = 10000
        current = 6520
        progress = (current / goal) * 100

        st.metric("Goal", f"${goal:,}")
        st.metric("Current", f"${current:,}")
        st.progress(progress / 100)
        st.caption(f"{progress:.1f}% complete - ${goal - current:,} to go")

        # Export button
        if st.button("📥 Export Expense Report", use_container_width=True):
            st.info("Report generation started...")


def display_market_insights(dashboard):
    """Display market insights"""
    st.header("🌍 Market Insights")

    # Exchange rates
    st.subheader("💱 Exchange Rates")

    rates = {
        'EUR': 0.92,
        'GBP': 0.79,
        'JPY': 148.50,
        'CAD': 1.35,
        'AUD': 1.52
    }

    cols = st.columns(len(rates))
    for idx, (currency, rate) in enumerate(rates.items()):
        with cols[idx]:
            st.metric(f"USD/{currency}", f"{rate:.3f}")

    # Market sentiment
    st.subheader("📊 Market Sentiment")

    sentiment_data = pd.DataFrame({
        'Market': ['Stocks', 'Bonds', 'Crypto', 'Commodities'],
        'Sentiment': [65, 45, 72, 58],
        'Change': [2.1, -0.5, 3.2, 1.5]
    })

    # Display as bar chart
    st.bar_chart(sentiment_data.set_index('Market')['Sentiment'])

    # Economic indicators
    st.subheader("📈 Economic Indicators")

    indicators = pd.DataFrame({
        'Indicator': ['Inflation', 'Unemployment', 'GDP Growth', 'Interest Rate'],
        'Current': [3.2, 3.7, 2.1, 5.25],
        'Previous': [3.1, 3.8, 1.9, 5.25],
        'Trend': ['↑', '↓', '↑', '→']
    })

    st.dataframe(indicators, hide_index=True, use_container_width=True)

    # Map visualization
    st.subheader("🗺️ Global Markets")

    # Create map data
    map_data = pd.DataFrame({
        'lat': [40.7128, 51.5074, 35.6762, 22.3193, 1.3521],
        'lon': [-74.0060, -0.1278, 139.6503, 114.1694, 103.8198],
        'city': ['New York', 'London', 'Tokyo', 'Hong Kong', 'Singapore'],
        'market': ['NYSE', 'LSE', 'TSE', 'HKEX', 'SGX']
    })

    # Display map
    st.map(map_data)

    # Risk assessment
    st.subheader("⚠️ Risk Assessment")

    risk_factors = st.multiselect(
        "Select risk factors:",
        ["Market Volatility", "Interest Rates", "Inflation", "Geopolitical", "Liquidity"],
        default=["Market Volatility", "Inflation"]
    )

    if risk_factors:
        risk_score = len(risk_factors) * 20
        st.metric("Overall Risk Score", f"{risk_score}/100")

        if risk_score > 70:
            st.error("⚠️ High Risk Level - Consider defensive positioning")
        elif risk_score > 40:
            st.warning("⚠️ Moderate Risk - Monitor closely")
        else:
            st.success("✅ Low Risk - Portfolio appears stable")

    # Market news
    with st.expander("📰 Latest News"):
        news = [
            ("Federal Reserve maintains interest rates", "High impact"),
            ("Tech earnings exceed expectations", "Medium impact"),
            ("Oil prices rise on supply concerns", "Medium impact"),
            ("Housing market shows stability", "Low impact")
        ]

        for headline, impact in news:
            st.write(f"• **{headline}**")
            st.caption(f"  Impact: {impact}")


def main():
    """Main application function"""
    # Initialize dashboard
    dashboard = FinanceDashboard()

    # Display header
    display_header()

    # Create sidebar
    selected_view, _ = create_sidebar()

    # Route to selected view
    if "Portfolio Overview" in selected_view:
        display_portfolio_overview(dashboard)
    elif "Stock Analysis" in selected_view:
        display_stock_analysis(dashboard)
    elif "Crypto Tracker" in selected_view:
        display_crypto_tracker(dashboard)
    elif "Expense Manager" in selected_view:
        display_expense_manager(dashboard)
    elif "Market Insights" in selected_view:
        display_market_insights(dashboard)

    # Footer
    st.markdown("---")

    footer_cols = st.columns(3)
    with footer_cols[0]:
        st.caption("📊 Personal Finance Dashboard")
    with footer_cols[1]:
        st.caption("Built with Streamlit")
    with footer_cols[2]:
        st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()