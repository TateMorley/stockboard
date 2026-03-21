import pandas as pd
import streamlit as st
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import time

def get_stock_data(ticker: str, days: int, company_name: str = None):
    """
    Fetch historical daily stock data for a given ticker.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL')
        days: Number of historical days to fetch
        company_name: Optional company name for display purposes

    Returns:
        DataFrame with columns: Date, Close, Open, High, Low
    """
    end = datetime.today()
    start = end - timedelta(days=days)

    data = yf.download(ticker, start=start, end=end)
    data = data[["Close", "Open", "High", "Low"]].reset_index()
    data.columns = ["Date", "Close", "Open", "High", "Low"]

    return data

def get_stock_data_hours(ticker: str, hours: int, company_name: str = None):
    """
    Fetch historical intraday (hourly) stock data for a given ticker.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL')
        hours: Number of historical hours to fetch
        company_name: Optional company name for display purposes

    Returns:
        DataFrame with columns: Date, Close, Open, High, Low
    """
    end = datetime.today()
    start = end - timedelta(hours=hours)

    # interval="1h" fetches hourly data instead of daily
    data = yf.download(ticker, start=start, end=end, interval="1h")
    data = data[["Close", "Open", "High", "Low"]].reset_index()
    data.columns = ["Date", "Close", "Open", "High", "Low"]

    return data

def lookup_ticker(company_name: str):
    """
    Resolve a company name to its stock ticker symbol.
    Prioritizes US-listed equities on NASDAQ or NYSE.

    Args:
        company_name: Full or partial company name (e.g. 'Apple' or 'Apple Inc.')

    Returns:
        Ticker symbol string, or None if not found
    """
    results = yf.Search(company_name).quotes

    # First pass: match name and filter to US-listed equities
    for result in results:
        if (company_name.lower() in result.get("longname", "").lower() and
            result.get("quoteType") == "EQUITY" and
            result.get("exchDisp") in ["NASDAQ", "NYSE"]):
            return result["symbol"]

    # Second pass: return highest-scored US equity even without name match
    for result in results:
        if (result.get("quoteType") == "EQUITY" and
            result.get("exchDisp") in ["NASDAQ", "NYSE"]):
            return result["symbol"]

    # Final fallback: return first result regardless of exchange
    return results[0]["symbol"] if results else None

def import_portfolio():
    """
    Render a file uploader widget and load a CSV portfolio into a DataFrame.
    Displays the imported portfolio as a table on success.
    """
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", label_visibility="collapsed")

    if uploaded_file is not None:
        st.session_state.portfolio = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file to import your portfolio.")

def analyze_portfolio(portfolio):
    """
    Analyze each stock in the portfolio and display current value and profit/loss.

    Args:
        portfolio: DataFrame with columns: Ticker, Shares, Purchase Price
    """
    portfolio_history = pd.DataFrame()

    for index, row in portfolio.iterrows():
        ticker = row["Ticker"]
        shares = row["Shares"]
        purchase_price = row["Purchase Price"]

        # Build portfolio history for chart
        data = get_stock_data(ticker, 3, company_name=ticker)
        data["Value"] = data["Close"] * shares
        data = data[["Date", "Value"]].rename(columns={"Value": ticker})

        if portfolio_history.empty:
            portfolio_history = data
        else:
            portfolio_history = pd.merge(portfolio_history, data, on="Date", how="inner")

        # Fetch current price and calculate metrics
        try:
            curr_price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[0]
            curr_value = shares * curr_price
            profit_loss = curr_value - (shares * purchase_price)
            profit_loss_percent = (profit_loss / (shares * purchase_price)) * 100
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            continue

        # Resolve full company name from ticker
        company_name = yf.Ticker(ticker).info.get("longName", ticker)

        st.write(f"**{company_name}**")
        st.write(f"{ticker}: Current Price: ${curr_price:.2f}")
        st.write(f"{ticker}: Current Value: ${curr_value:.2f}")
        st.write(f"{ticker}: Profit: ${profit_loss:.2f} ({profit_loss_percent:.2f}%)")

def chart_portfolio(portfolio):
    """
    Plot the total portfolio value over the last 72 hours as a line chart.
    Merges hourly price data for all tickers and sums their weighted values.

    Args:
        portfolio: DataFrame with columns: Ticker, Shares, Purchase Price
    """
    portfolio_history = pd.DataFrame()

    for index, row in portfolio.iterrows():
        ticker = row["Ticker"]
        shares = row["Shares"]

        # Fetch 72 hours of hourly data and weight by share count
        data = get_stock_data_hours(ticker, 72)
        data["Value"] = data["Close"] * shares
        data = data[["Date", "Value"]].rename(columns={"Value": ticker})

        if portfolio_history.empty:
            portfolio_history = data
        else:
            # Merge on Date to align all stocks to the same timestamps
            portfolio_history = pd.merge(portfolio_history, data, on="Date", how="inner")

    # Sum all ticker columns to get total portfolio value per timestamp
    tickers = list(portfolio["Ticker"])
    portfolio_history["Total"] = portfolio_history[tickers].sum(axis=1)

    fig = px.line(portfolio_history, x="Date", y="Total", title="Total Portfolio Value Over Time")
    st.plotly_chart(fig)

def llm_call_with_retry(client, messages, tools, max_retries=5):
    """
    Calls the Groq API repeatedly to avoid failure messages

    Args:
        messages: List of message dicts to send
        tools: list of tool definitions
        max_retries: number of chances to fails
    Returns:
        API response object, or None if they all fail
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model = "llama-3.3-70b-versatile",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            return response
        except Exception as e:
            if attempt < max_retries - 1:   # If it fails, wait one second then call again
                time.sleep(1)
            else:                           # If it fails 5 times, give an error
                raise e
