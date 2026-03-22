"""
StockBoard - AI-Powered Stock Analysis Dashboard
Built with Streamlit, Plotly, yfinance, and Groq LLM.

Features:
    - Live stock news feed
    - Interactive portfolio tracker with CSV import/export
    - AI-powered natural language stock queries
    - Candlestick and comparison charts
    - AI investment recommendations
"""

# API and library imports
from groq import Groq
from datetime import datetime, timedelta
import yfinance as yf
import json
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import pandas as pd
import feedparser
from urllib.parse import quote

# Custom utility functions
from utilities import *

load_dotenv()

try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception: # If this happens, then it's trying to run locally
    pass

# LLM setup
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# LLM function setup
tools = [
    { # Data setup
        "type": "function",
        "function": {
            "name": "get_stock_data",
            "description": "Get the stock data for a given ticker and time range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol e.g. NVDA."
                    },
                    "days": {
                        "type": "integer",
                        "description": "The number of days of historical data to fetch. Must always be a whole number."
                    },
                    "company_name": {
                        "type": "string",
                        "description": "The name of the company to get the stock ticker for."
                    }
                },
                "required": ["ticker", "days"]
            }
        }
    },
    { # Comparison setup
        "type": "function",
        "function": {
            "name": "compare_stocks",
            "description": "Compare the stock performance of two companies side by side.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker1": {
                        "type": "string",
                        "description": "The stock ticker symbol for the first company"
                    },
                    "ticker2": {
                        "type": "string",
                        "description": "The stock ticker symbol for the second company"
                    },
                    "days": {
                        "type": "integer",
                        "description": "The number of days of historical data to fetch for both companies. Must always be a whole number and the same for both companies."
                    }
                },
                "required": ["ticker1", "ticker2", "days"]
            }
        }
    },
    { # Recommendation setup
        "type": "function",
        "function": {
            "name": "get_recommendation",
            "description": "Provide a stock recommendation based on recent performance. Responses are a complete sentence",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "action": {"type": "string", "enum": ["Buy", "Hold", "Sell"]},
                    "confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
                    "reasoning": {"type": "string", "description": "Brief explanation for action choice"},
                    "risk": {"type": "string", "enum": ["Low", "Medium", "High"]}
                },
                "required": ["ticker", "action", "confidence", "reasoning", "risk"]
            }
        }
    }
]

# Title setup
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>StockBoard</h1>", unsafe_allow_html=True) # Title
st.divider()

# Portfolio chart setup and import
if "portfolio" not in st.session_state:
    if os.path.exists("portfolio.csv"):
        st.session_state.portfolio = pd.read_csv("portfolio.csv")
    else:
        st.session_state.portfolio = pd.DataFrame({
            "Ticker": ["AAPL", "GOOGL", "AMZN"],
            "Shares": [10, 2, 5],
            "Purchase Price": [150.0, 2800.0, 3500.0]
        })

main_col1, main_col2, main_col3 = st.columns([2, 4, 4])

with main_col1:
    # News Feed --------------------------------------------------------------------------------------------------
    st.subheader("Recent Stock News")
    query = "Recent stock news"
    url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)

    for entry in feed.entries[:10]:
        st.markdown(f"[{entry.title}]({entry.link})")
        st.caption(entry.published)
        st.divider()

with main_col2:
    # Portfolio Tracker ------------------------------------------------------------------------------------------
    st.subheader("Portfolio Tracker")

    # Ticker lookup
    st.caption("Ticker Lookup")
    lookup_col1, lookup_col2 = st.columns([3, 1])

    with lookup_col1:
        company_name = st.text_input("Enter a company name:")

    # Ticker display
    with lookup_col2:
        if st.button("Lookup Ticker"):
            if company_name:
                ticker = lookup_ticker(company_name)
                if ticker:
                    st.success(f"Ticker: {ticker}")
                else:
                    st.error("Company not found.")

    # Import section
    uploaded_file = st.file_uploader("Import Portfolio CSV", type="csv")
    if uploaded_file is not None:
        st.session_state.portfolio = pd.read_csv(uploaded_file)

    portfolio = st.data_editor(
        st.session_state.portfolio,
        num_rows="dynamic",
        key="portfolio_editor"
    )

    col1, col2 = st.columns(2)

    # Export section
    with col1:
        csv = portfolio.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Export Portfolio as CSV",
            data=csv,
            file_name='portfolio.csv',
            mime='text/csv'
        )

        st.caption("Data is not stored locally, make sure to export your data to save portfolio content across sessions")

    # Analyze section
    analyze_button = None

    with col2:
        analyze_button = st.button("Analyze Portfolio")

    a_col1, a_col2 = st.columns(2)

    if analyze_button:
        with a_col1:
            analyze_portfolio(portfolio)
        with a_col2:
            chart_portfolio(portfolio)

with main_col3:
    # LLM integration ------------------------------------------------------------------------------------------
    st.subheader("LLM Stock Analysis")

    user_message = st.text_input("Ask a question about a stock and its performance: ")
    first_placeholder = st.empty()  # Placeholder for the first plot

    compare_message = st.text_input("Compare two stocks: ")
    second_placeholder = st.empty()  # Placeholder for the second plot

    advise_question = st.text_input("Ask a question for advice on future stock actions")
    third_placeholder = st.empty()

    content_message = f"""You are an expert stock market analysis assistant with deep knowledge of financial markets,
    company history, and significant world events that impact stock performance.
    Today's date is {datetime.today().strftime('%Y-%m-%d')}.

    DATE HANDLING:
    - Always convert date references to an integer number of days from that date to today
    - For specific years (e.g. 'since 2019'), use January 1st of that year as the start date
    - For events (e.g. 'since COVID began'), use the most accurate date for that event (COVID: March 11, 2020)
    - For relative time (e.g. 'last quarter', 'past year', 'this month'), calculate the exact number of days
    - For seasons (e.g. 'since last summer'), use the meteorological start date of that season

    EVENT HANDLING:
    - When the user references a specific event like 'since the iPhone 16 release' or 'since Oracle bought TikTok',
    use your knowledge to identify the approximate date of that event, then calculate the number of days
    from that date to today and pass it as an integer.
    Examples:
    - 'since the iPhone 16 release' -> iPhone 16 released September 20, 2024 -> calculate days from 2024-09-20 to today
    - 'since they bought TikTok' -> identify the acquisition date -> calculate days from that date to today
    - 'past 4 months' -> 120 days
    - 'since 2019' -> calculate days from 2019-01-01 to today

    TICKER RESOLUTION:
    - Always resolve company names to their correct ticker symbol
    - For companies with multiple share classes (e.g. Google: GOOGL vs GOOG), default to the most traded class
    - For non-US companies, use their primary US-listed ticker if available

    ADVICE STRUCTURE:
    - When providing stock action response advice, the reason should be at least one complete sentence
    - When providing stock action response advice, the reason should be at most two complete sentences

    FUNCTION SELECTION:
    - Use get_stock_data for any request about a single stock
    - Use compare_stocks when the user mentions two companies or uses words like 'compare', 'vs', 'versus', 'against'
    - Use get_recommendation when the user is seeking advice for actions to take involving selling, buying, or holding shares of a stock.
    - If the user asks about a market event or news, still map it to the most relevant ticker(s)

    ALWAYS return valid function calls with integer values for days. Never return a string for days."""

    if user_message:
        try:
            with first_placeholder.spinner("Fetching stock data..."):
                response = llm_call_with_retry(
                    client=client,
                    messages=[
                        {"role": "system", "content": content_message},
                        {"role": "user", "content": user_message}
                    ],
                    tools=tools,
                )

                tool_call = response.choices[0].message.tool_calls[0]
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                arguments["days"] = int(arguments["days"])  # Ensure days is an integer

                if function_name == "get_stock_data":
                    stock_data = get_stock_data(arguments["ticker"], arguments["days"], arguments.get("company_name"))

                    fig = go.Figure(data=[go.Candlestick(
                        x=stock_data["Date"],
                        open=stock_data["Open"],
                        close=stock_data["Close"],
                        high=stock_data["High"],
                        low=stock_data["Low"]
                    )])
                    fig.update_layout(title=f"{arguments.get('company_name', arguments['ticker'])} Stock Performance Over Last {arguments['days']} Days")
                    first_placeholder.plotly_chart(fig)
        except Exception as e:
            st.error("Something went wrong. Try rephrasing or resubmitting your request.")

    if compare_message:
        try:
            with second_placeholder.spinner("Comparing stocks..."):
                response = llm_call_with_retry(
                    client=client,
                    messages=[
                        {"role": "system", "content": content_message},
                        {"role": "user", "content": compare_message}
                    ],
                    tools=tools,
                )

                tool_call = response.choices[0].message.tool_calls[0]
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                arguments["days"] = int(arguments["days"])  # Ensure days is an integer

            if function_name == "compare_stocks":
                stock_data1 = get_stock_data(arguments["ticker1"], arguments["days"])
                stock_data2 = get_stock_data(arguments["ticker2"], arguments["days"])

                fig = make_subplots(rows=1, cols=2, subplot_titles=(arguments["ticker1"], arguments["ticker2"]))

                fig.add_trace(go.Scatter(x=stock_data1["Date"], y=stock_data1["Close"], mode='lines', name='Close'), row=1, col=1)
                fig.add_trace(go.Scatter(x=stock_data1["Date"], y=stock_data1["Open"], mode='lines', name='Open'), row=1, col=1)

                fig.add_trace(go.Scatter(x=stock_data2["Date"], y=stock_data2["Close"], mode='lines', name='Close'), row=1, col=2)
                fig.add_trace(go.Scatter(x=stock_data2["Date"], y=stock_data2["Open"], mode='lines', name='Open'), row=1, col=2)

                fig.update_layout(title_text="Stock Performance Comparison", showlegend=False)
                second_placeholder.plotly_chart(fig)
        except Exception as e:
            st.error("Something went wrong. Try rephrasing or resubmitting your request.")

    if advise_question:
        try:
            with third_placeholder.spinner("Generating response"):
                response = llm_call_with_retry(
                    client=client,
                    messages=[
                        {"role": "system", "content": content_message},
                        {"role": "user", "content": advise_question}
                    ],
                    tools=tools,
                )

                tool_call = response.choices[0].message.tool_calls[0]
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
            if function_name == "get_recommendation":
                st.write(arguments["ticker"])

                # Action display
                action = arguments["action"]
                if action == "Buy":
                    st.success(f"**Action: {action}**")
                elif action == "Sell":
                    st.error(f"**Action: {action}**")
                elif action == "Hold":
                    st.warning(f"**Action: {action}**")

                # Confidence display
                confidence = arguments["confidence"]
                if confidence == "High":
                    st.success(f"**Confidence: {confidence}**")
                elif confidence == "Medium":
                    st.warning(f"**Confidence: {confidence}**")
                elif confidence == "Low":
                    st.error(f"**Confidence: {confidence}**")

                st.write(arguments["reasoning"])

                # Risk display
                risk = arguments["risk"]
                if risk == "Low":
                    st.success(f"**Risk: {risk}**")
                elif risk == "Medium":
                    st.warning(f"**Risk: {risk}**")
                elif risk == "Low":
                    st.error(f"**Risk: {risk}**")

                st.caption("⚠️ This is not financial advice. Always consult a professional before making investment decisions.")
        except Exception as e:
            st.error("Something went wrong. Try rephrasing or resubmitting your request.")
st.divider()
st.caption("⚠️ This tool is for informational and educational purposes only and does not constitute financial advice. Stock recommendations generated by AI are not reliable indicators of future performance. Always consult a licensed financial advisor before making investment decisions. Market data provided by Yahoo Finance.")
