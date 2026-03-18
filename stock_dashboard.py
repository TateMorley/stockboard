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

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

tools = [
    {
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
    {
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
    }
]


def get_stock_data(ticker: str, days: int, company_name: str = None):
    end = datetime.today()
    start = end - timedelta(days=days)

    data = yf.download(ticker, start=start, end=end)
    data = data[["Close", "Open", "High", "Low"]].reset_index()
    data.columns = ["Date", "Close", "Open", "High", "Low"]

    return data

st.title("Stock Data Analysis with LLMs")
user_message = st.text_input("Ask a question about a stock and its performance: ")
first_placeholder = st.empty()  # Placeholder for the first plot
compare_message = st.text_input("Compare two stocks: ")
second_placeholder = st.empty()  # Placeholder for the second plot

content_message = f"""You are an expert stock market analysis assistant with deep knowledge of financial markets,
company history, and significant world events that impact stock performance.
Today's date is {datetime.today().strftime('%Y-%m-%d')}.

DATE HANDLING:
- Always convert date references to an integer number of days from that date to today
- For specific years (e.g. 'since 2019'), use January 1st of that year as the start date
- For events (e.g. 'since COVID began'), use the most accurate date for that event (COVID: March 11, 2020)
- For relative time (e.g. 'last quarter', 'past year', 'this month'), calculate the exact number of days
- For seasons (e.g. 'since last summer'), use the meteorological start date of that season

TICKER RESOLUTION:
- Always resolve company names to their correct ticker symbol
- For companies with multiple share classes (e.g. Google: GOOGL vs GOOG), default to the most traded class
- For non-US companies, use their primary US-listed ticker if available

FUNCTION SELECTION:
- Use get_stock_data for any request about a single stock
- Use compare_stocks when the user mentions two companies or uses words like 'compare', 'vs', 'versus', 'against'
- If the user asks about a market event or news, still map it to the most relevant ticker(s)

ALWAYS return valid function calls with integer values for days. Never return a string for days."""

if user_message:
    with first_placeholder.spinner("Fetching stock data..."):
        response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": content_message
            },
            {"role": "user", "content": user_message}
        ],
        tools=tools,
        tool_choice="auto"
        )

        tool_call = response.choices[0].message.tool_calls[0]
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        arguments["days"] = int(arguments["days"])  # Ensure days is an integer


        if function_name == "get_stock_data":
            stock_data = get_stock_data(arguments["ticker"], arguments["days"], arguments.get("company_name"))

            fig = px.line(stock_data, x="Date", y=["Close", "Open"], title=f"{arguments['ticker']} Opening and Closing Price Over Time")
            first_placeholder.plotly_chart(fig)

if compare_message:
    with second_placeholder.spinner("Comparing stocks..."):
        response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
            messages=[
            {
                "role": "system",
                "content": content_message
            },
            {"role": "user", "content": compare_message}
        ],

        tools=tools,
        tool_choice="auto"
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
