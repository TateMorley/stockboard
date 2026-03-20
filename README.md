# Stock Data Dashboard

## Overview
An AI-powered stock analysis dashboard built in Python. Users can query stock data
in plain English and the app uses an LLM to automatically determine the right API
calls, fetch live market data, and visualize it interactively.

## Features
- LLM automated graph generation
- Stock comparison and isolated analysis
- More to come!

## Tech Stack
- Python
- Streamlit
- Plotly
- yfinance
- Groq (LLaMA 3.3)
- pandas

## Installation
1. Clone the repo
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\Activate.ps1`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file with your `GROQ_API_KEY`
6. Run with `streamlit run dashboard.py`

## Usage
Run the app with:
```
streamlit run dashboard.py
```

### Example Queries
- "Show me Apple's performance over the past 3 months"
- "Show me Nvidia's stock since they announced their 5000 series gpus"
- "Show Oracle's performance since they bought TikTok"
- "Compare Home Depot and Lowe's since the beginning of 2026"
- "Compare Amazon and Walmart's performance since COVID-19 lockdowns began"
- "Compare McDonalds and Burger King since the beginning of Q2"
- "Should I buy Apple shares right now?"
