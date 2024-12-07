# Quantitative Finance Dashboard

A comprehensive financial analytics dashboard built with Streamlit that provides tools for stock analysis, portfolio optimization, and options calculations. This project was developed as part of the Introduction to Quantitative Finance course at Ashoka University.

**Author**: Jigyansu Rout

## Features

- **Data Explorer**: Analyze stock data with interactive filters and visualizations
- **Portfolio Optimization**: Monte Carlo simulation-based portfolio optimization with efficient frontier visualization
- **Options Calculator**: Black-Scholes model implementation for option pricing and Greeks calculation
- **Stock Recommendations**: Technical analysis-based stock recommendations using momentum and mean reversion strategies

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the dashboard:
   ```bash
   streamlit run app.py
   ```
   The dashboard will open automatically in your default web browser at `http://localhost:8501`

## Requirements

- Python 3.7+
- See `requirements.txt` for full list of dependencies

## Usage

1. Once the dashboard is running, use the sidebar navigation to switch between different tools:
   - Options Calculator
   - Stock Recommendations
   - Portfolio Optimization

2. Each tool provides interactive inputs and real-time calculations:
   - Options Calculator: Enter stock price, strike price, volatility, etc. to calculate option prices and Greeks
   - Stock Recommendations: View technical analysis-based recommendations using momentum and mean reversion strategies
   - Portfolio Optimization: Input stock tickers and constraints to generate optimized portfolios

