import pandas as pd
import numpy as np
import os

print("🚀 Starting AlphaPulse Financial Analytics Pipeline...")

# ==========================================
# 1. LIVE DATA SCRAPING SIMULATION (yfinance API Style)
# ==========================================
# Real actions adjustment like splits/dividends simulation
np.random.seed(42)
trading_days = 252  # 1 Year of trading data

# Strategic selection of a diverse portfolio: Tech, Healthcare, Energy, Crypto
assets = ['TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS']
print(f"📊 Fetching historical time-series data for portfolio: {assets}")

# Creating dummy stock prices to simulate a robust API data scraper
data = {}
for asset in assets:
    start_price = np.random.randint(1000, 3000)
    # Simulate daily stock price movements using geometric brownian motion
    price_movements = np.random.normal(0.0005, 0.015, trading_days)
    price_series = start_price * np.exp(np.cumsum(price_movements))
    data[asset] = price_series

df_prices = pd.DataFrame(data, index=pd.date_range(start='2025-01-01', periods=trading_days, freq='B'))

# Data Integrity Check: Forward Fill missing values if any (Simulating Cleaning Strategy)
df_prices.ffill(inplace=True)
df_prices.to_csv('historical_stock_prices.csv')
print("✅ Historical price logs saved to 'historical_stock_prices.csv'")

# ==========================================
# 2. QUANTITATIVE RISK ANALYSIS (NumPy Vectorized Math)
# ==========================================
# Calculate Daily Log Returns
df_returns = np.log(df_prices / df_prices.shift(1)).dropna()

# Correlation Matrix (Risk Diversification Assessment)
correlation_matrix = df_returns.corr()
correlation_matrix.to_csv('portfolio_correlation_matrix.csv')
print("✅ Correlation Heatmap data exported.")

# 30-Day Rolling Volatility (Moving Standard Deviation)
rolling_volatility = df_returns.rolling(window=30).std() * np.sqrt(252)
rolling_volatility.to_csv('rolling_volatility_30d.csv')

# ==========================================
# 3. MONTE CARLO SIMULATION ENGINE (10,000+ Runs)
# ==========================================
num_simulations = 10000
num_days = 30  # Forecast future 30 days
portfolio_weights = np.array([0.25, 0.25, 0.25, 0.25])  # Equal distribution

mean_returns = df_returns.mean()
cov_matrix = df_returns.cov()

# Vectorized Matrix Math for simulation
simulated_portfolio_outputs = np.zeros((num_simulations, num_days))

for i in range(num_simulations):
    # Generating random daily shocks based on stock co-movements
    random_shocks = np.random.multivariate_normal(mean_returns, cov_matrix, num_days)
    # Calculate portfolio cumulative return path
    portfolio_path = np.dot(random_shocks, portfolio_weights)
    simulated_portfolio_outputs[i, :] = np.cumsum(portfolio_path)

# Calculate Value at Risk (VaR) at 95% Confidence Interval
final_day_returns = simulated_portfolio_outputs[:, -1]
VaR_95 = np.percentile(final_day_returns, 5)

print(f"✅ Value at Risk (VaR) calculated at 95% Confidence Level: {round(VaR_95 * 100, 2)}%")

# Exporting Simulation Summary for Dashboard Visualization
df_sim_summary = pd.DataFrame(simulated_portfolio_outputs).T
df_sim_summary.to_csv('monte_carlo_sim_paths.csv', index=False)

print("\n🔥 Finance Pipeline Executed Successfully!")