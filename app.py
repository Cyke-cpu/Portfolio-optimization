import warnings
warnings.filterwarnings("ignore")

from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.optimize import minimize


st.set_page_config(
    page_title="Modern Portfolio Optimizer",
    page_icon="📈",
    layout="wide"
)


# -----------------------------
# Helper Functions
# -----------------------------

@st.cache_data(show_spinner=False)
def download_market_data(tickers, start_date, end_date):
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError("No data downloaded. Check your tickers or date range.")

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers

    prices = prices.dropna(how="all")
    prices = prices.ffill().dropna()

    return prices


def portfolio_performance(weights, annual_returns, cov_matrix, risk_free_rate):
    portfolio_return = np.dot(weights, annual_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    if portfolio_volatility == 0:
        sharpe_ratio = 0
    else:
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    return portfolio_return, portfolio_volatility, sharpe_ratio


def negative_sharpe_ratio(weights, annual_returns, cov_matrix, risk_free_rate):
    return -portfolio_performance(weights, annual_returns, cov_matrix, risk_free_rate)[2]


def portfolio_volatility_only(weights, annual_returns, cov_matrix, risk_free_rate):
    return portfolio_performance(weights, annual_returns, cov_matrix, risk_free_rate)[1]


def run_monte_carlo(num_assets, simulations, annual_returns, cov_matrix, risk_free_rate):
    np.random.seed(42)
    simulation_results = np.zeros((3, simulations))
    weights_record = []

    for i in range(simulations):
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights)
        weights_record.append(weights)

        port_return, port_volatility, sharpe_ratio = portfolio_performance(
            weights, annual_returns, cov_matrix, risk_free_rate
        )

        simulation_results[0, i] = port_return
        simulation_results[1, i] = port_volatility
        simulation_results[2, i] = sharpe_ratio

    simulation_df = pd.DataFrame({
        "Return": simulation_results[0],
        "Volatility": simulation_results[1],
        "Sharpe Ratio": simulation_results[2]
    })

    return simulation_df, weights_record


def optimize_portfolios(num_assets, annual_returns, cov_matrix, risk_free_rate):
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1.0 / num_assets]

    max_sharpe_result = minimize(
        negative_sharpe_ratio,
        initial_guess,
        args=(annual_returns, cov_matrix, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    min_vol_result = minimize(
        portfolio_volatility_only,
        initial_guess,
        args=(annual_returns, cov_matrix, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not max_sharpe_result.success:
        raise RuntimeError("Maximum Sharpe optimization failed. Try fewer tickers or a longer date range.")

    if not min_vol_result.success:
        raise RuntimeError("Minimum volatility optimization failed. Try fewer tickers or a longer date range.")

    return max_sharpe_result.x, min_vol_result.x


# -----------------------------
# Sidebar Inputs
# -----------------------------

st.title("📈 Modern Portfolio Optimizer")
st.write(
    "An interactive portfolio optimization dashboard using Yahoo Finance data, "
    "Modern Portfolio Theory, Monte Carlo simulation, and Sharpe ratio optimization."
)

st.sidebar.header("Portfolio Settings")

default_tickers = "AAPL, MSFT, NVDA, JPM, XOM, GLD, TLT, SPY"
tickers_input = st.sidebar.text_input("Tickers", default_tickers)

start_date = st.sidebar.date_input("Start Date", date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", date.today())

risk_free_rate_percent = st.sidebar.number_input(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=4.0,
    step=0.25
)

simulations = st.sidebar.slider(
    "Monte Carlo Simulations",
    min_value=1000,
    max_value=50000,
    value=20000,
    step=1000
)

run_button = st.sidebar.button("Run Optimizer")

st.sidebar.caption("Educational project only. Not financial advice.")


# -----------------------------
# Main App
# -----------------------------

if not run_button:
    st.info("Enter your settings in the sidebar and click **Run Optimizer**.")
    st.stop()

try:
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

    if len(tickers) < 2:
        st.error("Please enter at least two tickers.")
        st.stop()

    risk_free_rate = risk_free_rate_percent / 100
    trading_days = 252

    with st.spinner("Downloading market data and running optimization..."):
        prices = download_market_data(tickers, start_date, end_date)
        returns = prices.pct_change().dropna()

        if returns.empty:
            st.error("Not enough price data to calculate returns. Try a longer date range.")
            st.stop()

        annual_returns = returns.mean() * trading_days
        annual_volatility = returns.std() * np.sqrt(trading_days)
        cov_matrix = returns.cov() * trading_days
        corr_matrix = returns.corr()

        num_assets = len(tickers)

        simulation_df, weights_record = run_monte_carlo(
            num_assets,
            simulations,
            annual_returns,
            cov_matrix,
            risk_free_rate
        )

        max_sharpe_weights, min_vol_weights = optimize_portfolios(
            num_assets,
            annual_returns,
            cov_matrix,
            risk_free_rate
        )

        max_sharpe_performance = portfolio_performance(
            max_sharpe_weights,
            annual_returns,
            cov_matrix,
            risk_free_rate
        )

        min_vol_performance = portfolio_performance(
            min_vol_weights,
            annual_returns,
            cov_matrix,
            risk_free_rate
        )

    st.subheader("Optimized Portfolio Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Max Sharpe Return", f"{max_sharpe_performance[0]:.2%}")
    col2.metric("Max Sharpe Volatility", f"{max_sharpe_performance[1]:.2%}")
    col3.metric("Max Sharpe Ratio", f"{max_sharpe_performance[2]:.2f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Min Vol Return", f"{min_vol_performance[0]:.2%}")
    col5.metric("Min Volatility", f"{min_vol_performance[1]:.2%}")
    col6.metric("Min Vol Sharpe", f"{min_vol_performance[2]:.2f}")

    st.subheader("Portfolio Allocation")

    allocation_table = pd.DataFrame({
        "Ticker": tickers,
        "Max Sharpe Weight (%)": max_sharpe_weights * 100,
        "Min Volatility Weight (%)": min_vol_weights * 100
    }).round(2)

    allocation_table = allocation_table.sort_values(
        by="Max Sharpe Weight (%)",
        ascending=False
    ).reset_index(drop=True)

    st.dataframe(allocation_table, use_container_width=True)

    st.subheader("Asset Risk and Return Metrics")

    asset_metrics = pd.DataFrame({
        "Annual Return": annual_returns,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": (annual_returns - risk_free_rate) / annual_volatility
    }).round(4)

    st.dataframe(asset_metrics, use_container_width=True)

    st.subheader("Efficient Frontier")

    fig, ax = plt.subplots(figsize=(11, 6))
    scatter = ax.scatter(
        simulation_df["Volatility"],
        simulation_df["Return"],
        c=simulation_df["Sharpe Ratio"],
        cmap="viridis",
        alpha=0.6
    )
    fig.colorbar(scatter, ax=ax, label="Sharpe Ratio")
    ax.scatter(
        max_sharpe_performance[1],
        max_sharpe_performance[0],
        marker="*",
        s=400,
        label="Maximum Sharpe Portfolio"
    )
    ax.scatter(
        min_vol_performance[1],
        min_vol_performance[0],
        marker="*",
        s=400,
        label="Minimum Volatility Portfolio"
    )
    ax.set_title("Monte Carlo Efficient Frontier")
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Asset Correlation Matrix")
    st.pyplot(fig)

    if "SPY" in returns.columns:
        st.subheader("Optimized Portfolio vs. SPY")

        portfolio_returns = returns.dot(max_sharpe_weights)
        comparison = pd.DataFrame({
            "Optimized Portfolio": portfolio_returns,
            "SPY": returns["SPY"]
        })

        cumulative = (1 + comparison).cumprod()

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(cumulative.index, cumulative["Optimized Portfolio"], linewidth=2, label="Optimized Portfolio")
        ax.plot(cumulative.index, cumulative["SPY"], linewidth=2, label="SPY")
        ax.set_title("Growth of $1: Optimized Portfolio vs. SPY")
        ax.set_xlabel("Date")
        ax.set_ylabel("Growth of $1")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.warning("SPY was not included in the ticker list, so benchmark comparison was skipped.")

    st.subheader("Interpretation")
    st.write(
        "The Maximum Sharpe portfolio targets the highest historical risk-adjusted return, "
        "while the Minimum Volatility portfolio targets the lowest historical portfolio risk. "
        "The Efficient Frontier shows the tradeoff between expected return and volatility across simulated portfolios."
    )

except Exception as error:
    st.error(f"Something went wrong: {error}")
