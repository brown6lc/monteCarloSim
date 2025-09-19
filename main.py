import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


def download_data(ticker, start="2010-01-01"):
    '''
    Retrieve data of a specified ticker starting at a certain time interval.

    Params:
        ticker: ticker to pull data from (VOO, TSLA, etc.)
        start (const): when to start pulling data from until present day

    Returns:
        data: historical price data for a given ticker
    '''
    data = yf.download(ticker, start=start)
    data = data[['Close']].rename(columns={'Close': ticker})
    return data


def plot_price(data, ticker):
    '''
    Plots the price of a provided ticker, uses return value from download_data.

    Params:
        data: data retrieved from running the download_data function
        ticker: ticker to plot price from

    Returns:
        none
    '''
    plt.figure(figsize=(15,6))
    plt.plot(data[ticker], label=ticker)
    plt.title(f"{ticker} Adjusted Close Price (2010–Present)")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plt.show()


def compute_log_returns(data, ticker):
    '''
    Computes log returns from a ticker with provided data.

    Params:
        data: historical price data
        ticker: ticker to use data from

    Returns:
        log_returns: log returns time series from price data
    '''
    log_returns = np.log(data[ticker] / data[ticker].shift(1)).dropna()
    return log_returns


def plot_log_returns(log_returns, ticker):
    '''
    Plots data generated from log_returns function.

    Params:
        log_returns: log returns time series from price data from log_returns function
        ticker: ticker to use data from

    Returns:
        none
    '''
    plt.figure(figsize=(15,6))
    plt.plot(log_returns, label="Log Returns", color="orange")
    plt.title(f"{ticker} Daily Logarithmic Returns")
    plt.xlabel("Date")
    plt.ylabel("Log Return")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_log_histogram(log_returns, ticker):
    '''
    Plots data generated from log_returns function.

    Params:
        log_returns: log returns time series from price data from log_returns function
        ticker: ticker to use data from

    Returns:
        mu: mean of log returns
        sigma: standard deviation of log returns
    '''
    plt.figure(figsize=(10,6))
    sns.histplot(log_returns, bins=100, kde=False, stat="density", color="skyblue", label="Log Returns")
    mu, sigma = log_returns.mean(), log_returns.std()
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    plt.plot(x, norm.pdf(x, mu, sigma), "r", label=f"Normal Fit\nμ={mu}, σ={sigma}")
    plt.title(f"{ticker} Daily Log Returns Distribution")
    plt.xlabel("Log Return")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()
    return mu, sigma


def compute_drift_volatility(log_returns):
    '''
    Computes drift and volatility of the log returns.

    Params:
        log_returns: log returns time series from price data from log_returns function

    Returns:
        mu_sample: average daily log return
        sigma_sample: daily volatility
        drift: daily drift term
        drift_annual: annualized drift term
        sigma_annual: annualized volatility
    '''
    mu_sample = log_returns.mean()
    sigma_sample = log_returns.std()
    drift = mu_sample + 0.5 * sigma_sample**2
    drift_annual = drift * 252
    sigma_annual = sigma_sample * np.sqrt(252)
    return mu_sample, sigma_sample, drift, drift_annual, sigma_annual


def monte_carlo_simulation(S0, drift, sigma, T=252, N=1000, seed=42):
    '''
    Performs a Monte Carlo simulation of stock prices using
    a geometric Brownian motion model.

    Params:
        S0 (float): initial stock price
        drift (float): expected return (per step, daily drift)
        sigma (float): volatility (per step, usually daily volatility)
        T (int): number of time steps (default 252 trading days)
        N (int): number of simulations (default 1000)
        seed (int): random seed for reproducibility

    Returns:
        np.ndarray: simulated stock price paths of shape (T, N)
    '''
    np.random.seed(seed)
    drift = float(drift)   # ensure scalar
    sigma = float(sigma)   # ensure scalar

    Z = np.random.standard_normal((T, N))
    daily_returns = np.exp((drift - 0.5 * sigma**2) + sigma * Z)

    S = np.zeros_like(daily_returns)
    S[0] = S0
    for t in range(1, T):
        S[t] = S[t-1] * daily_returns[t]
    return S


def plot_mc_paths(S, ticker, n_paths=10):
    '''
    Plots paths generated from the monte carlo simulation.

    Params:
        S: simulation data
        ticker: ticker for simulation to be performed on
        n_paths: number of paths to plot

    Returns:
        none
    '''
    T = S.shape[0]
    lower_bound = np.percentile(S, 2.5, axis=1)
    upper_bound = np.percentile(S, 97.5, axis=1)
    median_path = np.percentile(S, 50, axis=1)

    plt.figure(figsize=(14,7))
    plt.plot(S[:, :n_paths], alpha=0.3, color="gray", label="Sample Paths")
    plt.plot(median_path, color="blue", linewidth=2, label="Median Path")
    plt.fill_between(range(T), lower_bound, upper_bound, color="skyblue", alpha=0.4, label="95% Confidence Interval")
    plt.title(f"{ticker} Monte Carlo Simulation (1 Year, 95% Confidence Band)")
    plt.xlabel("Days")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_terminal_histogram(S, ticker):
    '''
    Plots a histogram of terminal (final-day) stock prices from a Monte Carlo simulation,
    fits a normal distribution curve, and overlays it for comparison.

    Params:
        S (np.ndarray): simulated stock price paths of shape (T, N)
        ticker (str): stock ticker symbol (for labeling the plot)

    Returns:
        np.ndarray: array of terminal prices (final-day prices from all simulations)
    '''
    terminal_prices = S[-1, :]
    plt.figure(figsize=(12,6))
    sns.histplot(terminal_prices, bins=50, kde=True, color="purple", stat="density")
    mu_T, sigma_T = terminal_prices.mean(), terminal_prices.std()
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    plt.plot(x, norm.pdf(x, mu_T, sigma_T), "r", label=f"Normal Fit\nμ={mu_T:.2f}, σ={sigma_T:.2f}")
    plt.title(f"{ticker} Simulated Terminal Price Distribution (1 Year)")
    plt.xlabel("Price ($)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()
    return terminal_prices


### testing ###
if __name__ == "__main__":
    ticker = "VOO"
    data = download_data(ticker)
    plot_price(data, ticker)

    log_returns = compute_log_returns(data, ticker)
    plot_log_returns(log_returns, ticker)
    mu, sigma = plot_log_histogram(log_returns, ticker)

    mu_sample, sigma_sample, drift, drift_annual, sigma_annual = compute_drift_volatility(log_returns)
    print(f"Daily drift: {drift}, Annualized drift: {drift_annual}, Annualized volatility: {sigma_annual}")

    S = monte_carlo_simulation(S0=data[ticker].iloc[-1], drift=drift, sigma=sigma_sample)
    plot_mc_paths(S, ticker)
    terminal_prices = plot_terminal_histogram(S, ticker)
