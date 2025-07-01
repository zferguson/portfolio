
import yfinance as yf
import pandas as pd
import streamlit as st
from typing import List, Optional, Dict

# Magnificent Seven stock tickers
TICKERS = [
    'AAPL',  # Apple
    'MSFT',  # Microsoft
    'GOOGL', # Alphabet (Google)
    'AMZN',  # Amazon
    'NVDA',  # Nvidia
    'META',  # Meta Platforms (Facebook)
    'TSLA'   # Tesla
]

# Benchmark index (S&P 500)
BENCHMARK = 'SPY'


@st.cache_data(show_spinner=False)
def get_market_data(
    tickers: List[str],
    benchmark: str,
    period: str = '5y',
    interval: str = '1d',
    ratios_list: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical prices, returns, and key ratios for given tickers and benchmark.
    Args:
        tickers: List of stock tickers.
        benchmark: Benchmark ticker (e.g., 'SPY').
        period: Data period (default '5y').
        interval: Data interval (default '1d').
        ratios_list: List of ratios to fetch (default: common ratios).
    Returns:
        dict with 'prices', 'returns', 'ratios' DataFrames.
    """
    all_tickers = tickers + [benchmark]
    try:
        data = yf.download(all_tickers, period=period, interval=interval, group_by='ticker', auto_adjust=True, progress=False)
        prices = pd.DataFrame({ticker: data[ticker]['Close'] for ticker in all_tickers})
        prices.index = pd.to_datetime(prices.index)
        returns = prices.pct_change().dropna()
    except Exception as e:
        st.error(f"Error downloading price data: {e}")
        return {'prices': pd.DataFrame(), 'returns': pd.DataFrame(), 'ratios': pd.DataFrame()}

    # Default ratios if not provided
    if ratios_list is None:
        ratios_list = ['trailingPE', 'priceToBook', 'dividendYield', 'marketCap', 'beta']

    ratios = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            ratios[ticker] = {r: info.get(r) for r in ratios_list}
        except Exception as e:
            st.warning(f"Could not fetch ratios for {ticker}: {e}")
            ratios[ticker] = {r: None for r in ratios_list}
    ratios_df = pd.DataFrame(ratios).T

    return {
        'prices': prices,
        'returns': returns,
        'ratios': ratios_df
    }


def get_single_ticker_data(ticker: str, period: str = '5y', interval: str = '1d') -> pd.DataFrame:
    """
    Fetch historical price data for a single ticker.
    """
    try:
        data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        data.index = pd.to_datetime(data.index)
        return data
    except Exception as e:
        st.error(f"Error downloading data for {ticker}: {e}")
        return pd.DataFrame()


# Example usage (remove or comment out in production Streamlit app):
if __name__ == "__main__":
    result = get_market_data(TICKERS, BENCHMARK)
    print(result)