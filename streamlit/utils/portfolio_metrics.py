import numpy as np
import pandas as pd
from scipy.optimize import minimize


def portfolio_performance(prices: pd.DataFrame, weights: dict, benchmark: str = None, risk_free_rate: float = 0.0, periods_per_year: int = 252):
    """
    Calculate annualized return, risk, Sharpe ratio, and return vs benchmark for a portfolio.
    Args:
        prices: DataFrame of asset prices (columns: tickers).
        weights: dict of {ticker: weight} (weights sum to 1).
        benchmark: optional benchmark ticker in prices.
        risk_free_rate: annual risk-free rate (as decimal).
        periods_per_year: trading periods per year (default 252 for daily).
    Returns:
        dict of portfolio metrics.
    """
    # Ensure weights align with available tickers
    tickers = [t for t in weights if t in prices.columns]
    w = np.array([weights[t] for t in tickers])
    price_data = prices[tickers]
    returns = price_data.pct_change().dropna()

    # Portfolio returns
    port_ret = returns @ w

    # Annualized return and risk
    mean_daily = np.mean(port_ret)
    std_daily = np.std(port_ret)
    ann_return = (1 + mean_daily) ** periods_per_year - 1
    ann_risk = std_daily * np.sqrt(periods_per_year)


    # Sharpe ratio
    sharpe = (ann_return - risk_free_rate) / ann_risk if ann_risk != 0 else np.nan

    # Sortino ratio (downside risk)
    downside_returns = port_ret[port_ret < risk_free_rate]
    if len(downside_returns) > 0:
        downside_std = downside_returns.std() * np.sqrt(periods_per_year)
        sortino = (ann_return - risk_free_rate) / downside_std if downside_std != 0 else np.nan
    else:
        sortino = np.nan

    # Tail risk (5th percentile loss, Value at Risk at 95%)
    tail_risk = np.percentile(port_ret, 5)

    # Benchmark comparison
    if benchmark and benchmark in prices.columns:
        bench_ret = prices[benchmark].pct_change().dropna()
        bench_ann_return = (1 + bench_ret.mean()) ** periods_per_year - 1
        bench_ann_risk = bench_ret.std() * np.sqrt(periods_per_year)
        excess_return = ann_return - bench_ann_return
        # Information ratio
        tracking_error = np.std(port_ret - bench_ret)
        info_ratio = excess_return / tracking_error if tracking_error != 0 else np.nan
    else:
        bench_ann_return = bench_ann_risk = excess_return = info_ratio = np.nan

    # Max drawdown
    cumulative = (1 + port_ret).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    return {
        'annualized_return': ann_return,
        'annualized_risk': ann_risk,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'tail_risk_5pct': tail_risk,
        'max_drawdown': max_drawdown,
        'benchmark_annualized_return': bench_ann_return,
        'benchmark_annualized_risk': bench_ann_risk,
        'excess_return': excess_return,
        'information_ratio': info_ratio,
        'weights': dict(zip(tickers, w)),
    }


def optimize_portfolio(prices: pd.DataFrame, metric: str = 'sharpe_ratio', benchmark: str = None, risk_free_rate: float = 0.0, periods_per_year: int = 252, bounds: tuple = (0, 1)):
    """
    Find the optimal portfolio weights to maximize (or minimize) a given metric.
    Args:
        prices: DataFrame of asset prices (columns: tickers).
        metric: Which metric to optimize ('sharpe_ratio', 'sortino_ratio', 'annualized_return', 'annualized_risk', etc.).
        benchmark: optional benchmark ticker in prices.
        risk_free_rate: annual risk-free rate (as decimal).
        periods_per_year: trading periods per year (default 252 for daily).
        bounds: tuple for min/max weight per asset (default (0,1)).
    Returns:
        dict with optimal weights and the metric value.
    """
    tickers = [t for t in prices.columns if t != benchmark]
    n = len(tickers)

    def obj(w):
        w = np.array(w)
        w = w / w.sum()  # Ensure weights sum to 1
        weights_dict = dict(zip(tickers, w))
        metrics = portfolio_performance(prices, weights_dict, benchmark, risk_free_rate, periods_per_year)
        val = metrics.get(metric, np.nan)
        # For risk, minimize; for return/Sharpe/Sortino, maximize (so minimize negative)
        if metric in ['annualized_risk', 'max_drawdown', 'tail_risk_5pct']:
            return val if not np.isnan(val) else 1e6
        else:
            return -val if not np.isnan(val) else 1e6

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bnds = [bounds] * n
    w0 = np.array([1.0 / n] * n)
    result = minimize(obj, w0, method='SLSQP', bounds=bnds, constraints=constraints)
    opt_w = result.x / result.x.sum()
    weights_dict = dict(zip(tickers, opt_w))
    metrics = portfolio_performance(prices, weights_dict, benchmark, risk_free_rate, periods_per_year)
    return {
        'optimal_weights': weights_dict,
        'metric_value': metrics.get(metric, np.nan),
        'all_metrics': metrics,
        'success': result.success,
        'message': result.message
    }