import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
from scipy.stats import norm, t, skewnorm, gennorm, entropy
import matplotlib.pyplot as plt

# -----------------------------
# Risk Measure Functions
# -----------------------------
def compute_standard_sharpe(returns, trading_days_per_year=252):
    returns = returns.squeeze()
    if returns.empty or returns.std() == 0:
        return np.nan
    return (returns.mean() / returns.std()) * np.sqrt(trading_days_per_year)

def compute_relative_power_sharpe(returns):
    returns = returns.squeeze()
    if returns.empty:
        return np.nan
    target_value = np.log(4)
    annual_factor = np.sqrt(252)
    def objective(P_X):
        return (np.mean(np.log(1 + (returns / P_X) ** 2)) - target_value) ** 2
    from scipy.optimize import minimize
    result = minimize(objective, 1.0, bounds=[(0, None)])
    if result.success and not np.isnan(result.x[0]):
        P_X = float(result.x[0])
        return (np.mean(np.log(1 + returns)) / P_X) * annual_factor if P_X > 0 else np.nan
    else:
        return np.nan

def compute_snr_sharpe(returns, trading_days_per_year=252):
    returns = returns.squeeze()
    if returns.empty:
        return np.nan
    log_returns = np.log(1 + returns)
    mean_lr = log_returns.mean()
    std_error = log_returns.std() / np.sqrt(len(log_returns))
    if std_error == 0:
        return np.nan
    return (mean_lr / std_error) * np.sqrt(trading_days_per_year)

def compute_median_sharpe(returns, trading_days_per_year=252):
    returns = returns.squeeze()
    if returns.empty:
        return np.nan
    med = np.median(returns)
    mad = np.median(np.abs(returns - med))
    if mad == 0:
        return np.nan
    return (med / mad) * np.sqrt(trading_days_per_year)

def compute_sortino_ratio(returns, trading_days_per_year=252):
    returns = returns.squeeze()
    if returns.empty:
        return np.nan
    mean_ret = returns.mean()
    neg_returns = returns[returns < 0]
    if neg_returns.std() == 0:
        return np.nan
    return (mean_ret / neg_returns.std()) * np.sqrt(trading_days_per_year)

def compute_max_drawdown(returns):
    returns = returns.squeeze()
    if returns.empty:
        return np.nan
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    dd = (cum_returns - peak) / peak
    return dd.min()

def compute_information_ratio(returns, benchmark_returns, trading_days_per_year=252):
    returns = returns.squeeze()
    benchmark_returns = benchmark_returns.squeeze()
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return np.nan
    ex_ret = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    if ex_ret.std() == 0:
        return np.nan
    return (ex_ret.mean() / ex_ret.std()) * np.sqrt(trading_days_per_year)

def compute_mvar(returns, confidence_level=0.95):
    returns = returns.squeeze()
    if returns.empty:
        return np.nan
    mean_ret = returns.mean()
    std_dev = returns.std()
    z = norm.ppf(confidence_level)
    return mean_ret + z * std_dev

def compute_evar(returns, alpha=0.95):
    returns = returns.squeeze()
    if returns.empty:
        return np.nan
    z_vals = np.linspace(0.01, 10, 100)
    mgf = lambda t: np.mean(np.exp(t * returns))
    evar_vals = [(1 / t) * (np.log(mgf(t)) - np.log(alpha)) for t in z_vals if mgf(t) > 0]
    return min(evar_vals) if evar_vals else np.nan

def compute_rlvar(returns, alpha=0.95, kappa=0.5):
    returns = returns.squeeze()
    if returns.empty:
        return np.nan
    mgf_kappa = lambda t: np.mean((1 + t * returns / kappa) ** kappa)
    z_vals = np.linspace(0.01, 10, 100)
    rlvar_vals = [(1 / t) * (np.log(mgf_kappa(t)) - np.log(alpha)) for t in z_vals if mgf_kappa(t) > 0]
    return min(rlvar_vals) if rlvar_vals else np.nan

def compute_ulcer_index(returns):
    returns = returns.squeeze()
    if returns.empty:
        return np.nan
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    dd = (cum_returns - peak) / peak
    return np.sqrt(np.mean(dd ** 2))

# -----------------------------
# Main Function: ranking_plot
# -----------------------------
def ranking_plot(symbols, static_start, static_end, benchmark_symbol):
    """
    For a given list of stock symbols and a benchmark,
    fetches historical data between static_start and static_end,
    computes risk metrics for each stock, computes unified integer risk rankings,
    creates a heatmap figure (without the color bar), and returns that figure.

    Parameters:
      - symbols (list): List of stock tickers.
      - static_start (str): Start date ('YYYY-MM-DD').
      - static_end (str): End date ('YYYY-MM-DD').
      - benchmark_symbol (str): Benchmark ticker.

    Returns:
      - fig (matplotlib.figure.Figure): Heatmap figure of unified risk rankings.
    """
    import numpy as np
    import pandas as pd
    import yfinance as yf
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Fetch historical data and compute returns for each stock.
    returns_data = {}
    for symbol in symbols:
        data = yf.download(symbol, start=static_start, end=static_end, progress=False)
        if data.empty:
            print(f"No data for {symbol}")
            continue
        price_series = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        returns = price_series.pct_change().dropna()
        if returns.empty:
            print(f"No returns data for {symbol}")
            continue
        returns_data[symbol] = returns

    # Fetch benchmark data.
    benchmark_data = yf.download(benchmark_symbol, start=static_start, end=static_end, progress=False)
    if benchmark_data.empty:
        raise ValueError("No benchmark data available.")
    benchmark_series = benchmark_data['Adj Close'] if 'Adj Close' in benchmark_data.columns else benchmark_data['Close']
    benchmark_returns = benchmark_series.pct_change().dropna()

    # Define risk measure functions.
    risk_measures = {
        'Standard Sharpe': lambda r: compute_standard_sharpe(r),
        'Relative Power Sharpe': lambda r: compute_relative_power_sharpe(r),
        'SNR Sharpe': lambda r: compute_snr_sharpe(r),
        'Median Sharpe': lambda r: compute_median_sharpe(r),
        'Sortino Ratio': lambda r: compute_sortino_ratio(r),
        'Maximum Drawdown': lambda r: compute_max_drawdown(r),
        'Information Ratio': lambda r: compute_information_ratio(r, benchmark_returns),
        'MVaR': lambda r: compute_mvar(r),
        'EVaR': lambda r: compute_evar(r),
        'RLVaR': lambda r: compute_rlvar(r),
        'Ulcer Index': lambda r: compute_ulcer_index(r)
    }

    # Compute metrics for each risk measure.
    metrics_dict = {name: {} for name in risk_measures.keys()}
    for symbol, returns in returns_data.items():
        for name, func in risk_measures.items():
            try:
                metrics_dict[name][symbol] = func(returns)
            except Exception:
                metrics_dict[name][symbol] = np.nan

    metrics_df = pd.DataFrame(metrics_dict)

    # Invert metrics for which higher is better so that lower values imply lower risk.
    metrics_higher_is_better = ['Standard Sharpe', 'Relative Power Sharpe', 'SNR Sharpe', 'Median Sharpe', 'Sortino Ratio', 'Information Ratio']
    risk_scores = metrics_df.copy()
    for metric in risk_scores.columns:
        if metric in metrics_higher_is_better:
            risk_scores[metric] = -risk_scores[metric]

    unified_risk_ranking = risk_scores.rank(ascending=True, method='average')
    # Convert to integer ranks.
    unified_risk_ranking = unified_risk_ranking.round().astype(int)

    # Create heatmap figure without the color bar.
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(unified_risk_ranking, annot=True, fmt="d", cmap="coolwarm", linewidths=0.5, ax=ax, cbar=False)
    ax.set_title('Static Stock Rankings (1 = Least Risk)', fontsize=16)
    ax.set_xlabel('Risk Metrics', fontsize=14)
    ax.set_ylabel('Stocks', fontsize=14)
    plt.tight_layout()

    return fig
