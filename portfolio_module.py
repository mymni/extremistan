import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm, skew, kurtosis
from scipy.stats.mstats import gmean
import datetime

def compute_relative_power_sharpe(returns_series):
    if returns_series.empty:
        return np.nan
    target_value = np.log(4)
    def objective(P_X):
        return (np.mean(np.log(1 + (returns_series / P_X)**2)) - target_value)**2
    initial_guess = 1.0
    result = minimize(objective, initial_guess, bounds=[(1e-9, None)])
    if result.success and not np.isnan(result.x[0]):
        P_X = float(result.x[0])
        mean_return = returns_series.mean()
        return (mean_return / P_X) if P_X > 0 else np.nan
    else:
        return np.nan

def objective_sharpe(weights, daily_ret):
    portfolio_return = np.dot(weights, daily_ret.mean()) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(daily_ret.cov() * 252, weights)))
    return -(portfolio_return / (portfolio_volatility + 1e-12))

def objective_cvar(weights, daily_ret):
    port_ret = np.dot(daily_ret, weights)
    mean_annual = port_ret.mean() * 252
    std_annual  = port_ret.std()  * np.sqrt(252)
    conf_level = 0.05
    cvar_metric = mean_annual - std_annual * norm.ppf(conf_level)
    return -cvar_metric

def objective_sortino(weights, daily_ret):
    port_daily_annualized = np.dot(daily_ret, weights) * 252
    downside = port_daily_annualized[port_daily_annualized < 0]
    if len(downside) == 0:
        return -1e9
    downside_std = downside.std()
    sortino_ratio = port_daily_annualized.mean() / (downside_std + 1e-12)
    return -sortino_ratio

def objective_variance(weights, daily_ret):
    cov = daily_ret.cov() * 252
    return np.dot(weights.T, np.dot(cov, weights))

def objective_relative_power_sharpe(weights, daily_ret):
    port_returns = pd.Series(np.dot(daily_ret, weights), index=daily_ret.index)
    rps = compute_relative_power_sharpe(port_returns)
    return -rps

def dynamic_investment(full_rets, study_start_date, study_end_date, invest_start_date, invest_end_date,
                       lookback_years, freq, obj_func, init_capital, symbols):
    # Set up initial guess, bounds, and constraints based on number of assets
    n = len(symbols)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((-1, 1) for _ in range(n))
    init_guess = np.array([1.0 / n] * n)

    # Create a date index for the investment period
    dates_idx = pd.date_range(start=invest_start_date, end=invest_end_date, freq=freq)
    if dates_idx[-1] < pd.to_datetime(invest_end_date):
        dates_idx = dates_idx.append(pd.DatetimeIndex([pd.to_datetime(invest_end_date)]))
    all_dates = []
    all_values = []
    capital = init_capital
    current_weights = None

    for i in range(1, len(dates_idx)):
        rebal_date = dates_idx[i]
        prev_date = dates_idx[i - 1]
        # Ensure the lookback period does not start before the study period begins:
        lookback_start = max(pd.to_datetime(study_start_date), rebal_date - pd.DateOffset(years=lookback_years))
        window_data = full_rets.loc[lookback_start: rebal_date - pd.Timedelta(days=1)]
        if len(window_data) == 0:
            continue
        result = minimize(lambda w: obj_func(w, window_data),
                          init_guess, method='SLSQP',
                          bounds=bounds, constraints=cons)
        new_weights = result.x if result.success else init_guess

        if current_weights is None:
            current_weights = new_weights

        daily_range = full_rets.loc[prev_date: rebal_date - pd.Timedelta(days=1)]
        if len(daily_range) > 0:
            port_ret_series = daily_range.dot(current_weights)
            growth = (1 + port_ret_series).cumprod()
            sub_port_values = capital * growth
            for dt, val in sub_port_values.items():
                all_dates.append(dt)
                all_values.append(val)
            capital = sub_port_values.iloc[-1]
        current_weights = new_weights

    out_series = pd.Series(data=all_values, index=all_dates).sort_index()
    return out_series

def run_simulation(study_start_date, study_end_date, invest_start_date, invest_end_date,
                   symbols, objective_choice='Sharpe', initial_investment=10000,
                   lookback_years=2, rebalance_freq='Q'):
    # Download full data from study_start_date to invest_end_date.
    full_data = yf.download(symbols, start=study_start_date, end=invest_end_date)['Close']
    full_returns = full_data.pct_change().dropna()

    # Set up available objective functions.
    objectives = {
        'Sharpe': objective_sharpe,
        'CVaR': objective_cvar,
        'Sortino': objective_sortino,
        'Variance': objective_variance,
        'PowerSharpe': objective_relative_power_sharpe
    }
    if objective_choice not in objectives:
        selected_objective = objective_sharpe
    else:
        selected_objective = objectives[objective_choice]

    # Run the dynamic investment simulation.
    result_series = dynamic_investment(full_returns, study_start_date, study_end_date,
                                       invest_start_date, invest_end_date, lookback_years,
                                       rebalance_freq, selected_objective, initial_investment, symbols)
    final_value = result_series.iloc[-1] if len(result_series) > 0 else None

    # Create a plot.
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(result_series) > 0:
        ax.plot(result_series.index, result_series, label=f"Dynamic {objective_choice}")
    else:
        ax.text(0.5, 0.5, "No rebalancing data produced.", horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
    ax.set_title("Dynamic Portfolio Value Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True)

    return final_value, fig
