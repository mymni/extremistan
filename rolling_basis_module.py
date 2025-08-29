
import numpy as np
import pandas as pd
import yfinance as yf

# import your risk measure functions here, e.g.:
# from trading_strategy_module import compute_rsi  # only if needed

import numpy as np
import pandas as pd
import yfinance as yf

# import each of your measure-functions:
from ranking_module import (
    compute_standard_sharpe,
    compute_sortino_ratio,
    compute_median_sharpe,
    compute_snr_sharpe,
    compute_relative_power_sharpe,
    compute_max_drawdown,
    compute_information_ratio,
    compute_mvar,
    compute_evar,
    compute_rlvar,
    compute_ulcer_index,
)

# Map human-friendly names → functions
RISK_FUNCS = {
    'Standard Sharpe': compute_standard_sharpe,
    'Sortino Ratio': compute_sortino_ratio,
    'Median Sharpe': compute_median_sharpe,
    'SNR Sharpe': compute_snr_sharpe,
    'Power Sharpe': compute_relative_power_sharpe,
    'Max Drawdown': compute_max_drawdown,
    'Information Ratio': compute_information_ratio,
    'MVaR': compute_mvar,
    'EVaR': compute_evar,
    'RLVaR': compute_rlvar,
    'Ulcer Index': compute_ulcer_index,
}

def compute_rolling_risk(ticker, start_date, end_date, window=126,
                         measures=None, benchmark=None):
    """
    If measures is None, compute all in RISK_FUNCS.
    If benchmark is needed by some measures (like info ratio), pass it here.
    """
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    price = df.get('Adj Close', df['Close'])
    daily_ret = price.pct_change().dropna()

    dates = daily_ret.index[window-1:]
    # decide which metrics to compute
    measures = measures or list(RISK_FUNCS.keys())
    risk_df = pd.DataFrame(index=dates, columns=measures, dtype=float)

    # if info ratio needs a benchmark series pre-computed:
    if 'Information Ratio' in measures and benchmark is not None:
        benchmark_price = benchmark.get('Adj Close', benchmark['Close'])
        benchmark_ret = benchmark_price.pct_change().dropna()
    else:
        benchmark_ret = None

    for dt in dates:
        window_ret = daily_ret.loc[:dt].tail(window)
        for name in measures:
            func = RISK_FUNCS[name]
            if name == 'Information Ratio':
                risk_df.at[dt, name] = func(window_ret, benchmark_ret)
            else:
                risk_df.at[dt, name] = func(window_ret)

    # compute rankings if you want to return those too
    rank_df = risk_df.rank(ascending=True, method='average')
    return risk_df, rank_df
