# Trading in Extremistan

A Python toolbox for robust financial modeling and heavy-tailed risk analysis.  
Leverages parametric distribution fitting, tail-sensitive risk measures, heavy-tailed MRF clustering, and portfolio optimization—packaged as both a CLI and importable library, with interactive demos via Jupyter and Streamlit.

---

## 🚀 Features

- **Data Ingestion** (`data_collection`)  
  Fetch historical price series (via `yfinance`) and compute returns.

- **Distribution Fitting** (`fitting`)  
  MLE fitting of Normal, Student-t, GED, and Skewed-Normal; evaluates goodness-of-fit (K-S, AIC/BIC).

- **Risk Analysis** (`risk`)  
  Static & rolling-window measures: VaR, CVaR, Sharpe, Relative‐Power Sharpe, Entropic VaR, Ulcer Index, Maximum Drawdown, Information Ratio, and more.

- **Dependency Modeling** (`mrf`)  
  Learn heavy-tailed Markov Random Fields for asset clustering under extreme co-movements.

- **Portfolio Optimization** (`optimization`)  
  Static & epoch-based dynamic rebalancing with SLSQP and Trust-Region solvers; supports typical risk-return criteria.

- **CLI Interface**  
  Subcommands:  
  ```bash
  trading-extremistan fit    # fit distributions
  trading-extremistan rank   # compute & rank risk metrics
  trading-extremistan roll   # rolling-window risk analysis
  trading-extremistan mrf    # learn heavy-tailed MRF graph
  trading-extremistan optimize  # run portfolio optimizations
