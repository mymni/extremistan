# Trading in Extremistan

A Python toolbox for robust financial modeling and heavy-tailed risk analysis.  
Leverages parametric distribution fitting, tail-sensitive risk measures, heavy-tailed MRF clustering, and portfolio optimization—packaged as both a CLI and importable library, with interactive demos via Jupyter and Streamlit.

---

## 🚀 Features

- **Data Ingestion**
  Fetch historical price series (via `yfinance`) and compute returns.

- **Distribution Fitting** 
  MLE fitting of Normal, Student-t, GED, and Skewed-Normal; evaluates goodness-of-fit (K-S, AIC/BIC).

- **Risk Analysis**
  Static & rolling-window measures: VaR, CVaR, Sharpe, Relative‐Power Sharpe, Entropic VaR, Ulcer Index, Maximum Drawdown, Information Ratio, and more.

- **Dependency Modeling**
  Learn heavy-tailed Markov Random Fields for asset clustering under extreme co-movements.

- **Portfolio Optimization** 
  Static & epoch-based dynamic rebalancing with SLSQP and Trust-Region solvers; supports typical risk-return criteria.
