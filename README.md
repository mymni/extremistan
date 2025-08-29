
# Trading in Extremistan

A Python toolbox for robust financial modeling and heavy-tailed risk analysis.  
Leverages parametric distribution fitting, tail-sensitive risk measures, heavy-tailed MRF clustering, and portfolio optimization—packaged as both a CLI and importable library, with interactive demos via Jupyter and Streamlit.

---

## 📘 Project Overview

**Team:**  Nadim Succar, Mahmoud Yamani, Mahdi Zwain <br/>
**Advisors:** Prof. Jihad Fahs and Prof. Ibrahim Abou-Faycal  
**Sponsor:** AlgoTraders  
**Institution:** American University of Beirut, Final Year Project, Spring 2025  

Inspired by Nassim Taleb’s _The Black Swan_, this project challenges the Gaussian assumptions behind classical risk modeling. It targets financial "Extremistan"—where fat tails and extreme events dominate the behavior of financial markets. The toolbox provides scalable, modular software for heavy-tailed modeling, tailored to financial practitioners and quantitative researchers.

---

## 🚀 Quick Start

To run the interactive Streamlit demo:

```bash
streamlit run app.py
```

---

## 📚 Theoretical Background

Most classical tools assume thin-tailed (Gaussian) return distributions, underestimating the impact of rare but catastrophic events. Instead, this toolbox embraces a heavy-tailed modeling approach. Core theoretical components include:

- **Non-Gaussian Parametric Fitting:** Models like Student-t, Skewed Normal, and GED distributions are fit using Maximum Likelihood Estimation.
- **Tail-Sensitive Risk Metrics:** Including EVaR, RLVaR, Ulcer Index, Relative Power Sharpe (our proposed risk measure), and others to capture downside and extreme risk.
- **Heavy-Tailed Dependency Modeling:** Student-t Markov Random Fields (MRFs) cluster assets based on co-extremal dependencies, overcoming limitations of Gaussian copulas.
- **Robust Portfolio Optimization:** Incorporates tail-aware risk into optimization using constrained solvers (SLSQP, Trust-Region) for both static and rolling-window strategies.

---

## 🚀 Features

### 🔹 Data Ingestion
- Fetches historical stock prices from Yahoo Finance via `yfinance`
- Computes log and simple returns for portfolio and risk analysis

### 🔹 Distribution Fitting
- Fits Normal, Student-t, GED, and Skewed-Normal via MLE
- Evaluates model quality using Kolmogorov-Smirnov tests, AIC, BIC, and GARCH

### 🔹 Risk Analysis
- Computes risk metrics: VaR, Standard Sharpe, Sortino, Ulcer Index, Entropic VaR, Relative Power Sharpe, Max Drawdown, Information Ratio, and more
- Supports both static and dynamic (rolling-window) analysis
- Ranks stocks by composite risk scores
- Estimates the optimal risk measure for every proposed trading strategy, and the optimal trading strategy for every computed risk measure

### 🔹 MRF-Based Dependency Modeling
- Learns heavy-tailed Markov Random Field graphs to detect clusters of co-dependent assets
- Provides interpretable adjacency matrices for robust diversification strategies

### 🔹 Portfolio Optimization
- Solves constrained optimization problems for various risk-return objectives
- Supports both static (in-sample) and dynamic (epoch-based) rebalancing
- Includes simulations to compare portfolio performance across risk measures

---

## 🧪 Experimental Findings

- **Student-t** consistently outperforms Gaussian fits in capturing real-world tail behavior pre- and post-COVID-19
- Tail-sensitive metrics such as Relative Power Sharpe outperform traditional ones under volatile conditions
- MRF clustering aligns with intuitive asset groupings, aiding in effective diversification
- Dynamic portfolios using tail-aware metrics show greater resilience in crisis periods

---

## 💻 Interfaces

- **CLI**: Execute workflows from the terminal using `argparse`-based commands
- **Library API**: Import modules for use in Python scripts and notebooks
- **Streamlit Demo**: Interactive frontend for exploring toolbox capabilities
- **Jupyter Bootstrap**: One-click setup notebook for all dependencies

---

## 🛠️ Technologies

- Python, NumPy, Pandas, SciPy, scikit-learn
- yfinance, Streamlit, matplotlib
- Custom implementations of MLE fitting, graph learning, and risk modeling
