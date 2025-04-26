import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
import rolling_basis_module    as rbm
import portfolio_module as pm
import ranking_module as rm
import yfinance as yf
import pandas as pd
import numpy as np


# Define a common list of stock options.
st.set_page_config(layout="wide")

st.title("Trading in Extremistan")

stock_options = ["AAPL","TSLA","JPM","WMT","JNJ","PEP","DIS","LMT","T","GOOGL"]

tabs = st.tabs([
    "Dynamic Portfolio Simulation",
    "Static Stock Ranking",
    "Distribution Fitting",
    "Rolling-Basis Risk"
])

# ---------------- Dynamic Simulation Tab ----------------
with tabs[0]:
    st.markdown("""
    ### Dynamic Portfolio Simulation
    Use the controls below to set the training (study) and testing periods, and select up to 8 stock tickers.
    """)
    study_start = st.date_input("Study Start Date (Training)", datetime(2009, 1, 1), key="dyn_start")
    study_end = st.date_input("Study End Date (Training) & Investment Start Date (Testing)", datetime(2019, 12, 1), key="dyn_end")
    invest_end = st.date_input("Investment End Date (Testing)", datetime(2020, 9, 1), key="dyn_invest_end")

    selected_tickers = st.multiselect("Select up to 8 Stock Tickers", options=stock_options,
                                      default=["AAPL", "TSLA", "JPM", "WMT", "JNJ", "PEP", "DIS", "LMT"], key="dyn_tickers")
    if len(selected_tickers) > 8:
        st.error("Please select at most 8 tickers.")

    objective_choice = st.selectbox("Select Objective Function",
                                    options=["Sharpe", "CVaR", "Sortino", "Variance", "PowerSharpe"], key="dyn_obj")

    if st.button("Run Dynamic Simulation", key="dyn_run"):
        if len(selected_tickers) > 8:
            st.error("Too many tickers selected. Please choose 8 or fewer.")
        else:
            study_start_date = study_start.strftime("%Y-%m-%d")
            study_end_date = study_end.strftime("%Y-%m-%d")
            invest_start_date = study_end_date  # As required.
            invest_end_date = invest_end.strftime("%Y-%m-%d")

            st.info("Running simulation. This may take a few moments...")
            final_value, fig = pm.run_simulation(
                study_start_date, study_end_date,
                invest_start_date, invest_end_date,
                selected_tickers,
                objective_choice=objective_choice,
                initial_investment=10000,
                lookback_years=2,
                rebalance_freq='Q'
            )
            if final_value is not None:
                st.success(f"Final Portfolio Value: ${final_value:,.2f}")
            else:
                st.error("Simulation did not produce any results. Check parameters and data availability.")
            st.pyplot(fig)

# ---------------- Static Ranking Tab ----------------
with tabs[1]:
    st.markdown("""
    ### Static Stock Ranking
    This demo computes unified risk rankings (displayed as a heatmap) across a full static period.
    """)
    static_start = st.date_input("Static Period Start Date", datetime(2018, 1, 1), key="stat_start")
    static_end = st.date_input("Static Period End Date", datetime(2021, 9, 30), key="stat_end")
    default_stocks = ["NEM", "GOLD", "APA"]
    ranking_symbols = st.multiselect("Select Stocks for Ranking",
                                     options=["NEM", "GOLD", "APA", "TSLA", "AAPL", "JPM"],
                                     default=default_stocks, key="stat_symbols")
    # Benchmark is hardcoded to "^GSPC" and not shown.
    benchmark_symbol = "^GSPC"

    if st.button("Show Ranking Heatmap", key="stat_run"):
        s_start = static_start.strftime("%Y-%m-%d")
        s_end = static_end.strftime("%Y-%m-%d")
        st.info("Fetching data and computing rankings. Please wait...")
        try:
            fig = rm.ranking_plot(ranking_symbols, s_start, s_end, benchmark_symbol)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error computing ranking: {e}")

# ---------------- Distribution Fitting Tab ----------------
with tabs[2]:
    st.markdown("### Distribution Fitting (Gaussian vs Student-t)")
    fit_stock = st.selectbox("Select Stock for Fitting Analysis", options=stock_options, key="fit_stock")
    fit_start = st.date_input("Fitting Analysis Start Date", datetime(2020, 1, 1), key="fit_start")
    fit_end = st.date_input("Fitting Analysis End Date", datetime(2020, 12, 31), key="fit_end")

    if st.button("Run Distribution Fitting", key="fit_run"):
         fit_start_date = fit_start.strftime("%Y-%m-%d")
         fit_end_date = fit_end.strftime("%Y-%m-%d")
         data_fit = yf.download(fit_stock, start=fit_start_date, end=fit_end_date, progress=False)
         if data_fit.empty:
             st.error("No data fetched for the selected stock.")
         else:
             price_series_fit = data_fit['Adj Close'] if 'Adj Close' in data_fit.columns else data_fit['Close']
             returns_fit = price_series_fit.pct_change().dropna()
             if returns_fit.empty:
                 st.error("No returns data computed.")
             else:
                 from scipy.stats import norm, t
                 norm_params = norm.fit(returns_fit)
                 t_params = t.fit(returns_fit)

                 fig_fit, ax_fit = plt.subplots(figsize=(10, 6))
                 ax_fit.hist(returns_fit, bins=50, density=True, alpha=0.6, color="gray", label="Empirical")
                 x_fit = np.linspace(returns_fit.min(), returns_fit.max(), 1000)
                 ax_fit.plot(x_fit, norm.pdf(x_fit, *norm_params), label="Gaussian Fit", color="blue", lw=2)
                 ax_fit.plot(x_fit, t.pdf(x_fit, *t_params), label="Student-t Fit", color="red", lw=2)
                 ax_fit.legend()
                 ax_fit.set_title(f"Distribution Fitting for {fit_stock}")
                 st.pyplot(fig_fit)

# ----------- Rolling-Basis Risk & Ranking Evolution -----------
with tabs[3]:
    st.header("Rolling-Basis Risk & Ranking Evolution")

    ticker_rb = st.selectbox("Select Ticker", stock_options, index=0, key="rbm_ticker")
    start_rb  = st.date_input("Start Date", datetime(2018, 1, 1), key="rbm_start")
    end_rb    = st.date_input("End Date",   datetime(2021, 9, 30), key="rbm_end")
    window_rb = st.slider("Rolling Window (days)", 30, 252, 126, key="rbm_window")

    all_measures = list(rbm.RISK_FUNCS.keys())
    chosen = st.multiselect("Which risk measures?", all_measures, default=["Standard Sharpe"])

    if st.button("Compute Rolling Risk", key="rbm_run"):
        # if you need a benchmark series for info ratio:
        benchmark = yf.download("^GSPC",
                               start=start_rb.strftime("%Y-%m-%d"),
                               end=end_rb.strftime("%Y-%m-%d"),
                               progress=False)

        risk_df, rank_df = rbm.compute_rolling_risk(
            ticker_rb,
            start_rb.strftime("%Y-%m-%d"),
            end_rb.strftime("%Y-%m-%d"),
            window=window_rb,
            measures=chosen,
            benchmark=benchmark
        )

        st.subheader("Raw Rolling Risk Measures")
        st.line_chart(risk_df[chosen], use_container_width=True)

        st.subheader("Rolling-Rankings")
        st.line_chart(rank_df[chosen], use_container_width=True)
