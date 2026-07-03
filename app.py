"""Interactive Streamlit dashboard for FFT option pricing.

Run with:  streamlit run app.py

Lets you vary the Variance-Gamma skew (theta) and kurtosis (nu) and see, in real
time, how the implied-volatility smile emerges and where Black-Scholes mis-prices
options relative to the fat-tailed VG model.
"""
import numpy as np
import streamlit as st

from pricing_engine import (
    FFTConfig,
    black_scholes_call,
    carr_madan_call_curve,
    vg_char_function,
)

st.set_page_config(page_title="FFT Option Pricing", layout="wide")
st.title("Option Pricing via the Carr-Madan FFT (Variance-Gamma model)")

with st.sidebar:
    st.header("Market parameters")
    S0 = st.number_input("Spot price S0", value=100.0, step=1.0)
    r = st.number_input("Risk-free rate r", value=0.05, step=0.01, format="%.4f")
    T = st.number_input("Maturity T (years)", value=0.50, step=0.05, format="%.2f")

    st.header("Variance-Gamma parameters")
    sigma = st.slider("sigma (volatility)", 0.05, 0.60, 0.20, 0.01)
    theta = st.slider("theta (skewness)", -0.50, 0.20, -0.14, 0.01)
    nu = st.slider("nu (kurtosis)", 0.01, 1.00, 0.20, 0.01)

    st.header("FFT settings")
    alpha = st.slider("alpha (damping)", 0.5, 3.0, 1.5, 0.1)
    logN = st.select_slider("N = 2^k", options=list(range(8, 15)), value=12)

cfg = FFTConfig(alpha=alpha, N=2 ** logN, eta=0.25)

# price the full strike curve in a single FFT
strikes, vg_prices = carr_madan_call_curve(
    vg_char_function, r, T, cfg, S0=S0, sigma=sigma, theta=theta, nu=nu)

# restrict to a sensible strike window around the spot
mask = (strikes > 0.5 * S0) & (strikes < 1.6 * S0)
strikes = strikes[mask]
vg_prices = np.maximum(vg_prices[mask], 0.0)
bs_prices = np.asarray(black_scholes_call(S0, strikes, T, r, sigma), dtype=float)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Call price vs strike")
    import pandas as pd
    df = pd.DataFrame({"Variance-Gamma (FFT)": vg_prices,
                       "Black-Scholes": bs_prices}, index=np.round(strikes, 2))
    df.index.name = "Strike K"
    st.line_chart(df)

with col2:
    st.subheader("VG − BS price difference")
    diff = pd.DataFrame({"VG − BS": vg_prices - bs_prices},
                        index=np.round(strikes, 2))
    diff.index.name = "Strike K"
    st.bar_chart(diff)
    st.caption("Negative theta (left skew) makes VG price out-of-the-money puts / "
               "deep OTM calls differently from Black-Scholes — the smile effect.")

# point price at the money
atm = float(np.interp(S0, strikes, vg_prices))
atm_bs = float(black_scholes_call(S0, S0, T, r, sigma))
m1, m2, m3 = st.columns(3)
m1.metric("ATM call — Variance-Gamma", f"{atm:.4f}")
m2.metric("ATM call — Black-Scholes", f"{atm_bs:.4f}")
m3.metric("VG − BS", f"{atm - atm_bs:+.4f}")

st.info("As theta → 0 and nu → 0 the Variance-Gamma model collapses to "
        "Black-Scholes and the two curves coincide.")
