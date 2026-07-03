"""Option pricing engine: Carr-Madan FFT with a Variance-Gamma model.

Implements European option pricing in the frequency domain, following:

    P. Carr and D. Madan, "Option valuation using the fast Fourier transform",
    Journal of Computational Finance, 2(4), 1999.

The asset log-price is modelled with the Variance-Gamma (VG) process, a pure-jump
process whose three parameters control volatility (sigma), kurtosis (nu) and
skewness (theta) -- so it captures the fat tails and skew that the Gaussian
Black-Scholes model cannot. Black-Scholes is provided as an analytic benchmark.

Core maths uses only NumPy + the stdlib (so it runs without SciPy); the Streamlit
dashboard (app.py) adds the interactive UI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Black-Scholes-Merton analytic benchmark
# --------------------------------------------------------------------------- #
def _norm_cdf(x):
    """Standard normal CDF (vectorized, stdlib only)."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def black_scholes_call(S0, K, T, r, sigma):
    """Analytic European call price under Black-Scholes-Merton."""
    S0, K = float(S0), np.asarray(K, dtype=float)
    if T <= 0 or sigma <= 0:
        return np.maximum(S0 - K, 0.0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


# --------------------------------------------------------------------------- #
# Characteristic functions of the terminal log-price ln(S_T)
# --------------------------------------------------------------------------- #
def bs_char_function(u, S0, r, T, sigma):
    """Characteristic function of ln(S_T) under geometric Brownian motion."""
    return np.exp(
        1j * u * (np.log(S0) + (r - 0.5 * sigma ** 2) * T)
        - 0.5 * sigma ** 2 * u ** 2 * T
    )


def vg_char_function(u, S0, r, T, sigma, theta, nu):
    """Characteristic function of ln(S_T) under the Variance-Gamma model.

    sigma : diffusion/volatility of the VG process (>0)
    nu    : variance rate of the subordinating gamma clock -> controls kurtosis (>0)
    theta : drift of the Brownian motion -> controls skewness (theta<0 => left skew)
    """
    # martingale (risk-neutral) drift correction so E[S_T] = S0 e^{rT}
    omega = (1.0 / nu) * np.log(1.0 - theta * nu - 0.5 * sigma ** 2 * nu)
    drift = np.log(S0) + (r + omega) * T
    return np.exp(1j * u * drift) * (
        1.0 - 1j * theta * nu * u + 0.5 * sigma ** 2 * nu * u ** 2
    ) ** (-T / nu)


# --------------------------------------------------------------------------- #
# Carr-Madan FFT pricer
# --------------------------------------------------------------------------- #
@dataclass
class FFTConfig:
    alpha: float = 1.5      # damping factor (makes the payoff square-integrable)
    N: int = 4096           # FFT size (power of 2)
    eta: float = 0.25       # spacing in the frequency (integration) grid


def carr_madan_call_curve(char_func, r, T, cfg: FFTConfig = FFTConfig(), **cf_kwargs):
    """Return (strikes, call_prices) across a whole grid of strikes in one FFT.

    char_func(u, **cf_kwargs) is the characteristic function of ln(S_T).
    Complexity is O(N log N) for the entire strike curve.
    """
    N, eta, alpha = cfg.N, cfg.eta, cfg.alpha
    lam = 2.0 * math.pi / (N * eta)        # log-strike spacing (lambda * eta = 2pi/N)
    b = N * lam / 2.0                       # log-strikes centred on ln(S0)-ish
    j = np.arange(N)
    v = eta * j                             # integration grid (v_0 = 0)

    # Carr-Madan integrand psi(v) with the damped characteristic function
    numerator = np.exp(-r * T) * char_func(v - (alpha + 1) * 1j, r=r, T=T, **cf_kwargs)
    denominator = alpha ** 2 + alpha - v ** 2 + 1j * (2 * alpha + 1) * v
    psi = numerator / denominator

    # Simpson's-rule weights: [1, 4, 2, 4, ..., 4, 1] * (1/3)
    jj = np.arange(1, N + 1)
    delta = np.zeros(N)
    delta[0] = 1.0
    simpson = (3.0 + (-1.0) ** jj - delta) / 3.0

    integrand = np.exp(1j * b * v) * psi * eta * simpson
    fft_values = np.fft.fft(integrand)

    k = -b + lam * j                        # log-strikes
    call_prices = np.exp(-alpha * k) / math.pi * np.real(fft_values)
    strikes = np.exp(k)
    return strikes, call_prices


def price_call(K, char_func, r, T, cfg: FFTConfig = FFTConfig(), **cf_kwargs):
    """Price a single strike K by interpolating the FFT strike curve."""
    strikes, prices = carr_madan_call_curve(char_func, r, T, cfg, **cf_kwargs)
    return float(np.interp(K, strikes, prices))


# Convenience wrappers ------------------------------------------------------- #
def vg_call(K, S0, r, T, sigma, theta, nu, cfg: FFTConfig = FFTConfig()):
    return price_call(K, vg_char_function, r, T, cfg,
                      S0=S0, sigma=sigma, theta=theta, nu=nu)


def bs_call_fft(K, S0, r, T, sigma, cfg: FFTConfig = FFTConfig()):
    return price_call(K, bs_char_function, r, T, cfg, S0=S0, sigma=sigma)


if __name__ == "__main__":
    # quick demo
    S0, r, T, sigma = 100.0, 0.05, 0.5, 0.2
    for K in (90, 100, 110):
        analytic = float(black_scholes_call(S0, K, T, r, sigma))
        fft = bs_call_fft(K, S0, r, T, sigma)
        vg = vg_call(K, S0, r, T, sigma, theta=-0.14, nu=0.2)
        print(f"K={K:3d} | BSM analytic={analytic:7.4f} | BSM-FFT={fft:7.4f} "
              f"| VG-FFT={vg:7.4f}")
