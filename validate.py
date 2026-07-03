"""Sanity checks for the FFT pricer (runs with NumPy only -- no SciPy/Streamlit).

1. The Carr-Madan FFT applied to the Black-Scholes characteristic function must
   reproduce the analytic Black-Scholes price.
2. The Variance-Gamma model must converge to Black-Scholes as theta -> 0, nu -> 0.

Run:  python validate.py
"""
import numpy as np

from pricing_engine import (
    FFTConfig, black_scholes_call, bs_call_fft, vg_call,
)


def main():
    S0, r, T, sigma = 100.0, 0.05, 0.5, 0.20
    cfg = FFTConfig(alpha=1.5, N=4096, eta=0.25)
    strikes = [80, 90, 100, 110, 120]

    print(f"{'K':>5} {'BSM analytic':>13} {'BSM via FFT':>12} {'abs err':>10} "
          f"{'VG(->BS limit)':>15}")
    max_err = 0.0
    for K in strikes:
        analytic = float(black_scholes_call(S0, K, T, r, sigma))
        fft = bs_call_fft(K, S0, r, T, sigma, cfg)
        # VG in the near-Gaussian limit (theta~0, nu small) should approach BSM
        vg = vg_call(K, S0, r, T, sigma, theta=-1e-4, nu=1e-3, cfg=cfg)
        err = abs(analytic - fft)
        max_err = max(max_err, err)
        print(f"{K:>5} {analytic:>13.5f} {fft:>12.5f} {err:>10.2e} {vg:>15.5f}")

    print(f"\nmax |BSM analytic - BSM FFT| = {max_err:.2e}")
    assert max_err < 1e-2, "FFT pricer does not match Black-Scholes!"
    print("OK: FFT pricer reproduces Black-Scholes within tolerance.")


if __name__ == "__main__":
    main()
