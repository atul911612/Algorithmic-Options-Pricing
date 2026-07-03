# Option Pricing using the Fast Fourier Transform (FFT)

European option pricing in the **frequency domain** using the **Carr–Madan FFT**
method with a **Variance-Gamma (VG)** model, benchmarked against **Black–Scholes–
Merton**. An interactive **Streamlit** dashboard lets you vary the VG skew/kurtosis
and watch the volatility smile emerge and Black–Scholes mis-pricing appear in real
time.

## Why FFT + Variance-Gamma?

- **Black–Scholes** assumes constant volatility and Gaussian log-returns, so it
  cannot reproduce the **volatility smile** seen in real markets.
- **Variance-Gamma** is a pure-jump process with three parameters — `sigma`
  (volatility), `nu` (kurtosis) and `theta` (skewness) — that captures fat tails
  and skew.
- Its probability density has no closed form, but its **characteristic function**
  does. The **Carr–Madan (1999)** method prices the *entire strike curve* from that
  characteristic function in a single **O(N log N)** FFT.

## Method

For a damping factor `alpha`, the call price is

```
c(k) = exp(-alpha·k)/pi · ∫₀^∞ exp(-i·v·k) · psi(v) dv ,   k = ln(K)
psi(v) = exp(-rT) · phi(v-(alpha+1)i) / (alpha² + alpha - v² + i(2alpha+1)v)
```

where `phi` is the characteristic function of `ln(S_T)`. The integral is
discretized with **Simpson's rule** on the constraint `lambda·eta = 2π/N` and
evaluated with the FFT (`numpy.fft` / `scipy.fft`).

## Files

| File | Purpose |
|---|---|
| `pricing_engine.py` | Core: Black–Scholes analytic + VG/BS characteristic functions + Carr–Madan FFT pricer (NumPy-only, no SciPy needed) |
| `app.py` | Streamlit dashboard (price curves, VG−BS difference, ATM metrics) |
| `validate.py` | Sanity checks: FFT reproduces analytic Black–Scholes; VG → BS in the Gaussian limit |
| `requirements.txt` | Dependencies |

## Setup & usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) verify the pricer is correct (NumPy only)
python validate.py

# 2) launch the interactive dashboard
streamlit run app.py
```

### Validation output (already verified)

```
    K  BSM analytic  BSM via FFT    abs err
   80      22.17456     22.17470   1.37e-04
   90      13.49852     13.49926   7.46e-04
  100       6.88873      6.89001   1.28e-03
  110       2.90647      2.90674   2.68e-04
  120       1.02262      1.02317   5.58e-04
max |BSM analytic - BSM FFT| = 1.28e-03  -> OK
```

## Key parameters

| Parameter | Meaning | Effect |
|---|---|---|
| `sigma` | VG volatility | overall option level |
| `theta` | skewness | `theta < 0` ⇒ left skew (equity-like), BS under-prices OTM |
| `nu` | kurtosis / jump intensity | fatter tails as `nu` grows |
| `alpha` | FFT damping | 1.1–1.75 typical; makes the payoff integrable |
| `N`, `eta` | FFT grid | accuracy vs. strike resolution (`lambda·eta = 2π/N`) |

## References

- P. Carr, D. Madan, *Option valuation using the fast Fourier transform*,
  Journal of Computational Finance, 1999.
- D. Madan, P. Carr, E. Chang, *The Variance Gamma Process and Option Pricing*,
  European Finance Review, 1998.
