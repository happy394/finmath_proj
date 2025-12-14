# market.py
from __future__ import annotations

import math
import numpy as np


# Black–Scholes helpers
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    sigma = max(sigma, 1e-12)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return 0.0
    sigma = max(sigma, 1e-12)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return S * pdf * math.sqrt(T)


def implied_vol_call(C_mkt: float, S: float, K: float, T: float, r: float,
                     init: float = 0.2, tol: float = 1e-6, max_iter: int = 50) -> float:
    C_mkt = max(C_mkt, max(S - K * math.exp(-r * T), 0.0))
    sigma = max(init, 1e-4)

    for _ in range(max_iter):
        C = bs_call_price(S, K, T, r, sigma)
        v = bs_vega(S, K, T, r, sigma)
        diff = C - C_mkt
        if abs(diff) < tol:
            return max(sigma, 1e-6)
        if v < 1e-10:
            break
        sigma -= diff / v
        sigma = float(np.clip(sigma, 1e-6, 5.0))

    lo, hi = 1e-6, 5.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        C = bs_call_price(S, K, T, r, mid)
        if C > C_mkt:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ----------------------------
# Market microstructure
# ----------------------------
class Market:
    """
    Stock + synthetic option surface.
    Stock price updated by net order flow (impact).
    Option IV updated by option flow.
    """

    def __init__(
        self,
        S0: float = 100.0,
        r: float = 0.0,
        T_option: float = 1.0,
        dt: float = 1.0 / 252.0,
        sigma_bs: float = 0.20,
        kappa_iv: float = 0.02,
        impact_lambda: float = 0.02,
        spread0: float = 0.02,
        spread_alpha: float = 0.10,
        seed: int | None = None,
    ):
        self.rng = np.random.default_rng(seed)

        self.S = float(S0)
        self.r = float(r)
        self.T = float(T_option)
        self.dt = float(dt)

        # Vol beliefs (annualized)
        self.sigma_bs = float(sigma_bs)
        self.sigma_heston_inst = float(sigma_bs)

        # Market IV (annualized)
        self.iv_atm = float(sigma_bs)

        self.stock_order_flow = 0.0
        self.option_order_flow = 0.0

        self.impact_lambda = float(impact_lambda)
        self.kappa_iv = float(kappa_iv)
        self.spread0 = float(spread0)
        self.spread_alpha = float(spread_alpha)

        self.bid = self.S - self.spread0
        self.ask = self.S + self.spread0

        self.K_grid = None
        self.call_prices = {}
        self.iv_surface = {}

    def update_quotes(self):
        s_t = self.spread0 + self.spread_alpha * abs(self.stock_order_flow)
        self.bid = self.S - s_t
        self.ask = self.S + s_t

    def execute_stock_market_order(self, q: float):
        q = float(q)
        if q == 0.0:
            return None

        self.stock_order_flow += q
        px = self.ask if q > 0 else self.bid

        # impact update (this is the only driver of S)
        self.S = max(1e-6, self.S + self.impact_lambda * q)

        return px

    def update_iv_from_flow(self):
        self.iv_atm = float(np.clip(self.iv_atm + self.kappa_iv * self.option_order_flow, 0.01, 2.0))

    def build_option_surface(self, p_heston_opt: float, K_multipliers=(0.9, 1.0, 1.1), smile_beta: float = 0.6):
        S = float(self.S)
        self.K_grid = [float(m * S) for m in K_multipliers]

        dispersion = abs(self.sigma_heston_inst - self.sigma_bs)
        base = float(self.iv_atm)

        self.call_prices = {}
        self.iv_surface = {}

        for K in self.K_grid:
            mny = abs(math.log(K / S))
            amp = float(smile_beta) * dispersion * float(p_heston_opt)
            ivK = float(np.clip(base + amp * mny, 0.01, 2.0))
            CK = bs_call_price(S, K, self.T, self.r, ivK)
            self.iv_surface[K] = ivK
            self.call_prices[K] = CK

    def reset_flows(self):
        self.stock_order_flow = 0.0
        self.option_order_flow = 0.0
