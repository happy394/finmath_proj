# model.py
from __future__ import annotations

import numpy as np
import pandas as pd
from mesa import Model
from mesa.time import RandomActivation

from market import Market
from agents import (
    Fundamentalist, Chartist, NoiseTrader,
    MarketMakerL2,
    BSOptionTrader, HestonOptionTrader
)


class ABMModel(Model):
    def __init__(
        self,
        seed: int = 1,
        n_fund: int = 50,
        n_chart: int = 20,
        n_noise: int = 30,
        n_opt_total: int = 40,
        p_heston_opt: float = 0.5,
        # Market params
        S0: float = 100.0,
        sigma_bs: float = 0.20,
        dt: float = 1.0/252.0,
        impact_lambda: float = 0.02,
        kappa_iv: float = 0.02,
        # Heston proxy dynamics (simple endogenous vol state)
        heston_kappa: float = 0.20,
        heston_sigma: float = 0.30,
        heston_floor: float = 0.05,
        heston_cap: float = 2.0,
        # Smile
        K_multipliers=(0.9, 1.0, 1.1),
        smile_beta: float = 0.6,
        # Fundamental anchor
        S_f: float = 100.0,
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        self.p_heston_opt = float(p_heston_opt)
        self.S_f = float(S_f)

        self.market = Market(
            S0=S0,
            dt=dt,
            sigma_bs=sigma_bs,
            impact_lambda=impact_lambda,
            kappa_iv=kappa_iv,
            seed=seed,
        )

        self.heston_kappa = float(heston_kappa)
        self.heston_sigma = float(heston_sigma)
        self.heston_floor = float(heston_floor)
        self.heston_cap = float(heston_cap)

        self.K_multipliers = tuple(K_multipliers)
        self.smile_beta = float(smile_beta)

        self.schedule = RandomActivation(self)

        # storage
        self.t = 0
        self.returns: list[float] = []
        self._pending_stock_q = 0.0
        self._pending_opt_q = 0.0

        # agents
        uid = 0
        for _ in range(n_fund):
            a = Fundamentalist(uid, self, S_f=S_f)
            self.schedule.add(a); uid += 1

        for _ in range(n_chart):
            a = Chartist(uid, self)
            self.schedule.add(a); uid += 1

        for _ in range(n_noise):
            a = NoiseTrader(uid, self)
            self.schedule.add(a); uid += 1

        # market maker (quotes)
        mm = MarketMakerL2(uid, self)
        self.schedule.add(mm); uid += 1

        # option traders
        n_heston = int(round(n_opt_total * self.p_heston_opt))
        n_bs = int(n_opt_total - n_heston)

        for _ in range(n_bs):
            a = BSOptionTrader(uid, self, sigma_bs=sigma_bs, risk_aversion=1.0, q_opt_max=2.0, hedge_k=0.25)
            self.schedule.add(a); uid += 1

        for _ in range(n_heston):
            a = HestonOptionTrader(uid, self, react_k=1.0, risk_aversion=1.0, q_opt_max=2.0, hedge_k=0.25)
            self.schedule.add(a); uid += 1

        self.history = []

    # order submission API
    def submit_stock_order(self, q: float):
        self._pending_stock_q += float(q)

    def submit_option_order(self, q_opt: float):
        self._pending_opt_q += float(q_opt)

    # helper metrics
    def _rv20(self):
        if len(self.returns) < 20:
            return np.nan
        arr = np.array(self.returns[-20:], dtype=float)
        return float(np.std(arr, ddof=1))

    def step(self):
        # 1) reset per-step flows + pending
        self.market.reset_flows()
        self._pending_stock_q = 0.0
        self._pending_opt_q = 0.0

        S_prev = float(self.market.S)

        # 2) agents decide (they call submit_*())
        self.schedule.step()

        # 3) execute net stock flow once (cleaner dynamics)
        self.market.update_quotes()
        q_stock = float(np.clip(self._pending_stock_q, -25.0, 25.0))
        self.market.execute_stock_market_order(q_stock)

        # 4) option flow updates IV
        self.market.option_order_flow = float(np.clip(self._pending_opt_q, -25.0, 25.0))
        self.market.update_iv_from_flow()

        # 5) update "Heston instantaneous vol proxy" эндогенно от движений
        # простая динамика: sigma_heston_inst тянется к |ret| и шумит
        S_now = float(self.market.S)
        ret = float(np.log(max(S_now, 1e-9) / max(S_prev, 1e-9)))
        self.returns.append(ret)

        target = abs(ret) / max(self.market.dt**0.5, 1e-12)  # rough scale to per-year-ish
        # smooth + noise
        sig = float(self.market.sigma_heston_inst)
        sig = sig + self.heston_kappa * (target - sig) * self.market.dt + self.heston_sigma * self.rng.normal(0.0, 1.0) * (self.market.dt**0.5)
        sig = float(np.clip(sig, self.heston_floor, self.heston_cap))
        self.market.sigma_heston_inst = sig

        # 6) rebuild option surface (K_low, ATM, K_high)
        self.market.build_option_surface(
            p_heston_opt=self.p_heston_opt,
            K_multipliers=self.K_multipliers,
            smile_beta=self.smile_beta
        )

        # 7) record
        rv20 = self._rv20()
        rv20_ann = rv20 * np.sqrt(252) if np.isfinite(rv20) else np.nan

        # strikes mapping
        Ks = self.market.K_grid
        K_low, K_atm, K_high = Ks[0], Ks[1], Ks[2]

        iv_atm = float(self.market.iv_surface[K_atm])
        iv_low = float(self.market.iv_surface[K_low])
        iv_high = float(self.market.iv_surface[K_high])

        C_atm = float(self.market.call_prices[K_atm])

        iv_low_minus_atm = iv_low - iv_atm
        iv_high_minus_atm = iv_high - iv_atm
        smile_strength = 0.5 * (abs(iv_low_minus_atm) + abs(iv_high_minus_atm))

        rv_minus_iv = (rv20_ann - iv_atm) if np.isfinite(rv20_ann) else np.nan

        self.history.append({
            "t": self.t,
            "S": S_now,
            "ret": ret,
            "rv20": rv20,
            "iv_atm": iv_atm,
            "C_atm": C_atm,
            "iv_low": iv_low,
            "iv_high": iv_high,
            "iv_low_minus_atm": iv_low_minus_atm,
            "iv_high_minus_atm": iv_high_minus_atm,
            "p_heston_opt": self.p_heston_opt,
            "rv20_ann": rv20_ann,
            "rv_minus_iv": rv_minus_iv,
            "smile_strength": smile_strength,
        })

        self.t += 1

    def run(self, n_steps: int = 1000) -> pd.DataFrame:
        for _ in range(n_steps):
            self.step()
        return pd.DataFrame(self.history)
