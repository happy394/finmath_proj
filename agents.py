# agents.py
from __future__ import annotations

import numpy as np
from mesa import Agent


# Base stock traders
class Fundamentalist(Agent):
    """
    Buys if S < S_f, sells if S > S_f.
    q = k*(S_f - S) + N(0, sigma_q)
    """

    def __init__(self, unique_id, model, S_f: float = 100.0, k: float = 0.08, sigma_q: float = 0.6, q_clip: float = 5.0):
        super().__init__(unique_id, model)
        self.S_f = float(S_f)
        self.k = float(k)
        self.sigma_q = float(sigma_q)
        self.q_clip = float(q_clip)

    def step(self):
        S = float(self.model.market.S)
        mispricing = self.S_f - S
        q = self.k * mispricing + self.model.rng.normal(0.0, self.sigma_q)
        q = float(np.clip(q, -self.q_clip, self.q_clip))
        self.model.submit_stock_order(q)


class Chartist(Agent):
    """
    Momentum trader based on last L returns.
    trend = sum r_{t-i}, i=1..L
    q = k*sign(trend) + noise
    """

    def __init__(self, unique_id, model, L: int = 10, k: float = 1.5, sigma_q: float = 0.4, q_clip: float = 5.0):
        super().__init__(unique_id, model)
        self.L = int(L)
        self.k = float(k)
        self.sigma_q = float(sigma_q)
        self.q_clip = float(q_clip)

    def step(self):
        rets = self.model.returns
        if len(rets) < self.L:
            return
        trend = float(np.sum(rets[-self.L:]))
        q = self.k * np.sign(trend) + self.model.rng.normal(0.0, self.sigma_q)
        q = float(np.clip(q, -self.q_clip, self.q_clip))
        self.model.submit_stock_order(q)


class NoiseTrader(Agent):
    """
    Random buy/sell independent from price.
    """

    def __init__(self, unique_id, model, q_max: float = 2.0):
        super().__init__(unique_id, model)
        self.q_max = float(q_max)

    def step(self):
        side = 1.0 if self.model.rng.random() < 0.5 else -1.0
        q = side * self.model.rng.uniform(0.0, self.q_max)
        self.model.submit_stock_order(float(q))


# Market Maker
class MarketMakerL2(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        self.model.market.update_quotes()


# Option traders: BS vs Heston
class OptionTraderBase(Agent):
    def __init__(self, unique_id, model, risk_aversion: float = 1.0, q_opt_max: float = 2.0, hedge_k: float = 0.25):
        super().__init__(unique_id, model)
        self.risk_aversion = float(risk_aversion)
        self.q_opt_max = float(q_opt_max)
        self.hedge_k = float(hedge_k)

    def model_iv(self) -> float:
        raise NotImplementedError

    def step(self):
        """
        Trade ATM call based on IV difference:
          if my_iv > mkt_iv -> buy calls (q_opt>0)
          else sell calls (q_opt<0)
        Hedge via small opposite stock flow.
        """
        mkt_iv = float(self.model.market.iv_atm)
        my_iv = float(self.model_iv())

        diff = my_iv - mkt_iv
        q_opt = float(np.clip(diff / max(self.risk_aversion, 1e-6), -self.q_opt_max, self.q_opt_max))

        if abs(q_opt) < 1e-6:
            return

        self.model.submit_option_order(q_opt)

        # simple hedge proxy (small to not dominate stock)
        q_stock = -self.hedge_k * np.sign(q_opt)
        self.model.submit_stock_order(float(q_stock))


class BSOptionTrader(OptionTraderBase):
    def __init__(self, unique_id, model, sigma_bs: float = 0.20, **kwargs):
        super().__init__(unique_id, model, **kwargs)
        self.sigma_bs = float(sigma_bs)

    def model_iv(self) -> float:
        return self.sigma_bs


class HestonOptionTrader(OptionTraderBase):
    def __init__(self, unique_id, model, react_k: float = 1.0, **kwargs):
        super().__init__(unique_id, model, **kwargs)
        self.react_k = float(react_k)

    def model_iv(self) -> float:
        return float(np.clip(self.react_k * self.model.market.sigma_heston_inst, 0.01, 2.0))
