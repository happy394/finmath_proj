# analysis.py
from __future__ import annotations

import numpy as np
import pandas as pd

from model import ABMModel


def run_one(
    p_heston_opt: float,
    n_steps: int = 1000,
    seed: int = 1,
    **kwargs
) -> pd.DataFrame:
    m = ABMModel(p_heston_opt=p_heston_opt, seed=seed, **kwargs)
    df = m.run(n_steps=n_steps)
    return df


def sweep_p_heston(
    p_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
    n_steps: int = 1000,
    seeds=(1, 2, 3, 4, 5),
    **kwargs
) -> pd.DataFrame:
    all_df = []
    for p in p_grid:
        for sd in seeds:
            df = run_one(p_heston_opt=float(p), n_steps=n_steps, seed=int(sd), **kwargs)
            df["seed"] = int(sd)
            df["scenario_p_heston"] = float(p)
            all_df.append(df)
    return pd.concat(all_df, ignore_index=True)
