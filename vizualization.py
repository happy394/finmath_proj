# vizualization.py
from __future__ import annotations
import matplotlib.pyplot as plt


def plot_stock_price(df):
    plt.figure()
    plt.plot(df["t"].values, df["S"].values)
    plt.title("Stock price S_t")
    plt.xlabel("step")
    plt.ylabel("price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_stock_realized_vol(df):
    plt.figure()
    plt.plot(df["t"].values, df["rv20"].values)
    plt.title("Stock realized volatility (window=20) [std of log-returns]")
    plt.xlabel("step")
    plt.ylabel("rv20")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_option_price_atm(df):
    plt.figure()
    plt.plot(df["t"].values, df["C_atm"].values)
    plt.title("Option price (ATM Call) C_t")
    plt.xlabel("step")
    plt.ylabel("C_atm")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_iv_atm(df):
    plt.figure()
    plt.plot(df["t"].values, df["iv_atm"].values)
    plt.title("Implied vol ATM (annualized)")
    plt.xlabel("step")
    plt.ylabel("IV(ATM)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_smile_strength(df):
    plt.figure()
    plt.plot(df["t"].values, df["iv_high_minus_atm"].values, label="IV(K_high)-IV(ATM)")
    plt.plot(df["t"].values, df["iv_low_minus_atm"].values, label="IV(K_low)-IV(ATM)")
    plt.title("Smile/skew proxies (vs ATM)")
    plt.xlabel("step")
    plt.ylabel("ΔIV")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
