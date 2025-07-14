import numpy as np
from ns_core import ns_yield

# Price a bond given NS parameters
def price_bond(beta0, beta1, beta2, lambd, maturity=10, coupon=0.03, freq=1):
    times = np.arange(1, maturity * freq + 1) / freq
    yts   = ns_yield(times, beta0, beta1, beta2, lambd)
    dfs   = np.exp(-yts * times)
    cfs   = np.full_like(times, coupon / freq)
    cfs[-1] += 1.0
    price = np.sum(cfs * dfs)
    return price

# Modified duration
def modified_duration(beta0, beta1, beta2, lambd, maturity=10, coupon=0.03, freq=1):
    times = np.arange(1, maturity * freq + 1) / freq
    yts   = ns_yield(times, beta0, beta1, beta2, lambd)
    dfs   = np.exp(-yts * times)
    cfs   = np.full_like(times, coupon / freq)
    cfs[-1] += 1.0
    weighted_times = times * cfs * dfs
    price = np.sum(cfs * dfs)
    return np.sum(weighted_times) / price

# DV01
def dv01(beta0, beta1, beta2, lambd, maturity=10, coupon=0.03, freq=1):
    price_0 = price_bond(beta0, beta1, beta2, lambd, maturity, coupon, freq)
    price_up = price_bond(beta0 + 0.0001, beta1, beta2, lambd, maturity, coupon, freq)
    return price_up - price_0

# Convexity
def convexity(beta0, beta1, beta2, lambd, maturity=10, coupon=0.03, freq=1):
    times = np.arange(1, maturity * freq + 1) / freq
    yts   = ns_yield(times, beta0, beta1, beta2, lambd)
    dfs   = np.exp(-yts * times)
    cfs   = np.full_like(times, coupon / freq)
    cfs[-1] += 1.0
    weight = times**2 * cfs * dfs
    price = np.sum(cfs * dfs)
    return np.sum(weight) / price
