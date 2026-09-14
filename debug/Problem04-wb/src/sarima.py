"""自实现的 SARIMA(p,d,q)(P,D,Q)_s（条件最小二乘 CSS 估计，不依赖 statsmodels）。

模型
----
    phi(B) Phi(B^s) w_t = theta(B) Theta(B^s) epsilon_t
    w_t = (1-B)^d (1-B^s)^D y_t

条件平方和（CSS）：
    epsilon_t = w_t - sum(a_lag * w_{t-lag}) - sum(b_lag * epsilon_{t-lag})
    SSE = sum_{t>=start} epsilon_t^2
其中 {a_lag} 是 phi(B)Phi(B^s) 的系数、{b_lag} 是 theta(B)Theta(B^s) 的系数
（两个多项式相乘，含 phi_i*Phi_j 的交叉项）。

用 Nelder-Mead 最小化 SSE，并对不平稳 / 不可逆的参数加惩罚。
只支持 d, D in {0, 1}（本项目用不到更高阶差分）。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

PENALTY = 1e8
ROOT_EPS = 1e-3


def _ar_lag_coeffs(ar: np.ndarray, sar: np.ndarray, s: int) -> dict:
    """phi(B)Phi(B^s) = 1 - sum(a_lag B^lag)，返回 {lag: a_lag}（lag>=1）。"""
    coeff = {0: 1.0}
    for i, value in enumerate(ar, start=1):
        coeff[i] = coeff.get(i, 0.0) - float(value)
    for j, value in enumerate(sar, start=1):
        lag = j * s
        coeff[lag] = coeff.get(lag, 0.0) - float(value)
    for i, vi in enumerate(ar, start=1):
        for j, vj in enumerate(sar, start=1):
            lag = i + j * s
            coeff[lag] = coeff.get(lag, 0.0) + float(vi) * float(vj)
    return {lag: -value for lag, value in coeff.items() if lag > 0}


def _ma_lag_coeffs(ma: np.ndarray, sma: np.ndarray, s: int) -> dict:
    """theta(B)Theta(B^s) = 1 + sum(b_lag B^lag)，返回 {lag: b_lag}（lag>=1）。"""
    coeff: dict = {}
    for k, value in enumerate(ma, start=1):
        coeff[k] = coeff.get(k, 0.0) + float(value)
    for l, value in enumerate(sma, start=1):
        lag = l * s
        coeff[lag] = coeff.get(lag, 0.0) + float(value)
    for k, vk in enumerate(ma, start=1):
        for l, vl in enumerate(sma, start=1):
            lag = k + l * s
            coeff[lag] = coeff.get(lag, 0.0) + float(vk) * float(vl)
    return coeff


def _roots_outside_unit(coeffs: np.ndarray, seasonal: np.ndarray, s: int) -> bool:
    """检验 1 + sum(coeffs B^i) 与 1 + sum(seasonal B^(j*s)) 的根是否全在单位圆外。"""
    for values, step in ((coeffs, 1), (seasonal, s)):
        if len(values) == 0:
            continue
        ascending = np.zeros(len(values) * step + 1)
        ascending[0] = 1.0
        for index, value in enumerate(values, start=1):
            ascending[index * step] = float(value)
        if not np.any(np.abs(ascending[1:]) > 0):
            continue
        try:
            roots = np.roots(ascending[::-1])
        except Exception:
            return False
        if roots.size and float(np.min(np.abs(roots))) <= 1.0 + ROOT_EPS:
            return False
    return True


def _residuals(w: np.ndarray, ar_lags: dict, ma_lags: dict, start: int) -> np.ndarray:
    n = len(w)
    eps = np.zeros(n)
    for t in range(start, n):
        value = 0.0
        for lag, coeff in ar_lags.items():
            value += coeff * w[t - lag]
        for lag, coeff in ma_lags.items():
            value += coeff * eps[t - lag]
        eps[t] = w[t] - value
    return eps


def difference_series(y: np.ndarray, d: int, D: int, s: int):
    """返回 (w, u)：u = (1-B)^d y，w = (1-B^s)^D u。"""
    u = np.asarray(y, dtype=float).copy()
    for _ in range(d):
        u = np.diff(u)
    w = u.copy()
    for _ in range(D):
        w = w[s:] - w[:-s]
    return w, u


def fit_sarima(y: np.ndarray, order, maxiter: int = 200, x0=None):
    """拟合 SARIMA，返回参数字典；失败抛 ValueError。"""
    p, d, q, P, D, Q, s = order
    if d not in (0, 1) or D not in (0, 1):
        raise ValueError("只支持 d, D in {0,1}")
    y = np.asarray(y, dtype=float)
    if y.size < 3 * s + p + P + q + Q + d + D * s + 5:
        raise ValueError("样本过短，无法拟合 SARIMA")

    centre = float(np.mean(y))
    spread = float(np.std(y))
    if spread < 1e-9:
        spread = 1.0
    ys = (y - centre) / spread
    w, u = difference_series(ys, d, D, s)

    zero_ar = _ar_lag_coeffs(np.zeros(p), np.zeros(P), s)
    zero_ma = _ma_lag_coeffs(np.zeros(q), np.zeros(Q), s)
    max_lag = max([0] + list(zero_ar) + list(zero_ma))
    start = max_lag

    def unpack(params):
        index = 0
        ar = params[index:index + p]; index += p
        sar = params[index:index + P]; index += P
        ma = params[index:index + q]; index += q
        sma = params[index:index + Q]; index += Q
        return ar, sar, ma, sma

    def objective(params):
        ar, sar, ma, sma = unpack(params)
        stationary = _roots_outside_unit(-ar, -sar, s)
        invertible = _roots_outside_unit(ma, sma, s)
        eps = _residuals(w, _ar_lag_coeffs(ar, sar, s), _ma_lag_coeffs(ma, sma, s), start)
        sse = float(np.sum(eps[start:] ** 2))
        if not (stationary and invertible):
            sse += PENALTY * (1.0 + abs(sse))
        return sse

    if x0 is None:
        x0 = np.full(p + P + q + Q, 0.2)
    best = minimize(
        objective, np.asarray(x0, dtype=float), method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-6},
    )
    ar, sar, ma, sma = unpack(best.x)
    if not (_roots_outside_unit(-ar, -sar, s) and _roots_outside_unit(ma, sma, s)):
        raise ValueError("SARIMA 拟合未得到平稳可逆解")
    return {
        "order": order, "ar": ar, "sar": sar, "ma": ma, "sma": sma,
        "centre": centre, "spread": spread, "w": w, "u": u,
        "sse": float(np.sum(_residuals(
            w, _ar_lag_coeffs(ar, sar, s), _ma_lag_coeffs(ma, sma, s), start
        )[start:] ** 2)),
        "ys_last": float(ys[-1]), "n": int(y.size),
    }


def forecast_one(y: np.ndarray, order, **kwargs) -> float:
    """用历史 y 拟合 SARIMA，外推 1 步并还原到原尺度。"""
    p, d, q, P, D, Q, s = order
    model = fit_sarima(y, order, **kwargs)
    ar_lags = _ar_lag_coeffs(model["ar"], model["sar"], s)
    ma_lags = _ma_lag_coeffs(model["ma"], model["sma"], s)
    start = max([0] + list(ar_lags) + list(ma_lags))
    w, u = model["w"], model["u"]
    eps = _residuals(w, ar_lags, ma_lags, start)

    n = len(w)
    value = 0.0
    for lag, coeff in ar_lags.items():
        value += coeff * w[n - lag]
    for lag, coeff in ma_lags.items():
        value += coeff * eps[n - lag]

    if D == 1:
        value = value + u[len(u) - s]
    if d == 1:
        value = value + model["ys_last"]
    return float(value * model["spread"] + model["centre"])


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n = 220
    s = 7
    season = np.array([0.9, 0.35, -0.1, -0.25, -0.35, -0.6, 0.05])
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.55 * noise[t - 1] + rng.normal(0, 0.12)
    y = 10.0 + np.array([season[t % s] for t in range(n)]) + noise
    truth = 10.0 + season[n % s] + 0.55 * noise[n - 1]
    print("truth(1 步) = %.4f" % truth)
    for order in [(1, 0, 1, 1, 0, 1, 7), (1, 0, 0, 1, 0, 0, 7),
                  (0, 0, 0, 1, 0, 1, 7), (1, 1, 1, 1, 0, 1, 7)]:
        try:
            fc = forecast_one(y, order)
            print("order %-22s forecast=%.4f  err=%+.4f" % (order, fc, fc - truth))
        except Exception as exc:
            print("order %-22s failed: %s" % (order, exc))