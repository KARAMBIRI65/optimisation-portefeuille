"""Optimisation moyenne-variance de Markowitz (1952).

Notations : w = poids, mu = rendements espérés annualisés, Sigma = covariance annualisée.
  rendement du portefeuille  = w' mu
  volatilité du portefeuille = sqrt(w' Sigma w)
Contraintes : somme des poids = 1, pas de vente à découvert (0 <= w_i <= w_max).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252


def estimate(returns, shrink=True):
    """Estime mu et Sigma annualisés.

    shrink=True : covariance de Ledoit-Wolf, plus stable que la covariance empirique
    quand on a beaucoup d'actifs pour peu d'observations.
    """
    mu = returns.mean() * TRADING_DAYS
    if shrink:
        cov = LedoitWolf().fit(returns.values).covariance_
        sigma = pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
    else:
        sigma = returns.cov()
    return mu, sigma * TRADING_DAYS


def perf(w, mu, sigma, rf=0.0):
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ sigma @ w))
    return ret, vol, (ret - rf) / vol


def _solve(objective, n, w_max=1.0, extra_constraints=()):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}, *extra_constraints]
    res = minimize(objective, np.full(n, 1 / n), method="SLSQP",
                   bounds=[(0, w_max)] * n, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-10})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def min_variance(sigma, w_max=1.0):
    """Portefeuille de variance minimale : min w' Sigma w."""
    S = np.asarray(sigma)
    return _solve(lambda w: w @ S @ w, len(S), w_max)


def max_sharpe(mu, sigma, rf=0.0, w_max=1.0):
    """Portefeuille tangent : maximise le ratio de Sharpe (mu_p - rf) / sigma_p."""
    m, S = np.asarray(mu), np.asarray(sigma)
    return _solve(lambda w: -(w @ m - rf) / np.sqrt(w @ S @ w), len(m), w_max)


def target_return(mu, sigma, target, w_max=1.0):
    """Variance minimale sous contrainte de rendement cible (un point de la frontière)."""
    m, S = np.asarray(mu), np.asarray(sigma)
    con = {"type": "eq", "fun": lambda w: w @ m - target}
    return _solve(lambda w: w @ S @ w, len(m), w_max, (con,))


def efficient_frontier(mu, sigma, n_points=40, w_max=1.0):
    """Renvoie un DataFrame (volatilité, rendement) des portefeuilles efficients."""
    S = np.asarray(sigma)
    w_mv = min_variance(S, w_max)
    lo = float(w_mv @ np.asarray(mu))
    # Rendement maximal atteignable avec la contrainte w_max
    hi = float(np.sort(np.asarray(mu))[::-1][: int(np.ceil(1 / w_max))].mean())
    rows = []
    for t in np.linspace(lo, hi * 0.999, n_points):
        w = target_return(mu, S, t, w_max)
        r, v, _ = perf(w, np.asarray(mu), S)
        rows.append((v, r))
    return pd.DataFrame(rows, columns=["volatilite", "rendement"])


def random_portfolios(mu, sigma, n=5000, seed=0):
    """Portefeuilles aléatoires (loi de Dirichlet) pour visualiser le nuage risque/rendement."""
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(len(mu)), n)
    m, S = np.asarray(mu), np.asarray(sigma)
    ret = W @ m
    vol = np.sqrt(np.einsum("ij,jk,ik->i", W, S, W))
    return pd.DataFrame({"volatilite": vol, "rendement": ret, "sharpe": ret / vol})
