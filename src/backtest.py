"""Backtest « walk-forward » : à chaque rebalancement, on n'utilise QUE le passé.

Chaque mois, on estime les paramètres sur la fenêtre des `lookback` derniers jours,
on calcule les nouveaux poids, on paie des frais de transaction, puis on laisse le
portefeuille évoluer jusqu'au rebalancement suivant (les poids dérivent avec les prix).
"""
import numpy as np
import pandas as pd

from . import clustering, markowitz

TRADING_DAYS = 252


def strategies(n_clusters=6, w_max=0.15):
    """Dictionnaire nom -> fonction(fenêtre de rendements) -> poids."""

    def equal_weight(R):
        return np.full(R.shape[1], 1 / R.shape[1])

    def min_var(R):
        _, S = markowitz.estimate(R)
        return markowitz.min_variance(S, w_max)

    def max_sharpe(R):
        mu, S = markowitz.estimate(R)
        return markowitz.max_sharpe(mu, S, w_max=w_max)

    def cluster_rp(R):
        _, S = markowitz.estimate(R)
        labels, _ = clustering.hierarchical_clusters(R, n_clusters)
        return clustering.cluster_risk_parity(R, labels, S.values)

    def hrp(R):
        _, S = markowitz.estimate(R)
        return clustering.hrp(R, S.values)

    return {
        "Équipondéré (1/N)": equal_weight,
        "Variance minimale": min_var,
        "Max Sharpe": max_sharpe,
        "Parité de risque par clusters": cluster_rp,
        "HRP": hrp,
    }


def run(returns, weight_fn, lookback=252, rebalance="ME", cost_bps=10):
    """Renvoie (série des rendements quotidiens nets, DataFrame des poids à chaque rebalancement)."""
    dates = returns.index
    rebal_dates = set(returns.groupby(dates.to_period(rebalance[0])).tail(1).index)
    w = None
    daily, hist = [], {}
    for i in range(lookback, len(dates)):
        d = dates[i]
        r = returns.iloc[i].values
        if w is not None:
            port_r = float(w @ r)
            w = w * (1 + r) / (1 + port_r)  # dérive des poids
            daily.append((d, port_r))
        if d in rebal_dates:
            new_w = weight_fn(returns.iloc[i - lookback + 1: i + 1])
            turnover = np.abs(new_w - (w if w is not None else 0)).sum()
            if daily:
                daily[-1] = (d, daily[-1][1] - turnover * cost_bps / 1e4)
            w = new_w
            hist[d] = new_w
    s = pd.Series(dict(daily), name="rendement")
    return s, pd.DataFrame(hist, index=returns.columns).T


def metrics(r, rf=0.0):
    """Indicateurs de performance classiques à partir de rendements quotidiens."""
    nav = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    cagr = nav.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    downside = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    dd = nav / nav.cummax() - 1
    return pd.Series({
        "Rendement annualisé": cagr,
        "Volatilité": vol,
        "Sharpe": (cagr - rf) / vol,
        "Sortino": (cagr - rf) / downside,
        "Perte max (drawdown)": dd.min(),
        "Calmar": cagr / abs(dd.min()),
        "VaR 95 % (jour)": r.quantile(0.05),
    })


def run_all(returns, benchmark=None, **kw):
    """Lance toutes les stratégies. Renvoie (rendements, poids, tableau des métriques)."""
    res, weights = {}, {}
    for name, fn in strategies().items():
        res[name], weights[name] = run(returns, fn, **kw)
    rets = pd.DataFrame(res)
    if benchmark is not None:
        rets["CAC 40 (indice)"] = benchmark.pct_change().reindex(rets.index)
    rets = rets.dropna()
    table = rets.apply(metrics).T
    return rets, weights, table
