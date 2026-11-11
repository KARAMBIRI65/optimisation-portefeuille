"""Téléchargement et préparation des données de marché (Yahoo Finance, gratuit)."""
from pathlib import Path

import numpy as np
import pandas as pd

# Univers : grandes capitalisations du CAC 40 (tickers Yahoo Finance, place de Paris)
TICKERS = {
    "AI.PA": "Air Liquide", "AIR.PA": "Airbus", "ALO.PA": "Alstom", "BN.PA": "Danone",
    "BNP.PA": "BNP Paribas", "CA.PA": "Carrefour", "CAP.PA": "Capgemini", "CS.PA": "AXA",
    "DG.PA": "Vinci", "DSY.PA": "Dassault Systèmes", "EL.PA": "EssilorLuxottica",
    "EN.PA": "Bouygues", "ENGI.PA": "Engie", "GLE.PA": "Société Générale", "HO.PA": "Thales",
    "KER.PA": "Kering", "LR.PA": "Legrand", "MC.PA": "LVMH", "ML.PA": "Michelin",
    "OR.PA": "L'Oréal", "ORA.PA": "Orange", "PUB.PA": "Publicis", "RI.PA": "Pernod Ricard",
    "RMS.PA": "Hermès", "RNO.PA": "Renault", "SAF.PA": "Safran", "SAN.PA": "Sanofi",
    "SGO.PA": "Saint-Gobain", "SU.PA": "Schneider Electric", "TTE.PA": "TotalEnergies",
    "VIE.PA": "Veolia", "ACA.PA": "Crédit Agricole",
}
BENCHMARK = "^FCHI"  # indice CAC 40

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_prices(start="2015-01-01", end="2025-10-31", refresh=False, demo=False):
    """Renvoie (prix des actions, prix du CAC 40), avec cache CSV local.

    Les prix sont ajustés des dividendes et des divisions d'actions.
    demo=True : données SIMULÉES (pour tester le code sans connexion internet).
    """
    if demo:
        return synthetic_prices(start, end)
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / "prix.csv"
    if cache.exists() and not refresh:
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        import yfinance as yf

        raw = yf.download(list(TICKERS) + [BENCHMARK], start=start, end=end,
                          auto_adjust=True, progress=False)
        prices = raw["Close"]
        prices.to_csv(cache)

    bench = prices.pop(BENCHMARK) if BENCHMARK in prices else None
    # On garde les actions avec au moins 95 % d'historique, puis on comble les trous
    prices = prices.loc[:, prices.notna().mean() > 0.95].ffill().dropna()
    if bench is not None:
        bench = bench.reindex(prices.index).ffill()
    return prices, bench


def log_returns(prices):
    """Rendements logarithmiques quotidiens."""
    return np.log(prices / prices.shift(1)).dropna()


def simple_returns(prices):
    """Rendements simples quotidiens (utilisés pour le backtest)."""
    return prices.pct_change().dropna()


def synthetic_prices(start="2015-01-01", end="2025-10-31", seed=42):
    """Prix SIMULÉS par un modèle à facteurs (marché + secteurs + bruit propre).

    Ne sert qu'à tester le code hors ligne : les résultats n'ont aucune valeur réelle.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n, names = len(dates), list(TICKERS)
    sectors = rng.integers(0, 6, len(names))
    vol_regime = np.where(rng.random(n).cumsum() % 400 < 60, 2.0, 1.0)  # périodes de crise
    market = rng.standard_t(5, n) * 0.009 * vol_regime
    sect = rng.standard_normal((n, 6)) * 0.006
    beta = rng.uniform(0.6, 1.4, len(names))
    idio = rng.standard_normal((n, len(names))) * rng.uniform(0.008, 0.016, len(names))
    drift = rng.uniform(0.0, 0.0006, len(names))
    r = drift + market[:, None] * beta + sect[:, sectors] + idio
    prices = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=dates, columns=names)
    bench = pd.Series(5000 * np.exp(np.cumsum(market + 0.0002)), index=dates, name=BENCHMARK)
    return prices, bench
