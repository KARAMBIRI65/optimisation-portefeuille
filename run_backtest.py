"""Lance l'étude complète en ligne de commande.

    python run_backtest.py          # vraies données Yahoo Finance
    python run_backtest.py --demo   # données simulées (test hors ligne)
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import backtest, data

demo = "--demo" in sys.argv
prices, bench = data.load_prices(demo=demo)
print(f"{prices.shape[1]} actions, du {prices.index[0]:%d/%m/%Y} au {prices.index[-1]:%d/%m/%Y}"
      + ("  [DONNÉES SIMULÉES]" if demo else ""))

rets, weights, table = backtest.run_all(data.simple_returns(prices), bench)
print(table.round(3).to_string())

nav = (1 + rets).cumprod()
ax = nav.plot(figsize=(11, 5), logy=True, title="Valeur de 1 € investi (échelle log)")
ax.set_ylabel("Valeur du portefeuille")
plt.tight_layout()
plt.savefig("figures/backtest.png", dpi=150)
table.to_csv("figures/metriques.csv")
