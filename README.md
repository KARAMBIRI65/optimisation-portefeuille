# Optimisation de portefeuille sur le CAC 40 : Markowitz, clustering et backtest

Projet personnel de data science appliquée à la finance de marché.
**Objectif** : construire un portefeuille d'actions françaises qui offre un meilleur rapport rendement/risque que l'indice CAC 40, et vérifier honnêtement, par un backtest sans regard vers le futur, si les méthodes d'optimisation tiennent leurs promesses.

## Ce que fait le projet

| Étape | Méthodes | Fichier |
|---|---|---|
| Données | 32 actions du CAC 40, 2015 à 2025, prix ajustés (Yahoo Finance, gratuit) | `src/data.py` |
| Markowitz | Frontière efficiente, variance minimale, Sharpe maximal, covariance de Ledoit-Wolf, poids plafonnés à 15 % | `src/markowitz.py` |
| Clustering | Distance de corrélation, classification hiérarchique (Ward), K-means sur ACP, parité de risque par clusters, HRP (López de Prado) | `src/clustering.py` |
| Backtest | Walk-forward, rebalancement mensuel, fenêtre glissante d'un an, frais de 0,10 %, Sharpe, Sortino, drawdown, VaR, analyse de sensibilité | `src/backtest.py` |

Toute l'étude, avec graphiques et explications, est dans **`notebooks/optimisation_portefeuille.ipynb`**.

## Stratégies comparées

1. **Équipondéré (1/N)** : la référence naïve, souvent difficile à battre.
2. **Variance minimale** : on ne cherche qu'à réduire le risque.
3. **Max Sharpe** : le portefeuille « tangent » de Markowitz.
4. **Parité de risque par clusters** : chaque groupe d'actions corrélées pèse autant dans le risque.
5. **HRP** : allocation hiérarchique, sans inversion de la matrice de covariance.
6. **CAC 40** : l'indice, comme point de comparaison.

## Résultats

> À remplir après avoir lancé le notebook avec les vraies données : copie ici le tableau des métriques et le graphique `figures/backtest.png`.

| Stratégie | Rendement annualisé | Volatilité | Sharpe | Perte max |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

**Ce qu'on observe en général** (et qu'il est intéressant de vérifier) : le Max Sharpe, brillant sur le papier, déçoit souvent hors échantillon car les rendements espérés sont très mal estimés ; les méthodes fondées uniquement sur le risque (variance minimale, HRP) sont plus stables et limitent les pertes en période de crise (2020, 2022).

## Lancer le projet

```bash
git clone https://github.com/<ton-pseudo>/optimisation-portefeuille.git
cd optimisation-portefeuille
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt

python run_backtest.py          # étude complète en une commande -> figures/
jupyter notebook notebooks/     # ou le notebook détaillé
```

Les prix sont téléchargés une fois puis gardés dans `data/prix.csv`. Sans connexion internet, `python run_backtest.py --demo` (ou `DEMO = True` dans le notebook) utilise des données simulées pour tester le code. Ces résultats simulés n'ont aucune valeur financière.

## Structure

```
optimisation-portefeuille/
├── notebooks/optimisation_portefeuille.ipynb   # l'étude complète
├── src/
│   ├── data.py          # téléchargement et rendements
│   ├── markowitz.py     # optimisation moyenne-variance
│   ├── clustering.py    # CAH, K-means, parité par clusters, HRP
│   └── backtest.py      # moteur walk-forward et métriques
├── run_backtest.py      # script en une commande
├── data/                # cache des prix (non versionné)
└── figures/             # graphiques générés
```

## Limites

- **Biais du survivant** : on utilise la composition actuelle du CAC 40, donc des entreprises qui ont « survécu » jusqu'à aujourd'hui.
- Frais de transaction simplifiés, pas de fiscalité ni de contraintes de liquidité.
- Rendements espérés estimés par la moyenne historique, l'estimateur le plus fragile.

## Pistes d'amélioration

Modèle de Black-Litterman, covariance dynamique (GARCH / DCC), contraintes sectorielles, optimisation de la CVaR, extension à d'autres classes d'actifs (obligations, or).

## Références

- H. Markowitz (1952), *Portfolio Selection*, Journal of Finance.
- O. Ledoit et M. Wolf (2004), *Honey, I Shrunk the Sample Covariance Matrix*.
- M. López de Prado (2016), *Building Diversified Portfolios that Outperform Out-of-Sample*.
- R. Michaud (1989), *The Markowitz Optimization Enigma: Is 'Optimized' Optimal?*

---
Auteur : Issa Aziz KARAMBIRI, élève ingénieur en mathématiques appliquées (data, IA, finance).
