"""Clustering des actions et allocation hiérarchique.

Idée : deux actions très corrélées apportent peu de diversification. On regroupe les
actions qui « bougent ensemble », puis on répartit le risque entre groupes plutôt
qu'entre actions individuelles.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def corr_distance(returns):
    """Distance de corrélation : d_ij = sqrt((1 - rho_ij) / 2), entre 0 et 1."""
    corr = returns.corr()
    return np.sqrt(((1 - corr) / 2).clip(lower=0)), corr


def hierarchical_clusters(returns, n_clusters=6, method="ward"):
    """Classification ascendante hiérarchique sur la distance de corrélation."""
    dist, _ = corr_distance(returns)
    Z = linkage(squareform(dist.values, checks=False), method=method)
    labels = fcluster(Z, n_clusters, criterion="maxclust")
    return pd.Series(labels, index=returns.columns, name="cluster"), Z


def kmeans_clusters(returns, n_clusters=6, n_components=5, seed=0):
    """K-means sur les premières composantes principales des rendements standardisés."""
    X = ((returns - returns.mean()) / returns.std()).T.values  # une ligne par action
    emb = PCA(n_components=n_components, random_state=seed).fit_transform(X)
    labels = KMeans(n_clusters, n_init=20, random_state=seed).fit_predict(emb)
    return pd.Series(labels + 1, index=returns.columns, name="cluster"), emb


def _inverse_vol_weights(cov):
    iv = 1 / np.sqrt(np.diag(cov))
    return iv / iv.sum()


def cluster_risk_parity(returns, labels, cov):
    """Allocation en deux étapes :
    1. dans chaque cluster, poids inversement proportionnels à la volatilité ;
    2. entre clusters, poids inversement proportionnels à la volatilité du sous-portefeuille.
    Résultat : chaque groupe contribue à peu près autant au risque total.
    """
    cov = pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
    w = pd.Series(0.0, index=returns.columns)
    sub_vol, sub_w = {}, {}
    for c in labels.unique():
        names = labels.index[labels == c]
        c_cov = cov.loc[names, names].values
        wi = _inverse_vol_weights(c_cov)
        sub_w[c] = pd.Series(wi, index=names)
        sub_vol[c] = np.sqrt(wi @ c_cov @ wi)
    inv = {c: 1 / v for c, v in sub_vol.items()}
    total = sum(inv.values())
    for c, wi in sub_w.items():
        w[wi.index] = wi * inv[c] / total
    return w.values


def hrp(returns, cov):
    """Hierarchical Risk Parity (López de Prado, 2016) : bissection récursive de
    l'arbre de clustering, avec une allocation inverse-variance à chaque niveau."""
    cov = pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
    dist, _ = corr_distance(returns)
    Z = linkage(squareform(dist.values, checks=False), method="single")
    order = list(returns.columns[leaves_list(Z)])
    w = pd.Series(1.0, index=order)
    stack = [order]
    while stack:
        items = stack.pop()
        if len(items) < 2:
            continue
        left, right = items[: len(items) // 2], items[len(items) // 2:]

        def var(ix):
            c = cov.loc[ix, ix].values
            iv = 1 / np.diag(c)
            iv /= iv.sum()
            return iv @ c @ iv

        a = 1 - var(left) / (var(left) + var(right))
        w[left] *= a
        w[right] *= 1 - a
        stack += [left, right]
    return w.reindex(returns.columns).values
