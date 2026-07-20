"""
Unsupervised clustering of countries by attack profile (brief step 4).

Each country is described by a normalised behavioural profile — the mix of
attack types, target types and suicide/lethality rates it exhibits — and grouped
with K-Means. This surfaces "families" of countries that experience structurally
similar terrorism, independent of raw volume.

    * features : per-country share of each attack type + target type,
                 suicide rate, lethality rate, mean severity (standardised);
    * K chosen by silhouette over a small grid;
    * PCA(2) only for a readable scatter of the clusters.

Run:
    python src/clustering.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from data import RANDOM_STATE
from preprocessing import load_clean

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"

MIN_INCIDENTS = 50  # ignore countries with too little data to profile


def build_profiles() -> pd.DataFrame:
    """Per-country behavioural feature matrix."""
    df = load_clean()
    counts = df.country.value_counts()
    keep = counts[counts >= MIN_INCIDENTS].index
    df = df[df.country.isin(keep)]

    attack = pd.crosstab(df.country, df.attack_type, normalize="index")
    target = pd.crosstab(df.country, df.target_type, normalize="index")
    attack.columns = [f"atk_{c}" for c in attack.columns]
    target.columns = [f"tgt_{c}" for c in target.columns]

    rates = df.groupby("country").agg(
        suicide_rate=("suicide", "mean"),
        lethality_rate=("lethal", "mean"),
        mean_severity=("severity", "mean"),
        incidents=("event_id", "count"),
    )
    profiles = attack.join(target).join(rates)
    return profiles


def choose_k(X: np.ndarray, ks=range(3, 8)) -> int:
    best_k, best_s = 3, -1.0
    for k in ks:
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE,
                        n_init=10).fit_predict(X)
        s = silhouette_score(X, labels)
        if s > best_s:
            best_k, best_s = k, s
    return best_k


def run(save: bool = True) -> pd.DataFrame:
    profiles = build_profiles()
    feature_cols = [c for c in profiles.columns if c != "incidents"]
    X = StandardScaler().fit_transform(profiles[feature_cols])

    k = choose_k(X)
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    profiles["cluster"] = km.fit_predict(X)
    print(f"[clustering] K={k} chosen by silhouette; "
          f"{len(profiles)} countries profiled")

    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    profiles["pc1"], profiles["pc2"] = coords[:, 0], coords[:, 1]

    # Describe each cluster by its most distinctive features.
    summary_rows = []
    for c in sorted(profiles.cluster.unique()):
        sub = profiles[profiles.cluster == c]
        prof_means = sub[feature_cols].mean()
        distinct = (prof_means - profiles[feature_cols].mean()).sort_values(
            ascending=False).head(3)
        summary_rows.append({
            "cluster": c,
            "n_countries": len(sub),
            "example_countries": ", ".join(sub.sort_values(
                "incidents", ascending=False).index[:4]),
            "distinctive_traits": "; ".join(
                f"{k_}(+{v:.2f})" for k_, v in distinct.items()),
            "lethality_rate": sub.lethality_rate.mean(),
            "suicide_rate": sub.suicide_rate.mean(),
        })
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))

    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        profiles.reset_index().to_csv(
            REPORT_DIR / "country_clusters.csv", index=False)
        summary.to_csv(REPORT_DIR / "cluster_summary.csv", index=False)

        plt.figure(figsize=(10, 7))
        for c in sorted(profiles.cluster.unique()):
            sub = profiles[profiles.cluster == c]
            plt.scatter(sub.pc1, sub.pc2, s=40, label=f"cluster {c}")
        # Annotate the biggest countries.
        for name, r in profiles.sort_values(
                "incidents", ascending=False).head(15).iterrows():
            plt.annotate(name, (r.pc1, r.pc2), fontsize=7, alpha=0.8)
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.title(f"Clusters de pays par profil d'attaque (K={k}, PCA 2D)")
        plt.legend(fontsize=8); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "country_clusters.png", dpi=130)
        plt.close()
        print(f"[clustering] saved scatter + 2 tables to {REPORT_DIR}")

    return profiles


if __name__ == "__main__":
    run()
