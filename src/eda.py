"""
Exploratory spatio-temporal analysis of the GTD.

Exports:
    1. incidents & deaths per year          -> the global trend (+ 1993 gap)
    2. incidents by region across decades    -> shifting geography
    3. attack-type / target-type typology    -> how attacks are carried out
    4. top countries and groups              -> concentration
    5. interactive incident density map (Plotly)

Run:
    python src/eda.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from preprocessing import load_clean

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"


def run(save: bool = True) -> pd.DataFrame:
    df = load_clean()
    sns.set_theme(style="whitegrid")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. incidents & deaths per year ---------------------------------
    per_year = df.groupby("year").agg(
        incidents=("event_id", "count"), deaths=("n_kill", "sum"))
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(per_year.index, per_year.incidents, color="#4c72b0",
            alpha=0.8, label="incidents")
    ax1.set_ylabel("Incidents", color="#4c72b0")
    ax2 = ax1.twinx()
    ax2.plot(per_year.index, per_year.deaths, color="#c44e52", lw=2,
             label="morts")
    ax2.set_ylabel("Morts", color="#c44e52")
    ax1.set_xlabel("Année")
    ax1.set_title("Incidents terroristes et morts par an (GTD, 1970-2017)\n"
                  "note : 1993 absent de la base source")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "incidents_deaths_per_year.png", dpi=130)
    plt.close()

    # --- 2. region x decade heatmap -------------------------------------
    rd = df.pivot_table(index="region", columns="decade", values="event_id",
                        aggfunc="count", fill_value=0)
    rd = rd.loc[rd.sum(axis=1).sort_values(ascending=False).index]
    plt.figure(figsize=(11, 6))
    sns.heatmap(rd, cmap="rocket_r", annot=True, fmt="d",
                cbar_kws={"label": "incidents"})
    plt.title("Évolution géographique des incidents par décennie")
    plt.xlabel("Décennie"); plt.ylabel("Région")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "region_decade_heatmap.png", dpi=130)
    plt.close()

    # --- 3. attack & target typology ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    df.attack_type.value_counts().head(8).iloc[::-1].plot(
        kind="barh", ax=axes[0], color="#55a868")
    axes[0].set_title("Types d'attaque")
    df.target_type.value_counts().head(8).iloc[::-1].plot(
        kind="barh", ax=axes[1], color="#8172b3")
    axes[1].set_title("Types de cible")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "attack_target_typology.png", dpi=130)
    plt.close()

    # --- 4. top countries -----------------------------------------------
    top_c = df.country.value_counts().head(12)
    plt.figure(figsize=(9, 5.5))
    top_c.iloc[::-1].plot(kind="barh", color="#c44e52")
    plt.xlabel("Incidents"); plt.title("Pays les plus touchés (1970-2017)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "top_countries.png", dpi=130)
    plt.close()

    # --- 5. interactive density map -------------------------------------
    geo = df[df.geo_known].sample(min(20000, df.geo_known.sum()),
                                  random_state=42)
    fig = px.density_map(
        geo, lat="lat", lon="lon", radius=4, zoom=0,
        map_style="carto-positron", center={"lat": 20, "lon": 20},
        title="Densité géographique des incidents (échantillon 20k)")
    fig.write_html(FIG_DIR / "incident_density_map.html",
                   include_plotlyjs="cdn")

    if save:
        per_year.to_csv(REPORT_DIR / "yearly_summary.csv")
        print("[eda] saved 4 figures + density map + yearly_summary.csv")
        print(f"[eda] peak year: {per_year.incidents.idxmax()} "
              f"({per_year.incidents.max():,} incidents)")
    return per_year


if __name__ == "__main__":
    run()
