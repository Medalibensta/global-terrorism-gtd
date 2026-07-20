"""
Terrorism-pressure index per country / region (brief step 3).

A country's "pressure" is more than a raw incident count: a few very lethal
attacks and a steady drip of small ones are different phenomena. The index
combines four min-max-normalised, interpretable components per country:

    frequency  — log incidents (heavy tail across countries);
    lethality  — total deaths (log);
    intensity  — mean severity per incident (deaths + 0.5·wounded);
    recency    — share of incidents in the last decade of the data (2008-2017).

    pressure = 0.4·frequency + 0.3·lethality + 0.15·intensity + 0.15·recency

This is a descriptive, relative index over 1970-2017 — explicitly not a forecast
or a normative ranking (see README ethics section).

Run:
    python src/pressure_index.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

from preprocessing import load_clean

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"

WEIGHTS = {"frequency": 0.40, "lethality": 0.30,
           "intensity": 0.15, "recency": 0.15}
RECENT_FROM = 2008


def _minmax(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else s * 0.0


def build_index(by: str = "country") -> pd.DataFrame:
    df = load_clean()
    grp = df.groupby(by)
    comp = grp.agg(
        incidents=("event_id", "count"),
        deaths=("n_kill", "sum"),
        mean_severity=("severity", "mean"),
    )
    recent = (df[df.year >= RECENT_FROM].groupby(by).event_id.count()
              .rename("recent"))
    comp = comp.join(recent).fillna({"recent": 0})
    comp["recency"] = comp.recent / comp.incidents

    comp["frequency_n"] = _minmax(np.log1p(comp.incidents))
    comp["lethality_n"] = _minmax(np.log1p(comp.deaths))
    comp["intensity_n"] = _minmax(comp.mean_severity)
    comp["recency_n"] = _minmax(comp.recency)
    comp["pressure_index"] = (
        WEIGHTS["frequency"] * comp.frequency_n
        + WEIGHTS["lethality"] * comp.lethality_n
        + WEIGHTS["intensity"] * comp.intensity_n
        + WEIGHTS["recency"] * comp.recency_n
    )
    return comp.sort_values("pressure_index", ascending=False).reset_index()


def run(save: bool = True) -> pd.DataFrame:
    country = build_index("country")
    print("=== Terrorism pressure index — top 15 countries ===")
    print(country.head(15)[["country", "incidents", "deaths",
                            "mean_severity", "recency",
                            "pressure_index"]].round(3).to_string(index=False))

    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        country.to_csv(REPORT_DIR / "pressure_index_country.csv", index=False)

        top = country.head(15).sort_values("pressure_index")
        plt.figure(figsize=(9, 6.5))
        plt.barh(top.country, top.pressure_index,
                 color=plt.cm.YlOrRd(top.pressure_index))
        plt.xlabel("Indice de pression (0-1)")
        plt.title("Indice de pression terroriste — top 15 pays (1970-2017)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "pressure_index_top_countries.png", dpi=130)
        plt.close()

        # Choropleth over all countries.
        fig = px.choropleth(
            country, locations="country", locationmode="country names",
            color="pressure_index", color_continuous_scale="YlOrRd",
            hover_data={"incidents": True, "deaths": True,
                        "pressure_index": ":.3f"},
            title="Indice de pression terroriste par pays (GTD 1970-2017)")
        fig.write_html(FIG_DIR / "pressure_index_map.html",
                       include_plotlyjs="cdn")
        print(f"[pressure] saved table + chart + choropleth to {REPORT_DIR}")

    return country


if __name__ == "__main__":
    run()
