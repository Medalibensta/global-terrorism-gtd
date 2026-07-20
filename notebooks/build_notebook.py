"""
Build and execute notebooks/04_global_terrorism.ipynb.

Light aggregates run live; heavy modelling artefacts are displayed from the
CSVs/figures produced by src/, keeping the notebook fast and single-sourced.

Run:
    python notebooks/build_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "04_global_terrorism.ipynb"


def md(t): return nbf.v4.new_markdown_cell(t.strip())
def code(t): return nbf.v4.new_code_cell(t.strip())


cells = [
    md("""
# Analyse spatio-temporelle du terrorisme mondial (GTD)

> ⚠️ **Cadre et éthique** — projet à visée **académique et analytique**. Il
> décrit des dynamiques historiques (1970-2017) à partir de données ouvertes ;
> il ne prétend **ni prédire des attaques**, ni cibler des populations, ni
> fonder des décisions opérationnelles. Les biais de reporting et les limites
> éthiques sont discutés en fin de notebook et dans le README.

**Angle décisionnel** — comprendre *où*, *quand* et *comment* le terrorisme se
manifeste, construire un **indice de pression** par pays et repérer des
**familles de pays** aux profils d'attaque similaires.

**Données** — Global Terrorism Database (START / University of Maryland),
**181 691 incidents**, 1970-2017. Chargée localement depuis le CSV officiel
Kaggle (non redistribuable).

**Plan** : nettoyage → EDA spatio-temporelle → indice de pression → clustering
→ modèles prédictifs → limites & éthique.
"""),
    code("""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import load_clean

df = load_clean()
print(f"{len(df):,} incidents | {df.year.min()}-{df.year.max()}")
print(f"morts totales : {df.n_kill.sum():,.0f} | "
      f"incidents létaux : {df.lethal.mean()*100:.1f}%")
df[["year", "country", "region", "attack_type", "target_type",
    "n_kill", "n_wound"]].head()
"""),
    md("""
## 1. Qualité des données — l'incomplétude est un fait, pas un détail

La GTD est riche mais lacunaire par endroits : casualties parfois absentes,
groupe responsable inconnu une fois sur deux, et **l'année 1993 entièrement
perdue** par les archivistes de START. On quantifie ces trous avant toute
analyse.
"""),
    code("""
quality = pd.Series({
    "casualties renseignées": df.casualties_known.mean(),
    "coordonnées valides": df.geo_known.mean(),
    "groupe identifié": df.group_known.mean(),
    "part année 1993 (perdue)": (df.year == 1993).mean(),
}).mul(100).round(1)
display(quality.to_frame("% des incidents"))
"""),
    md("""
## 2. EDA spatio-temporelle

Tendance globale (avec le trou de 1993), déplacement géographique par décennie,
et typologie des attaques et des cibles.
"""),
    code("""
for fig in ["incidents_deaths_per_year.png", "region_decade_heatmap.png",
            "attack_target_typology.png", "top_countries.png"]:
    display(Image(ROOT / "reports" / "figures" / fig, width=820))
"""),
    md("""
**Lecture** — après une activité modérée jusqu'aux années 2000, les incidents
explosent à partir de ~2004 pour culminer en **2014 (~16 900 incidents)**, tirés
par le Moyen-Orient/Afrique du Nord et l'Asie du Sud. Le **bombardement/explosif**
domine largement les modes opératoires, les **civils** sont la cible la plus
fréquente. Carte de densité interactive :
`reports/figures/incident_density_map.html`.
"""),
    md("""
## 3. Indice de pression terroriste par pays

Composite normalisé : **fréquence** (0,4) + **létalité** (0,3) + **intensité**
moyenne par incident (0,15) + **récence** 2008-2017 (0,15). Descriptif et
relatif — pas une prévision.
"""),
    code("""
pressure = pd.read_csv(ROOT / "reports" / "pressure_index_country.csv")
display(pressure.head(12)[["country", "incidents", "deaths",
                           "mean_severity", "pressure_index"]].round(3))
display(Image(ROOT / "reports" / "figures" /
              "pressure_index_top_countries.png", width=720))
"""),
    md("""
Irak, Afghanistan et Pakistan dominent l'indice, cohérent avec les faits connus.
La choroplèthe mondiale est dans `reports/figures/pressure_index_map.html`.
"""),
    md("""
## 4. Familles de pays (clustering non supervisé)

K-Means sur le **profil comportemental** de chaque pays (mix de types
d'attaque/cible, taux de suicide et de létalité). K choisi par silhouette.
"""),
    code("""
clusters = pd.read_csv(ROOT / "reports" / "cluster_summary.csv")
display(clusters)
display(Image(ROOT / "reports" / "figures" / "country_clusters.png", width=760))
"""),
    md("""
**Lecture** — trois familles se dégagent : (0) pays à **forte létalité par
incident** (Sahel, RDC — assauts armés meurtriers) ; (1) profil **occidental**
à faible létalité, ciblant infrastructures et entreprises (Espagne, USA,
France) ; (2) pays à **fort volume et bombardements** (Irak, Pakistan,
Afghanistan). Le volume brut et le profil sont donc deux axes distincts.
"""),
    md("""
## 5. Modèles prédictifs (split temporel honnête)

Entraînement sur ≤ 2013, test sur ≥ 2014 — prédire le futur à partir du passé.
Deux tâches à partir des seules caractéristiques de l'incident (région, type
d'attaque/cible/arme, suicide, année).
"""),
    code("""
clf = pd.read_csv(ROOT / "reports" / "classification_metrics.csv")
reg = pd.read_csv(ROOT / "reports" / "regression_metrics.csv")
print("Classification — incident létal ?")
display(clf.round(3))
print("Régression — sévérité (log1p)")
display(reg.round(3))
display(Image(ROOT / "reports" / "figures" /
              "severity_pred_vs_actual.png", width=520))
"""),
    md("""
**Lecture** — la léthalité est prédictible modérément (Gradient Boosting :
accuracy 0,71, ROC-AUC 0,78) : le *type d'attaque* et la *région* portent
l'essentiel du signal. La **sévérité exacte** reste difficile (R² ≈ 0,24) — sans
surprise : le nombre de victimes dépend de facteurs contingents (lieu précis,
foule, réponse) absents des features. C'est un résultat honnête, pas un échec :
il borne ce que ce type de données permet de dire.
"""),
    md("""
## 6. Limites & éthique (section obligatoire)

**Biais de données**
- **Biais de reporting** : couverture médiatique et sources inégales selon les
  époques et les régions ; la hausse post-2004 mêle recrudescence réelle et
  meilleure documentation.
- **Incomplétude historique** : casualties manquantes (~9 %), groupe inconnu
  (~46 %), **année 1993 entièrement absente**.
- **Définition** : « terrorisme » repose sur les critères d'inclusion de START,
  qui comportent une part de jugement et évoluent dans le temps.

**Éthique**
- Usage **descriptif et académique** uniquement. Ces modèles **ne doivent pas**
  servir à prédire des attaques individuelles, profiler des communautés, ou
  justifier des mesures sécuritaires ciblées — les corrélations historiques ne
  sont ni causales ni prescriptives.
- Un indice de « pression » par pays peut être **stigmatisant** s'il est sorti
  de son contexte : il mesure une exposition historique, pas une culpabilité ni
  un risque futur.
- Toute réutilisation devrait s'accompagner d'une réflexion sur les
  conséquences (dual-use) et respecter la licence de la GTD.
"""),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python",
                              "name": "python3"}
    nb.cells = cells
    print(f"[notebook] executing {len(cells)} cells...")
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(HERE)}}).execute()
    nbf.write(nb, NB_PATH)
    print(f"[notebook] written -> {NB_PATH}")


if __name__ == "__main__":
    main()
