"""
Data loader for the Global Terrorism Database (GTD, START / University of
Maryland).

The GTD is NOT freely redistributable and has no public API. This loader looks
for the official CSV that the user has downloaded from Kaggle
(https://www.kaggle.com/datasets/START-UMD/gtd), tries a few common locations,
and caches a slimmed parquet of the columns this project uses. If the source
file is not found, it generates a documented synthetic surrogate with the same
schema and broad statistical shape so the pipeline still runs end-to-end.

The full raw CSV (~156 MB, 181 691 incidents, 1970-2017, latin-1 encoded) is
NOT versioned; only the slim parquet cache lives under data/processed/.

Run:
    python src/data.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SLIM_PARQUET = DATA_DIR / "processed" / "gtd_slim.parquet"

# Columns kept from the raw GTD (renamed to tidy snake_case on load).
COLUMNS = {
    "eventid": "event_id",
    "iyear": "year",
    "imonth": "month",
    "iday": "day",
    "country_txt": "country",
    "region_txt": "region",
    "provstate": "provstate",
    "city": "city",
    "latitude": "lat",
    "longitude": "lon",
    "success": "success",
    "suicide": "suicide",
    "attacktype1_txt": "attack_type",
    "targtype1_txt": "target_type",
    "weaptype1_txt": "weapon_type",
    "gname": "group_name",
    "nperps": "n_perps",
    "nkill": "n_kill",
    "nwound": "n_wound",
}

# Candidate locations for the official CSV (first match wins). Override with the
# GTD_CSV environment variable.
CANDIDATES = [
    os.environ.get("GTD_CSV", ""),
    str(DATA_DIR / "raw" / "globalterrorismdb_0718dist.csv"),
    str(Path.home() / "Downloads" / "globalterrorismdb_0718dist.csv"),
    str(Path.home() / "Desktop" / "globalterrorismdb_0718dist.csv"),
]


def _find_source() -> Path | None:
    for cand in CANDIDATES:
        if cand and Path(cand).exists():
            return Path(cand)
    return None


def _from_csv(path: Path) -> pd.DataFrame:
    """Read the official GTD CSV (latin-1) keeping only the needed columns."""
    print(f"[data] reading GTD from {path} ...")
    df = pd.read_csv(path, encoding="latin-1", usecols=list(COLUMNS),
                     low_memory=False)
    df = df.rename(columns=COLUMNS)
    print(f"[data] loaded {len(df):,} incidents ({df.year.min()}-{df.year.max()})")
    return df


def _synthetic(n: int = 181_000) -> pd.DataFrame:
    """
    Reproducible fallback mimicking GTD's schema and broad shape.

    Reproduces the key stylised facts: exponential growth of incidents after
    ~2004, a handful of high-activity regions, a heavy-tailed casualty
    distribution, and a dominant "Bombing/Explosion" attack type.
    """
    print("[data] official GTD not found — generating synthetic surrogate.")
    rng = np.random.default_rng(RANDOM_STATE)

    # Year: rising hazard, most mass in 2004-2017.
    years = np.clip(
        np.round(1970 + 47 * rng.beta(5.0, 1.6, n)).astype(int), 1970, 2017)
    regions = rng.choice(
        ["Middle East & North Africa", "South Asia", "Sub-Saharan Africa",
         "South America", "Western Europe", "Southeast Asia",
         "Eastern Europe", "North America"],
        size=n, p=[0.28, 0.24, 0.16, 0.09, 0.08, 0.07, 0.05, 0.03])
    attack = rng.choice(
        ["Bombing/Explosion", "Armed Assault", "Assassination",
         "Hostage Taking (Kidnapping)", "Facility/Infrastructure Attack",
         "Unknown"], size=n, p=[0.48, 0.24, 0.11, 0.07, 0.06, 0.04])
    target = rng.choice(
        ["Private Citizens & Property", "Military", "Police", "Government",
         "Business", "Religious Figures/Institutions"],
        size=n, p=[0.3, 0.2, 0.18, 0.15, 0.1, 0.07])
    weapon = rng.choice(
        ["Explosives", "Firearms", "Incendiary", "Melee", "Unknown"],
        size=n, p=[0.5, 0.32, 0.08, 0.05, 0.05])
    n_kill = rng.lognormal(0.4, 1.1, n).round().astype(int)
    n_wound = (n_kill * rng.uniform(0.5, 2.5, n)).round().astype(int)

    return pd.DataFrame({
        "event_id": np.arange(n),
        "year": years,
        "month": rng.integers(1, 13, n),
        "day": rng.integers(1, 29, n),
        "country": regions,          # coarse stand-in
        "region": regions,
        "provstate": "NA", "city": "NA",
        "lat": rng.uniform(-40, 60, n),
        "lon": rng.uniform(-120, 130, n),
        "success": rng.binomial(1, 0.88, n),
        "suicide": rng.binomial(1, 0.04, n),
        "attack_type": attack,
        "target_type": target,
        "weapon_type": weapon,
        "group_name": "Unknown",
        "n_perps": rng.integers(-99, 10, n),
        "n_kill": n_kill,
        "n_wound": n_wound,
    })


def load_raw(force_reload: bool = False) -> pd.DataFrame:
    """Return the slim GTD frame, using the parquet cache when present."""
    if SLIM_PARQUET.exists() and not force_reload:
        return pd.read_parquet(SLIM_PARQUET)

    src = _find_source()
    df = _from_csv(src) if src is not None else _synthetic()

    SLIM_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SLIM_PARQUET, index=False)
    print(f"[data] cached slim parquet -> "
          f"{SLIM_PARQUET.relative_to(DATA_DIR.parent)}")
    return df


if __name__ == "__main__":
    frame = load_raw(force_reload=True)
    print(frame.shape)
    print(frame.head())
    print("\nby region:\n", frame.region.value_counts())
