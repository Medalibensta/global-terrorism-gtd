"""
Cleaning and typing of the GTD slim table.

The GTD has well-known data-quality quirks that must be handled explicitly:

    * casualties (`nkill`, `nwound`) are missing for many older/unresolved
      incidents — treated as 0 for aggregate counts, but a `casualties_known`
      flag is kept so analyses can exclude imputed zeros;
    * `imonth`/`iday` can be 0 (unknown) — clipped to 1 for a valid date;
    * `nperps` uses -99 / -9 as unknown sentinels — set to NaN;
    * 1993 is entirely missing from the GTD (data lost by START) — flagged;
    * a derived `severity` = nkill + 0.5·nwound and a binary `lethal` = nkill>0
      target for the modelling step.

Run:
    python src/preprocessing.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data import RANDOM_STATE, load_raw

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CLEAN_PARQUET = DATA_DIR / "processed" / "gtd_clean.parquet"


def clean(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the cleaned, typed, feature-augmented incident table."""
    df = load_raw() if df is None else df.copy()

    # --- casualties -----------------------------------------------------
    df["casualties_known"] = df["n_kill"].notna() & df["n_wound"].notna()
    df["n_kill"] = df["n_kill"].fillna(0).clip(lower=0)
    df["n_wound"] = df["n_wound"].fillna(0).clip(lower=0)
    df["severity"] = df["n_kill"] + 0.5 * df["n_wound"]
    df["lethal"] = (df["n_kill"] > 0).astype(int)

    # --- perpetrators sentinels -> NaN ----------------------------------
    df["n_perps"] = df["n_perps"].replace({-99: np.nan, -9: np.nan})
    df.loc[df["n_perps"] < 0, "n_perps"] = np.nan

    # --- dates ----------------------------------------------------------
    df["month_clean"] = df["month"].replace(0, 1).clip(1, 12)
    df["day_clean"] = df["day"].replace(0, 1).clip(1, 28)
    df["date"] = pd.to_datetime(
        dict(year=df["year"], month=df["month_clean"], day=df["day_clean"]),
        errors="coerce")
    df["decade"] = (df["year"] // 10 * 10).astype(int)
    df["is_1993_gap"] = df["year"] == 1993  # GTD-known missing year

    # --- text tidiness --------------------------------------------------
    for col in ["attack_type", "target_type", "weapon_type", "region",
                "country", "group_name"]:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
    df["group_known"] = (df["group_name"] != "Unknown") & \
                        (df["group_name"] != "Unknown Group")

    # Valid coordinates flag (many incidents lack geocoding).
    df["geo_known"] = df["lat"].between(-90, 90) & df["lon"].between(-180, 180)

    return df.reset_index(drop=True)


def save(df: pd.DataFrame) -> None:
    CLEAN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_PARQUET, index=False)
    print(f"[preprocess] saved -> {CLEAN_PARQUET.relative_to(DATA_DIR.parent)}")


def load_clean() -> pd.DataFrame:
    if CLEAN_PARQUET.exists():
        return pd.read_parquet(CLEAN_PARQUET)
    out = clean()
    save(out)
    return out


if __name__ == "__main__":
    frame = clean()
    save(frame)
    print(f"{len(frame):,} incidents | {frame.year.min()}-{frame.year.max()}")
    print(f"casualties known : {frame.casualties_known.mean()*100:.1f}%")
    print(f"geo known        : {frame.geo_known.mean()*100:.1f}%")
    print(f"group identified : {frame.group_known.mean()*100:.1f}%")
    print(f"total killed      : {frame.n_kill.sum():,.0f}")
    print(f"lethal incidents  : {frame.lethal.mean()*100:.1f}%")
