from __future__ import annotations

from pathlib import Path
import pandas as pd


def ensure_parent_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_df(df: pd.DataFrame, path_base: str | Path, index: bool = False) -> Path:
    p = Path(path_base)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        out = p.with_suffix(".parquet")
        df.to_parquet(out, index=index)
        return out
    except Exception:
        out = p.with_suffix(".pkl")
        df.to_pickle(out)
        return out


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    p = ensure_parent_dir(path)
    df.to_csv(p, index=index)
    return p


def load_df(path_base: str | Path) -> pd.DataFrame:
    p = Path(path_base)
    parquet = p.with_suffix(".parquet")
    pkl = p.with_suffix(".pkl")
    if parquet.exists():
        return pd.read_parquet(parquet)
    if pkl.exists():
        return pd.read_pickle(pkl)
    raise FileNotFoundError(f"No parquet or pkl found for base path: {p}")


def load_optional_df(path_base: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        return load_df(path_base)
    except FileNotFoundError:
        return pd.DataFrame(columns=columns)
