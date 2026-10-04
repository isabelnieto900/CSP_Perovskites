"""Reading and writing data, tables and figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import config


def load_master(path: Path | str = config.MASTER_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def load_processed(name: str) -> pd.DataFrame:
    return pd.read_csv(config.PROCESSED_DIR / name)


def save_csv(df: pd.DataFrame, path: Path | str, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    return save_csv(df, config.TABLES_DIR / name, index=index)


def save_fig(fig: plt.Figure, name: str, dpi: int = 400, directory: Path | str = config.FIGURES_DIR) -> Path:
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
