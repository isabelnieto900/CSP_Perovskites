"""Project-wide constants and paths."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("CSP_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
BINARY_DIR = PROCESSED_DIR / "binary"

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
SHAP_DIR = RESULTS_DIR / "shap"

MASTER_FILE = RAW_DIR / "Master_File.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CORR_THRESHOLD = 0.9
N_RUNS = 10
CV_FOLDS = 5

TARGET = "crystal_system"
META_COLS = ["Formula", "A", "B"]

CRYSTAL_SYSTEMS = [
    "Cubic", "Orthorhombic", "Trigonal",
    "Tetragonal", "Monoclinic", "Hexagonal", "Triclinic",
]

ENGINEERED_FEATURES = [
    "ARR_A", "ARR_B", "MPR_A", "MPR_B", "END_A", "END_B",
    "OF", "t", "BL_AB", "BL_AX", "BL_BX", "ENR",
]

TOP10_M4 = ["MPR_A", "OF", "ARR_B", "t", "r_A12", "END_A", "END_B", "ARR_A", "BP_A", "VR_A"]
TOP10_M7 = ["END_B", "MPR_A", "AM_A", "END_A", "IE_B", "BP_A", "ARR_B", "VR_B", "ARR_A", "t"]

TOP_MODELS = ["Random Forest", "Extra Trees", "LightGBM"]
