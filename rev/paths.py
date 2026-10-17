# rev/paths.py
"""
All paths used by the scripts.

Set the environment variable EEG_DATA_ROOT to the folder that contains the five
datasets (see README.md for the expected layout). If it is not set, ./data inside
the repository is used. Outputs are written to ./outputs.
OLD_PROJECT_DIR is only needed for the optional audit of the earlier analysis (step 01).
"""
import os
from pathlib import Path

REV_DIR     = Path(__file__).resolve().parent.parent          # repository root
DATA_ROOT   = Path(os.environ.get("EEG_DATA_ROOT", REV_DIR / "data"))
# results of the earlier analysis; only needed by the optional audit (step 01)
OLD_PROJECT = Path(os.environ.get("OLD_PROJECT_DIR", REV_DIR / "earlier_analysis"))

OLD_CORPUS = OLD_PROJECT / "outputs" / "corpus.npz"
OLD_CACHE  = OLD_PROJECT / "results" / "cache"
OLD_CSV    = OLD_PROJECT / "results" / "csv"

DATASET_DIRS = {
    "SAM40":            DATA_ROOT / "SAM40",
    "MentalArithmetic": DATA_ROOT / "MentalArithmetic",
    "DASPS":            DATA_ROOT / "DASPS",
    "Mendeley":         DATA_ROOT / "Mendeley",
    "BIDS":             DATA_ROOT / "BIDS",
}

# metadata files (demographics / clinical scales)
META = {
    "MA_subject_info": DATA_ROOT / "MentalArithmetic" / "eeg-during-mental-arithmetic-tasks-1.0.0" / "subject-info.csv",
    "BIDS_participants": DATA_ROOT / "BIDS" / "EEG" /
        "BIDS-structured resting-state electroencephalography (EEG) data extracted from an experimental paradigm" /
        "participants.tsv",
    "DASPS_ratings": DATA_ROOT / "DASPS" / "Documents and Code" / "participant_rating_dasps.xlsx",
    "SAM40_scales": DATA_ROOT / "SAM40" / "Data" / "scales.xls",
}

OUT_DIR = REV_DIR / "outputs"


def out(sub: str) -> Path:
    """Create outputs/<sub>/ and return it."""
    p = OUT_DIR / sub
    p.mkdir(parents=True, exist_ok=True)
    return p
