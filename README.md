# Pooled accuracy is not transfer: dataset confounding across public EEG datasets

Analysis code for the article

> Javed A, Guo B, Junior MM, Ahmed M. *Pooled Accuracy Is Not Transfer: Evaluating Dataset Confounding When Pooling Public EEG Datasets for Anxiety Classification.* Submitted to *Neuroinformatics*.

The scripts pool five public EEG datasets, harmonise them to ten common channels, and test how much of a pooled participant-wise estimate reflects within-dataset discrimination, differences between datasets, and transfer to an unseen dataset (leave-one-dataset-out, LODO). Every table and figure in the article is produced by the scripts below.

## Contents

| Path | Purpose |
|---|---|
| `rev/` | Shared code: paths, dataset loaders, models, training and statistics |
| `00_check_env.py` | Checks the Python environment and data paths |
| `01_audit_existing_results.py` | Optional audit of an earlier analysis (needs files that are not part of this repository) |
| `02_build_corpus_v2.py` | Loads the five datasets, harmonises them and saves the corpus (`outputs/02_corpus_v2/`) |
| `03_within_dataset_cv.py` | Within-dataset participant-grouped cross-validation |
| `04_pooled_repeated.py` | Pooled evaluation over 5 participant splits × 3 seeds, dataset-prior baseline |
| `05_lodo_multiseed.py` | Leave-one-dataset-out, 3 seeds, participant-cluster bootstrap CIs |
| `06_architectures.py` | Seven classifiers and their computational cost |
| `07_positive_control.py` | Mental arithmetic versus rest, within and across datasets |
| `08_adversarial_v2.py` | Dataset-adversarial training (DANN), λ sweep, identity probe, collapse check |
| `09_euclidean_alignment.py` | Euclidean alignment |
| `10_normalisation_sensitivity.py` | Ratio-preserving and absolute-amplitude scaling |
| `11_balanced_prior.py` | Class-balancing and sample-size controls |
| `12_sam40_task_labels.py` | Alternative SAM 40 labelling |
| `13_spectral_signature.py` | Spectra, spectral parameterisation, frontal alpha asymmetry, age |
| `16_state_only_lodo.py` | LODO restricted to the three state-labelled datasets |
| `15_figures.py`, `17_old_figures_redraw.py`, `18_final_figures.py` | Figures (final numbered figures in `outputs/18_final_figures/`) |
| `colab_runner.ipynb` | Runs the GPU-heavy steps on Google Colab |
| `outputs/` | Result tables (`.csv`, `RESULTS.txt`), per-window predictions (`preds/*.npz`) and figures |

Step 14 does not exist; the numbering follows the order in which the analyses were added.

## Data

The datasets are not redistributed here. Download them from their providers:

| Dataset | Source |
|---|---|
| DASPS | https://ieee-dataport.org/open-access/dasps-database |
| SAM 40 | https://doi.org/10.6084/m9.figshare.14562090 |
| Mental Arithmetic (PhysioNet EEGMAT) | https://physionet.org/content/eegmat/ |
| Mendeley GAD-7 | https://doi.org/10.17632/rh3fy75zdv.1 |
| SRM resting state (OpenNeuro ds003775) | https://doi.org/10.18112/openneuro.ds003775.v1.2.1 |

Place them under one folder (default: `data/` in this repository) with this layout:

```
data/
├── SAM40/Data/                      raw_data/*.mat, scales.xls, Coordinates.locs
├── MentalArithmetic/eeg-during-mental-arithmetic-tasks-1.0.0/   Subject*.edf, subject-info.csv
├── DASPS/DASPS+HAM labels/          DASPS+HAM labels.mat
├── DASPS/Documents and Code/        participant_rating_dasps.xlsx
├── Mendeley/Anxiety_Dataset/Multi-channel Wireless EEG Recordings/   <class folders>/*.mat
└── BIDS/EEG/BIDS-structured resting-state electroencephalography (EEG) data extracted from an experimental paradigm/
                                     the OpenNeuro dataset: participants.tsv and derivatives/cleaned_data (*.set)
```

To keep the data elsewhere, set `EEG_DATA_ROOT`:

```bash
# Windows (PowerShell)
$env:EEG_DATA_ROOT = "D:\eeg_data"
# Linux / macOS
export EEG_DATA_ROOT=/path/to/eeg_data
```

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install torch                     # or the CUDA build from https://pytorch.org
pip install -r requirements.txt
python 00_check_env.py
```

Tested with Python 3.11 (Windows, CPU) and Python 3.12 (Google Colab, NVIDIA T4).

## Reproducing the results

Run the scripts in numerical order from the repository root, for example:

```bash
python 02_build_corpus_v2.py
python 03_within_dataset_cv.py
python 04_pooled_repeated.py
python 05_lodo_multiseed.py
...
python 15_figures.py
python 17_old_figures_redraw.py
python 18_final_figures.py
```

Each step writes to `outputs/<step>/` and can be resumed: finished runs are cached. Steps 03–12 and 16 train neural networks and are much faster on a GPU (see `colab_runner.ipynb`). Steps 13, 15, 17 and 18 need no training.

Files that are large or can be regenerated are not tracked: the harmonised corpus arrays (`outputs/02_corpus_v2/X_*.npy`, about 670 MB) and the LODO model weights (`outputs/05_lodo/models/`). Run step 02 to recreate the corpus.

## Licence

Code: MIT (see `LICENSE`). The datasets keep the licences of their providers.

## Citation

See `CITATION.cff`. Please also cite the five dataset publications listed in the article.
