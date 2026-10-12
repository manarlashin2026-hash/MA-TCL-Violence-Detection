# MA-TCL: Explicit Temporal Contact Modeling for Violence Detection in Surveillance Video

Code accompanying the article *Explicit Temporal Contact Modeling for Violence Detection in Surveillance Video* (M. Lashin, O. Zahra, A. T. Elgammal, IEEE Access).

MA-TCL combines a VideoMAE appearance stream with a 16-dimensional Temporal Contact Length (TCL) descriptor computed from frame-level proximity and motion scores. A risk gate predicted from the TCL embedding modulates the appearance embedding before classification.

## Requirements

```
pip install -r requirements.txt
```

The experiments were run with PyTorch 2.10 (CUDA 12.8) on a single NVIDIA RTX 5060 (8 GB). The VideoMAE backbone (`MCG-NJU/videomae-base`) is downloaded automatically by `transformers`.

## Data

Both datasets are public and are not redistributed here.

* **SC-Fight-300**: Surveillance Camera Fight Dataset (Akti et al., IPTA 2019), 150 fight and 150 non-fight clips.
* **RWF-2000**: Cheng et al., ICPR 2020. The official `train`/`val` partition is used as training and test set.

Expected layout (relative to the repository root):

```
data/
  SC-Fight-300/fight/*.mp4
  SC-Fight-300/noFight/*.mp4
  RWF-2000/train/Fight/*.avi
  RWF-2000/train/NonFight/*.avi
  RWF-2000/val/Fight/*.avi
  RWF-2000/val/NonFight/*.avi
```

## Reproducing the reported results

All commands are run from the repository root.

| Result in the article | Command | Output folder |
|---|---|---|
| Table 3 (configuration 5), Table 4 | `python train_scfight_ma_tcl.py` | `fight_model_ma_tcl_final/` |
| Table 5, Fig. 5, Fig. 6 | `python train_rwf2000_official.py` | `rwf2000_official_split_results/` |
| Table 6 (TCL input set to zero) | `python train_rwf2000_official_noTCL.py` | `rwf2000_official_split_results_noTCL/` |
| Figs. 4, 7, 8; paired test of Table 6 | `python make_figures_from_videos.py` (after the three runs above) | `revision_figures/` |
| Fig. 9 | `python make_fig9_temporal_profiles.py revision_figures/tcl_features_scfight.csv revision_figures/tcl_features_rwf.csv <out_dir>` | `<out_dir>` |
| Figs. 6 and 10 | `python make_figures_from_logs.py rwf2000_official_split_results/training_history.json rwf2000_official_split_results/bootstrap_results.json rwf2000_official_split_results_noTCL/bootstrap_results.json <out_dir>` | `<out_dir>` |
| Table 3 (configurations 1-4) | scripts in `development/` (configuration 3 must be run before configuration 4) | see each script |

Notes:

* SC-Fight-300 uses a stratified 80/20 split with seed 42 (240 training / 60 validation clips).
* RWF-2000 results are reported for the final training epoch. Early stopping monitors macro-F1 on the official test partition because the benchmark defines no separate validation set.
* Confidence intervals are percentile bootstrap intervals (1,000 resamples) over the held-out predictions of the final-epoch model.
* The RWF-2000 training scripts save their full state after every epoch and resume automatically if restarted. `train_rwf2000_official_noTCL.py` caches the pre-processed frames on disk (about 10 GB) to shorten training; this does not change the inputs seen by the model.

## Logs of the reported runs

`results/` contains the training histories, bootstrap intervals, and split information of the runs reported in the article, and `results/revision_figures/` contains the per-clip TCL features, the per-clip predictions of the reported models, and the consistency checks written by `make_figures_from_videos.py`. For example, the final-epoch test accuracy on RWF-2000 is 89.50% for MA-TCL and 88.25% with the TCL input set to zero.

## Citation

If you use this code, please cite the article above.
