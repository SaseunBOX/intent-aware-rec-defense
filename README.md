# Intent-Aware Multi-Stage Defense for Personalized Recommender Systems

This repository contains the code, experiment scripts, data seed files, images, and thesis artifacts for an intent-aware multi-stage defense prototype for personalized recommender systems.

The project studies how a multi-stage defense pipeline can reduce harmful content amplification while preserving recommendation utility. The core pipeline includes intent classification, risk classification, policy fusion, replacement-based reranking, and offline evaluation.

## Repository structure

- `scripts/`: runnable experiment scripts
- `src/`: modular reusable code
- `configs/`: configuration files
- `data/raw/external/moral_education/`: small moral-education seed pool used by the project
- `results/tables/`: main result tables
- `results/plots/`: generated experiment plots and evaluation figures
- `images/`: thesis-related image assets that may be used in the final paper
- `chapters/`: LaTeX thesis chapters

## Data

This repository includes the small `moral_education` seed pool located at:

```text
data/raw/external/moral_education/
```

The project also uses external datasets such as:

- MIND-small
- Aegis 2.0

Large third-party raw datasets are not redistributed in this repository. Please follow the thesis appendix and project documentation to reconstruct the full processed environment when using external datasets.

## Images and thesis artifacts

The `images/` directory stores figure assets that may appear in the thesis or supporting documentation. These files are separated from generated experimental plots in `results/plots/`:

- `images/`: manually prepared or thesis-related image assets
- `results/plots/`: plots generated from experiments or evaluation scripts

## Main outputs

The repository includes:

- code for preprocessing, training, reranking, and evaluation
- thesis-ready tables and plots
- image assets for thesis writing
- appendix-ready summary artifacts
- a small moral-education seed dataset for reproducibility

## Reproducibility

A typical workflow is:

1. Create the Python environment.
2. Prepare or download the required external datasets.
3. Use the included `data/raw/external/moral_education/` files as the moral-education seed pool.
4. Run the scripts in `scripts/` in the documented order.
5. Check generated tables in `results/tables/` and plots in `results/plots/`.

## Notes

Do not commit private credentials, API keys, `.env` files, or large model checkpoints. If large datasets or model files are needed, store them externally and document their download or reconstruction procedure.
