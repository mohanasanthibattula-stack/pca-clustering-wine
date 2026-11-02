# UCI Wine Dataset Provenance & Metadata

## Overview
This directory contains the canonical raw dataset `wine.csv` (178 samples, 13 continuous chemical constituents, and 1 integer target column `cultivar`).

## Dataset Details
- **File**: `data/wine.csv`
- **Samples ($n$)**: 178 observations
- **Features ($p$)**: 13 numeric physicochemical variables:
  1. `alcohol`
  2. `malic_acid`
  3. `ash`
  4. `alcalinity_of_ash`
  5. `magnesium`
  6. `total_phenols`
  7. `flavanoids`
  8. `nonflavanoid_phenols`
  9. `proanthocyanins`
  10. `color_intensity`
  11. `hue`
  12. `od280/od315_of_diluted_wines`
  13. `proline`
- **Target Variable**: `cultivar` (values `0`, `1`, `2` corresponding to Cultivar 1, Cultivar 2, Cultivar 3)
- **Class Distribution**:
  - Cultivar 0: 59 samples (33.1%)
  - Cultivar 1: 71 samples (39.9%)
  - Cultivar 2: 48 samples (27.0%)

## Scientific Source & Citation
- **Creators / Collectors**: M. Forina, et al., Institute of Pharmaceutical and Food Analysis and Technologies, Genoa, Italy.
- **Donor**: Stefan Aeberhard, Institute of Pharmaceutical and Food Analysis and Technologies, Genoa, Italy.
- **Repository**: [UCI Machine Learning Repository: Wine Dataset](https://archive.ics.uci.edu/dataset/109/wine) (1991).
- **Software Distribution**: Bundled within `scikit-learn` via `sklearn.datasets.load_wine()`.

## License
This dataset is licensed under a **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
You are free to share and adapt the material as long as appropriate credit is given.

## Unsupervised Modeling Invariant
In accordance with strict unsupervised learning methodology, the ground-truth `cultivar` labels are completely excluded from preprocessing, scaling, dimensionality reduction (PCA), and cluster formation. They are solely inspected at the post-hoc evaluation stage to compute Adjusted Rand Index (ARI) and contingency alignment.
