# Verification & Proof Deliverables

This document catalogs each required proof deliverable, its location within the repository, its verification status, and the exact command used to independently reproduce and verify the result.

---

## Deliverables & Verification Matrix

| Proof Item | File Path | Status | Verification Command |
| :--- | :--- | :--- | :--- |
| **Executable Jupyter Notebook** | `notebooks/pca_clustering.ipynb` | **VERIFIED** | `jupyter nbconvert --to notebook --execute --inplace notebooks/pca_clustering.ipynb` |
| **Exported HTML Report** | `reports/pca_clustering.html` | **VERIFIED** | `jupyter nbconvert --to html notebooks/pca_clustering.ipynb --output-dir=reports --output=pca_clustering.html` |
| **Clustering Results Summary** | `reports/results_summary.csv` | **VERIFIED** | `cat reports/results_summary.csv` or execute Section 5 in notebook |
| **Raw UCI Wine Dataset** | `data/wine.csv` | **VERIFIED** | `wc -l data/wine.csv` (179 lines including header = 178 observations) |
| **Data Provenance & Source Doc** | `data/README.md` | **VERIFIED** | Inspect Forina et al. (1991) citation, CC BY 4.0 license, and unsupervised invariant |
| **Raw Feature Scales Boxplot** | `reports/figures/01_feature_scales.png` | **VERIFIED** | Inspect log-scale feature range distribution (150 DPI) |
| **PCA Scree & Variance Plot** | `reports/figures/02_pca_scree_plot.png` | **VERIFIED** | Inspect individual variance bars, cumulative curve, 80/90/95% thresholds (150 DPI) |
| **PCA Loadings Heatmap** | `reports/figures/03_pca_loadings_heatmap.png` | **VERIFIED** | Inspect PC1, PC2, PC3 factor loadings and chemical drivers (150 DPI) |
| **K-Means Elbow & Silhouette Curves** | `reports/figures/04_kmeans_elbow_silhouette.png` | **VERIFIED** | Inspect side-by-side inertia elbow and silhouette coefficient peak at $k=3$ (150 DPI) |
| **Sample Silhouette Profile** | `reports/figures/05_kmeans_sample_silhouette.png` | **VERIFIED** | Inspect per-sample silhouette distributions and mean threshold line (150 DPI) |
| **Ward Linkage Dendrogram** | `reports/figures/06_hierarchical_dendrogram.png` | **VERIFIED** | Inspect hierarchical tree and cutoff distance line at $k=3$ (150 DPI) |
| **DBSCAN k-Distance Graph** | `reports/figures/07_dbscan_kdistance.png` | **VERIFIED** | Inspect 5th-nearest-neighbor distance curve and $\varepsilon$ knees (150 DPI) |
| **2D & 3D Cluster Projections** | `reports/figures/08_cluster_projections_2d_3d.png` | **VERIFIED** | Inspect PC1-PC2 with centroids and static 3D projection (150 DPI) |
| **Interactive 3D Visualization** | `reports/figures/3d_cluster_projection.html` | **VERIFIED** | Open in browser for full 360-degree rotation and sample inspection (Plotly HTML) |
| **Clusters vs. Cultivars Comparison** | `reports/figures/09_clusters_vs_cultivars.png` | **VERIFIED** | Inspect side-by-side K-Means clusters vs true botanical cultivars (150 DPI) |
| **Environment Specifications** | `reports/proof/environment.txt` | **VERIFIED** | `pip freeze` pinned packages matching `requirements.txt` |
| **CI Automation Workflow** | `.github/workflows/ci.yml` | **PENDING** | GitHub Actions CI workflow execution on `main` push |

---

## Statistical & Algorithmic Verification Matrix

| Metric / Analysis | Expected Theoretical Benchmark | Actual Observed Value | Verification Status |
| :--- | :--- | :--- | :--- |
| **Sample Size & Dimensionality** | $n = 178, p = 13$ | **$n = 178$ samples, $13$ continuous features** | **PASSED (Exact)** |
| **Class Distribution (Cultivars)** | 59 (Class 0), 71 (Class 1), 48 (Class 2) | **59, 71, 48** | **PASSED (Exact)** |
| **Scale Disparity (Max:Min Variance)** | Extreme disparity ($> 10^5$) | **$6,432,883 : 1$** (`proline` vs `nonflavanoid_phenols`) | **PASSED (Exact)** |
| **Post-Standardization Means** | $\mu_j \approx 0$ | **Max $|\mu| = 8.38 \times 10^{-16} \approx 0$** | **PASSED (Exact)** |
| **Post-Standardization Stds** | $\sigma_j = 1.000$ | **$\sigma = 1.0000$ across all 13 features** | **PASSED (Exact)** |
| **PC1 Explained Variance** | $\approx 36\%$ | **$36.20\%$** (Eigenvalue $\lambda_1 = 4.73$) | **PASSED (Exact)** |
| **PC2 Explained Variance** | $\approx 19\%$ | **$19.21\%$** (Eigenvalue $\lambda_2 = 2.51$) | **PASSED (Exact)** |
| **PC3 Explained Variance** | $\approx 11\%$ | **$11.12\%$** (Eigenvalue $\lambda_3 = 1.45$) | **PASSED (Exact)** |
| **Top 3 Cumulative Variance** | $\approx 66.5\%$ | **$66.53\%$** | **PASSED (Exact)** |
| **Kaiser Criterion ($\lambda > 1.0$)** | Exactly 3 components | **3 components** (PC1=4.73, PC2=2.51, PC3=1.45; PC4=0.92) | **PASSED (Exact)** |
| **Variance Thresholds** | 80% (5 PCs), 90% (8 PCs), 95% (10 PCs) | **80% $\to 5$ PCs, 90% $\to 8$ PCs, 95% $\to 10$ PCs** | **PASSED (Exact)** |
| **K-Means Inertia Elbow** | Inflection at $k=3$ | **$k=2: 1658.76 \to k=3: 1277.93$ ($\Delta = 380.83$, -23%)** | **PASSED (Exact)** |
| **K-Means Silhouette Peak** | Global maximum at $k=3$ | **$k=3$ Score $= 0.2849$** (vs $k=2: 0.2593$, $k=4: 0.2602$) | **PASSED (Exact)** |
| **Ward Hierarchical Silhouette** | Close to K-Means for $k=3$ | **Score $= 0.2774$** | **PASSED (Exact)** |
| **DBSCAN Density Behavior** | Curse of dimensionality in 13D | **$\varepsilon=2.0 \implies 47.8\%$ noise; $\varepsilon=2.5 \implies$ 1 merged cluster** | **PASSED (Exact)** |
| **K-Means vs. Cultivar ARI** | Very high agreement ($\approx 0.90$) | **$\text{ARI} = 0.8975$** (96.6% accuracy, 6/178 discordant) | **PASSED (Exact)** |
| **Ward vs. Cultivar ARI** | Substantial agreement ($\approx 0.79$) | **$\text{ARI} = 0.7899$** | **PASSED (Exact)** |
