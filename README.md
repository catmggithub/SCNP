# SCNP: Single-Chain Nanoparticle Analysis with Diffusion Maps

Code for reproducing results from the paper: [Uncovering Patterns in Single-Chain Nanoparticle Formation with Diffusion Maps](https://drive.google.com/file/d/1-7boAG0BeFYtlzX715l6qX8tP-b2ybJ0/view?usp=sharing)

This method uses diffusion maps to find low-dimensional patterns in SCNP trajectory data.

---

| File | Description |
|------|-------------|
| `diffusionMap.py` | Computes diffusion map eigenvectors per trajectory || `createUmap.py` | Projects eigenvectors into 2D with UMAP |
| `visualizeUMAP.py` | Plots UMAP output |
| `initialConditions.py` | Links eigenvectors to starting conditions |
| `randomForest.py` | Predicts patterns using random forests |
| `plotTrajectories.py` | Plots SCNP trajectories |
| `visualizeEig2.py`, `visualizeEig3.py` | Plots 2nd & 3rd eigenvectors |
| `visualizeDensityHist.py` | Density plots for embeddings |

---
