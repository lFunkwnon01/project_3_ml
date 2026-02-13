# CS3061 – Machine Learning – Project 3: Clustering (UJIIndoorLoc)

This repository contains our implementation for **Project 3: Clustering**, using the **UJIIndoorLoc** WiFi fingerprint dataset (RSSI from WAPs) to explore **unsupervised learning**:
- **K-Means** and **DBSCAN** clustering
- Dimensionality reduction for visualization (**PCA**, **t-SNE**)
- **Internal evaluation** (e.g., Silhouette, Davies–Bouldin, Calinski–Harabasz)
- **External evaluation** using ground-truth labels (e.g., **BUILDINGID**, **FLOOR**, **BUILDING_FLOOR**) with ARI / NMI / Homogeneity / Completeness / V-measure

Project instructions reference: **CS3061 – Project 3: Clustering** (Feb 6, 2026).

Dataset source: [UJIIndoorLoc (UCI)](https://archive.ics.uci.edu/dataset/310/ujiindoorloc) – you must download it separately.

---

## Repository Contents

Main scripts (as provided in this repo):

- **`kmeans_internal.py`**  
  Internal evaluation of K-Means for a range of `k` values (Elbow/Inertia, Silhouette, Davies–Bouldin, Calinski–Harabasz), plus silhouette plot and t-SNE plot for a final `k`.

- **`kmeans_external.py`**  
  External evaluation of K-Means for `k ∈ {7,16,17}` comparing clusters vs ground truth (`BUILDINGID`, `BUILDING_FLOOR`) and generating bar plots, heatmaps, and t-SNE colored by truth.

- **`dbscan_external (1).py`**  
  DBSCAN pipeline (includes PCA selection, k-distance plot, DBSCAN clustering, and external metrics after removing noise).  
  **Note:** this script expects a specific folder structure (`data/`, `results/plots/`) and uses SciPy.

- **`main_compare_representations (1).py`**  
  Experiments comparing representations (includes a POWED transformation idea, PCA, KMeans/DBSCAN, and multiple plots).

---

## Requirements

### Software
- Python **3.10+** recommended
- pip (latest)

### Python Libraries
Core libraries used across scripts:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `seaborn`
- `scipy` (required by DBSCAN scripts using `cdist`)

---

## Setup (Step-by-step)

### 1) Clone the repo
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd <YOUR_REPO_FOLDER>
```

### 2) Create and activate a virtual environment
Windows (PowerShell)
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
python -m pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn seaborn scipy
```
### 4) Download the dataset (UJIIndoorLoc)
Download from UCI: UJIIndoorLoc dataset (training file is typically trainingData.csv).
Place it somewhere on your machine.

Configure Dataset Path (IMPORTANT)
Two scripts (kmeans_internal.py, kmeans_external.py) currently use a hard-coded path:

```bash
DATA_PATH = r"C:\Users\ASUS\Downloads\trainingData.csv"
```

Before running, change DATA_PATH to point to your local trainingData.csv.

Example:

Windows: r"C:\Users\<you>\Downloads\trainingData.csv"
macOS/Linux: r"/Users/<you>/Downloads/trainingData.csv"
DBSCAN scripts path expectations
dbscan_external (1).py and main_compare_representations (1).py expect:

<repo_root>/
  data/
    trainingData.csv
  results/
    plots/
So for those scripts, the easiest setup is:

Create a data/ folder at repo root
Copy trainingData.csv into data/trainingData.csv
