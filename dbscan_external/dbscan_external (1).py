import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
# =============================
# 1. Paths
# =============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

# Eliminamos plots viejos
for file in os.listdir(PLOTS_DIR):
    file_path = os.path.join(PLOTS_DIR, file)

    if os.path.isfile(file_path) and file.lower().endswith(".png"):
        os.remove(file_path)

print("plots viejos borrados.")


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# =============================
# 2. Load dataset
# =============================

TRAIN_PATH = os.path.join(DATA_DIR, "trainingData.csv")

df = pd.read_csv(TRAIN_PATH)

print("Dataset loaded:", df.shape)


# =============================
# 3. Select WAP features
# =============================

X = df.filter(regex="^WAP")
print("WAP shape:", X.shape)


# =============================
# 4. Replace 100 → -110
# =============================

X = X.replace(100, -110)

# =============================
# POWED representation (según paper)
# =============================
# Acá se implementa: Positive = RSS - min_rssi (clip >= 0)
# y Powed = (Positive^beta) / ((-min_rssi)^beta) -> rango [0,1]
min_rssi = -110.0   # valor usado para los "no detect" (coherente con -110)
beta = 2.0          # exponent; lo puedes cambiar a 1.0, 1.5, 2.0 en pruebas
print("Representación: POWED, beta =", beta)

# Acá se crea Positive = RSS - min_rssi y se fuerza mínimo 0
X_pos = (X - min_rssi).clip(lower=0.0)

# Acá se normaliza por la potencia máxima posible (-min_rssi)^beta
X_scaled = (X_pos ** beta) / ((-min_rssi) ** beta)
# X_powed
# =============================
# 5. Standardize
# =============================

#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X_powed)


# =============================
# 6. PCA (Full)
# =============================

pca_full = PCA()
pca_full.fit(X_scaled)

explained_variance_ratio = pca_full.explained_variance_ratio_
cum_sum = np.cumsum(explained_variance_ratio)


# =============================
# 7. Scree Plot
# =============================

plt.figure(figsize=(10, 6))

plt.bar(
    range(1, len(explained_variance_ratio) + 1),
    explained_variance_ratio,
    alpha=0.5,
    label="Individual"
)

plt.step(
    range(1, len(cum_sum) + 1),
    cum_sum,
    where="mid",
    label="Cumulative"
)

plt.axhline(y=0.80, linestyle="--", label="80%")
plt.axhline(y=0.95, linestyle="--", label="95%")

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot")
plt.legend()
plt.grid()

plt.savefig(os.path.join(PLOTS_DIR, f"new_scree_{timestamp}.png"), dpi=300)
plt.close()


# =============================
# 8. Choose k PCA
# =============================

k_80 = np.argmax(cum_sum >= 0.80) + 1
k_95 = np.argmax(cum_sum >= 0.95) + 1

print(f"Components for 80%: {k_80}")
print(f"Components for 95%: {k_95}")

k_pca = k_80


# =============================
# 9. PCA Final
# =============================

pca_final = PCA(n_components=k_pca, random_state=42)
X_pca = pca_final.fit_transform(X_scaled)

print("After PCA:", X_pca.shape)
loadings = np.abs(pca_final.components_)

mean_loading = loadings.mean(axis=0)

top_waps = np.argsort(mean_loading)[-20:]

print("Top WAPs used in DBSCAN PCA:")
print(top_waps)

# =============================
# 15. K-Distance (DBSCAN Prep)
# =============================

min_samples = 2*k_pca

print("DBSCAN min_samples:", min_samples)


neigh = NearestNeighbors(n_neighbors=min_samples)
nbrs = neigh.fit(X_pca)

distances, _ = nbrs.kneighbors(X_pca)

k_distances = np.sort(distances[:, min_samples - 1])
eps_value = np.percentile(k_distances, 90)

print(f"Suggested eps (90th percentile): {eps_value:.2f}")

plt.figure(figsize=(9, 6))
plt.plot(k_distances)

plt.xlabel("Sorted Points")
plt.ylabel("k-NN Distance (ε)")
plt.title("K-Distance Plot (DBSCAN)")
plt.grid()

plt.savefig(os.path.join(PLOTS_DIR, f"new_kdistance_{timestamp}.png"), dpi=300)
plt.close()

from sklearn.cluster import DBSCAN

# Selección manual desde k-distance
   # ajusta según tu gráfico

print("Using eps:", eps_value)

dbscan = DBSCAN(
    eps=eps_value,
    min_samples=min_samples,
    metric="euclidean"
)

db_labels = dbscan.fit_predict(X_pca)

df = df.copy()
df["DBSCAN_Cluster"] = db_labels

# Sacando informacion de los clusters DBSCAN
unique, counts = np.unique(db_labels, return_counts=True)

print("DBSCAN clusters:")
for u, c in zip(unique, counts):
    print(f"Cluster {u}: {c}")

# t-SNE para DBSCAN
tsne_db = TSNE(
    n_components=2,
    random_state=42,
    init="pca",
    perplexity=30
)

X_tsne_db = tsne_db.fit_transform(X_pca)

plt.figure(figsize=(10, 7))

for lab in np.unique(db_labels):

    mask = db_labels == lab

    if lab == -1:
        plt.scatter(
            X_tsne_db[mask, 0],
            X_tsne_db[mask, 1],
            s=10,
            c="gray",
            alpha=0.3,
            label="Noise"
        )
    else:
        plt.scatter(
            X_tsne_db[mask, 0],
            X_tsne_db[mask, 1],
            s=12,
            alpha=0.7,
            label=f"Cluster {lab}"
        )

plt.legend()
plt.title("DBSCAN + t-SNE")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")

plt.savefig(os.path.join(PLOTS_DIR, f"new_dbscan_tsne_{timestamp}.png"), dpi=300)
plt.close()

# COMENZANDO EVALUACION EXTERNA

#--------------------------
# GROUND TRUTH
# -------------------------
building = df['BUILDINGID'].astype(int)
floor = df['FLOOR'].astype(int)
building_floor = building.astype(str) + "_" + floor.astype(str)

print("\nGround truth labels:")
print("  Buildings:", building.nunique())
print("  Floors:", floor.nunique())
print("  Building-Floor combinations:", building_floor.nunique())

print("\nDBSCAN Results:")
unique_clusters = np.unique(db_labels)
n_clusters = len(unique_clusters[unique_clusters != -1])
n_noise = np.sum(db_labels == -1)

print("  Clusters found:", n_clusters)
print("  Noise points:", n_noise)
print("  Total labels:", len(unique_clusters))

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)

def compute_external_metrics(y_true, y_pred):
    return {
        'ARI': adjusted_rand_score(y_true, y_pred),
        'NMI': normalized_mutual_info_score(y_true, y_pred),
        'Homogeneity': homogeneity_score(y_true, y_pred),
        'Completeness': completeness_score(y_true, y_pred),
        'V-measure': v_measure_score(y_true, y_pred),
    }

mask_clean = db_labels != -1

building_clean = building[mask_clean]
bf_clean = building_floor[mask_clean]
labels_clean = db_labels[mask_clean]

print("\nAfter removing noise:")
print("  Points evaluated:", len(labels_clean))
print("  Clusters evaluated:", len(np.unique(labels_clean)))

print("\n" + "="*70)
print("EXTERNAL METRICS (WITHOUT NOISE)")
print("="*70)

metrics_building_clean = compute_external_metrics(building_clean, labels_clean)
metrics_bf_clean = compute_external_metrics(bf_clean, labels_clean)

df_external_clean = pd.DataFrame([
    {"Target": "BUILDINGID", **metrics_building_clean},
    {"Target": "BUILDING_FLOOR", **metrics_bf_clean}
])

print(df_external_clean.to_string(index=False))

import matplotlib.pyplot as plt
import numpy as np

metrics_to_plot = ['ARI', 'NMI', 'Homogeneity', 'Completeness', 'V-measure']

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(metrics_to_plot))
values = [metrics_building_clean[m] for m in metrics_to_plot]

ax.bar(x, values)

ax.set_xticks(x)
ax.set_xticklabels(metrics_to_plot)
ax.set_ylabel("Score")
ax.set_title("DBSCAN External Metrics vs BUILDINGID (No Noise)")
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

import seaborn as sns

ct_building = pd.crosstab(
    pd.Series(labels_clean, name='Cluster'),
    pd.Series(building_clean, name='BUILDINGID'),
    normalize='index'
)

plt.figure(figsize=(8, 10))
sns.heatmap(ct_building, annot=True, fmt='.2f', cmap='YlGnBu')
plt.title("DBSCAN: Cluster vs BUILDINGID")
plt.tight_layout()
plt.show()
