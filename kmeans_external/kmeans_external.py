import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score, 
    normalized_mutual_info_score,
    homogeneity_score, 
    completeness_score, 
    v_measure_score
)

# -------------------------
# CONFIG
# -------------------------
RANDOM_STATE = 42

# Cambia esto a tu ruta real:
DATA_PATH = r"C:\Users\ASUS\Downloads\trainingData.csv"

# Carpeta para guardar figuras
OUTPUT_DIR = "figures_external"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# PCA y k
N_PCA = 176
K_CANDIDATES = [7, 16, 17]
TSNE_SAMPLE = 5000

# -------------------------
# 1) LOAD + PREPROCESO (igual que interno)
# -------------------------
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

X = df.filter(regex=r"^WAP", axis=1).copy()
X = X.replace(100, -110).astype(np.float32)

print("Dataset shape:", df.shape)
print("WAP features:", X.shape[1])

# Scale + PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=N_PCA, svd_solver="randomized", random_state=RANDOM_STATE)
X_reduced = pca.fit_transform(X_scaled)

print("Reduced to:", X_reduced.shape[1], "components")
print("Variance explained:", round(pca.explained_variance_ratio_.sum(), 4))

# -------------------------
# 2) ETIQUETAS REALES (ground truth)
# -------------------------
building = df['BUILDINGID'].astype(int)
floor = df['FLOOR'].astype(int)
building_floor = building.astype(str) + "_" + floor.astype(str)

print("\nGround truth labels:")
print("  Buildings:", building.nunique())
print("  Floors:", floor.nunique())
print("  Building-Floor combinations:", building_floor.nunique())

# -------------------------
# 3) ENTRENAR K-MEANS PARA k = 7, 16, 17
# -------------------------
models = {}
labels_dict = {}

for k in K_CANDIDATES:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
    labels = km.fit_predict(X_reduced)
    models[k] = km
    labels_dict[k] = labels
    print(f"\nK-Means k={k} trained. Cluster sizes:")
    print(pd.Series(labels).value_counts().sort_index().to_string())

print("\n" + "="*70)
print("EXTERNAL EVALUATION: Comparing clusters vs ground truth")
print("="*70)

# -------------------------
# 4) CALCULAR MÉTRICAS EXTERNAS
# -------------------------
def compute_external_metrics(y_true, y_pred):
    return {
        'ARI': adjusted_rand_score(y_true, y_pred),
        'NMI': normalized_mutual_info_score(y_true, y_pred),
        'Homogeneity': homogeneity_score(y_true, y_pred),
        'Completeness': completeness_score(y_true, y_pred),
        'V-measure': v_measure_score(y_true, y_pred),
    }

# Comparar contra BUILDINGID
results_building = []
for k in K_CANDIDATES:
    metrics = compute_external_metrics(building, labels_dict[k])
    metrics['k'] = k
    results_building.append(metrics)

df_building = pd.DataFrame(results_building)[['k', 'ARI', 'NMI', 'Homogeneity', 'Completeness', 'V-measure']]

# Comparar contra BUILDING_FLOOR
results_bf = []
for k in K_CANDIDATES:
    metrics = compute_external_metrics(building_floor, labels_dict[k])
    metrics['k'] = k
    results_bf.append(metrics)

df_bf = pd.DataFrame(results_bf)[['k', 'ARI', 'NMI', 'Homogeneity', 'Completeness', 'V-measure']]

print("\n" + "="*70)
print("EXTERNAL METRICS vs BUILDINGID")
print("="*70)
print(df_building.to_string(index=False))

print("\n" + "="*70)
print("EXTERNAL METRICS vs BUILDING_FLOOR")
print("="*70)
print(df_bf.to_string(index=False))

# -------------------------
# IMAGEN 1: Gráfico de barras comparativo (BUILDINGID)
# -------------------------
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(K_CANDIDATES))
width = 0.15

metrics_to_plot = ['ARI', 'NMI', 'Homogeneity', 'Completeness', 'V-measure']
for i, metric in enumerate(metrics_to_plot):
    values = df_building[metric].values
    ax.bar(x + i*width, values, width, label=metric)

ax.set_xlabel('k (número de clusters)', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('External Metrics vs BUILDINGID', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(K_CANDIDATES)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "external_metrics_buildingid.png"), dpi=160)
plt.show()

# -------------------------
# IMAGEN 2: Gráfico de barras comparativo (BUILDING_FLOOR)
# -------------------------
fig, ax = plt.subplots(figsize=(10, 6))
for i, metric in enumerate(metrics_to_plot):
    values = df_bf[metric].values
    ax.bar(x + i*width, values, width, label=metric)

ax.set_xlabel('k (número de clusters)', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('External Metrics vs BUILDING_FLOOR', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(K_CANDIDATES)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "external_metrics_building_floor.png"), dpi=160)
plt.show()

# -------------------------
# IMAGEN 3: Heatmap Cluster vs BUILDINGID (k=16)
# -------------------------
labels_16 = labels_dict[16]

ct_building = pd.crosstab(
    pd.Series(labels_16, name='Cluster'),
    pd.Series(building, name='BUILDINGID'),
    normalize='index'
)

plt.figure(figsize=(8, 10))
sns.heatmap(ct_building, annot=True, fmt='.2f', cmap='YlGnBu', cbar_kws={'label': 'Proporción'})
plt.title('Heatmap: Cluster vs BUILDINGID (k=16)', fontsize=14, fontweight='bold')
plt.xlabel('BUILDINGID (Ground Truth)', fontsize=12)
plt.ylabel('Cluster ID', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "external_heatmap_buildingid_k16.png"), dpi=160)
plt.show()

# -------------------------
# IMAGEN 4: Heatmap Cluster vs BUILDING_FLOOR (k=16)
# -------------------------
ct_bf = pd.crosstab(
    pd.Series(labels_16, name='Cluster'),
    pd.Series(building_floor, name='Building_Floor'),
    normalize='index'
)

plt.figure(figsize=(12, 10))
sns.heatmap(ct_bf, annot=True, fmt='.2f', cmap='RdYlGn', cbar_kws={'label': 'Proporción'})
plt.title('Heatmap: Cluster vs BUILDING_FLOOR (k=16)', fontsize=14, fontweight='bold')
plt.xlabel('Building_Floor (Ground Truth)', fontsize=12)
plt.ylabel('Cluster ID', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "external_heatmap_building_floor_k16.png"), dpi=160)
plt.show()

# -------------------------
# t-SNE (muestra para visualización)
# -------------------------
rng = np.random.default_rng(RANDOM_STATE)
n_sample = min(TSNE_SAMPLE, X_reduced.shape[0])
idx = rng.choice(X_reduced.shape[0], size=n_sample, replace=False)

tsne = TSNE(
    n_components=2, 
    random_state=RANDOM_STATE, 
    init='pca', 
    learning_rate='auto', 
    perplexity=30, 
    max_iter=1000
)
X_tsne = tsne.fit_transform(X_reduced[idx])

# -------------------------
# IMAGEN 5: t-SNE coloreado por BUILDINGID (ground truth)
# -------------------------
plt.figure(figsize=(11, 7))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=building.iloc[idx], cmap='tab10', s=15, alpha=0.7)
plt.colorbar(scatter, label='BUILDINGID', ticks=[0, 1, 2])
plt.title('t-SNE colored by Ground Truth: BUILDINGID', fontsize=14, fontweight='bold')
plt.xlabel('t-SNE 1', fontsize=12)
plt.ylabel('t-SNE 2', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "external_tsne_buildingid_truth.png"), dpi=160)
plt.show()

# -------------------------
# IMAGEN 6: t-SNE coloreado por BUILDING_FLOOR (ground truth)
# -------------------------
bf_sample = building_floor.iloc[idx]
bf_codes, bf_uniques = pd.factorize(bf_sample)

plt.figure(figsize=(11, 7))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=bf_codes, cmap='tab20', s=15, alpha=0.7)
plt.colorbar(scatter, label='Building_Floor Code')
plt.title('t-SNE colored by Ground Truth: BUILDING_FLOOR', fontsize=14, fontweight='bold')
plt.xlabel('t-SNE 1', fontsize=12)
plt.ylabel('t-SNE 2', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "external_tsne_building_floor_truth.png"), dpi=160)
plt.show()

print("\n" + "="*70)
print("EXTERNAL EVALUATION COMPLETE")
print("="*70)
print(f"Total de 6 imágenes guardadas en: {OUTPUT_DIR}/")
print("\nResumen de imágenes generadas:")
print("  1. external_metrics_buildingid.png")
print("  2. external_metrics_building_floor.png")
print("  3. external_heatmap_buildingid_k16.png")
print("  4. external_heatmap_building_floor_k16.png")
print("  5. external_tsne_buildingid_truth.png")
print("  6. external_tsne_building_floor_truth.png")