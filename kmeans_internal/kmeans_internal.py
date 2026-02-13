import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    davies_bouldin_score, calinski_harabasz_score
)

# -------------------------
# CONFIG
# -------------------------
RANDOM_STATE = 42

# Cambia esto a tu ruta real:
DATA_PATH = r"C:\Users\ASUS\Downloads\trainingData.csv"

# Carpeta para guardar figuras
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Rango de k a evaluar
K_MIN, K_MAX = 2, 20

# Para que silhouette/t-SNE no sea lentísimo con ~20k filas:
SIL_SAMPLE_N = 5000  # muestra para silhouette_score (None = todo)
TSNE_SAMPLE_N = 5000  # muestra para t-SNE
SIL_PLOT_SAMPLE_N = 5000  # muestra para silhouette plot

# PCA fijo (como su equipo)
N_PCA = 176

# k final para análisis/plots
K_FINAL = 16

# k candidatos para comparación
K_CANDIDATES = [7, 16, 17]

# -------------------------
# 1) LOAD + FEATURES
# -------------------------
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

X = df.filter(regex=r"^WAP", axis=1).copy()
if X.shape[1] == 0:
    raise ValueError("No se encontraron columnas WAP*. Revisa el CSV o nombres de columnas.")

# Reemplazar '100' (no detectado) por -110 (señal muy débil)
X = X.replace(100, -110).astype(np.float32)

print("Rows:", df.shape[0])
print("WAP features:", X.shape[1])

# -------------------------
# 2) SCALE + PCA
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=N_PCA, svd_solver="randomized", random_state=RANDOM_STATE)
X_reduced = pca.fit_transform(X_scaled)

print("Reduced dims:", X_reduced.shape[1])
print("Explained variance (sum):", float(pca.explained_variance_ratio_.sum()))

# -------------------------
# 3) INTERNAL EVAL ACROSS k (2-20)
# -------------------------
k_values = list(range(K_MIN, K_MAX + 1))

inertias = []
silhouettes = []
davies = []
calinskis = []

# Si hay más filas que SIL_SAMPLE_N, usamos sample_size para silhouette_score
sil_sample_size = SIL_SAMPLE_N if (SIL_SAMPLE_N is not None and X_reduced.shape[0] > SIL_SAMPLE_N) else None

for k in k_values:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
    labels = km.fit_predict(X_reduced)

    inertias.append(float(km.inertia_))
    silhouettes.append(
        float(silhouette_score(X_reduced, labels, sample_size=sil_sample_size, random_state=RANDOM_STATE)))
    davies.append(float(davies_bouldin_score(X_reduced, labels)))
    calinskis.append(float(calinski_harabasz_score(X_reduced, labels)))

metrics_df = pd.DataFrame({
    "k": k_values,
    "inertia": inertias,
    "silhouette(sample)": silhouettes,
    "davies_bouldin": davies,
    "calinski_harabasz": calinskis
})

print("\n=== INTERNAL METRICS TABLE (k=2-20) ===")
print(metrics_df.to_string(index=False))

# -------------------------
# 4) PLOTS (internal metrics vs k)
# -------------------------
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker="o")
plt.title("Elbow Method (Inertia) vs k")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "internal_elbow_inertia_vs_k.png"), dpi=160)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouettes, marker="o", color="crimson")
plt.title("Silhouette Score (sample) vs k")
plt.xlabel("k")
plt.ylabel("Silhouette")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "internal_silhouette_vs_k.png"), dpi=160)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(k_values, davies, marker="o", color="darkgreen")
plt.title("Davies–Bouldin Index vs k (lower is better)")
plt.xlabel("k")
plt.ylabel("DB Index")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "internal_davies_bouldin_vs_k.png"), dpi=160)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(k_values, calinskis, marker="o", color="purple")
plt.title("Calinski–Harabasz Index vs k (higher is better)")
plt.xlabel("k")
plt.ylabel("CH Index")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "internal_calinski_harabasz_vs_k.png"), dpi=160)
plt.show()

# -------------------------
# 5) COMPARACIÓN DETALLADA: k = 7, 16, 17
# -------------------------
print("\n" + "=" * 60)
print("COMPARATIVE ANALYSIS: k = 7, 16, 17")
print("=" * 60)

comparison_rows = []

for k in K_CANDIDATES:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
    labels = km.fit_predict(X_reduced)

    cluster_sizes = pd.Series(labels).value_counts()

    comparison_rows.append({
        "k": k,
        "inertia": float(km.inertia_),
        "silhouette": float(silhouette_score(
            X_reduced, labels,
            sample_size=sil_sample_size,
            random_state=RANDOM_STATE
        )),
        "davies_bouldin": float(davies_bouldin_score(X_reduced, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X_reduced, labels)),
        "min_cluster_size": int(cluster_sizes.min()),
        "max_cluster_size": int(cluster_sizes.max()),
        "mean_cluster_size": float(cluster_sizes.mean()),
    })

comparison_df = pd.DataFrame(comparison_rows)
print("\n=== INTERNAL SUMMARY (k = 7, 16, 17) ===")
print(comparison_df.to_string(index=False))

# -------------------------
# 6) TRAIN FINAL KMEANS (k=K_FINAL) + EXTRA INTERNAL VISUALS
# -------------------------
kmeans_final = KMeans(n_clusters=K_FINAL, random_state=RANDOM_STATE, n_init="auto")
final_labels = kmeans_final.fit_predict(X_reduced)

sil_final = float(silhouette_score(X_reduced, final_labels, sample_size=sil_sample_size, random_state=RANDOM_STATE))
db_final = float(davies_bouldin_score(X_reduced, final_labels))
ch_final = float(calinski_harabasz_score(X_reduced, final_labels))
inertia_final = float(kmeans_final.inertia_)

print(f"\n=== FINAL KMEANS INTERNAL (k={K_FINAL}) ===")
print("Inertia:", inertia_final)
print("Silhouette (sample):", sil_final)
print("Davies–Bouldin:", db_final)
print("Calinski–Harabasz:", ch_final)

# Cluster sizes
sizes = pd.Series(final_labels).value_counts().sort_index()
print("\nCluster sizes:")
print(sizes.to_string())

# -------------------------
# 7) Silhouette plot (sampled)
# -------------------------
rng = np.random.default_rng(RANDOM_STATE)
n_for_plot = min(SIL_PLOT_SAMPLE_N, X_reduced.shape[0])
idx = rng.choice(X_reduced.shape[0], size=n_for_plot, replace=False)

sil_vals = silhouette_samples(X_reduced[idx], final_labels[idx])
sample_clusters = final_labels[idx]

plt.figure(figsize=(9, 6))
y_lower = 10
for c in np.unique(sample_clusters):
    c_vals = sil_vals[sample_clusters == c]
    c_vals.sort()
    y_upper = y_lower + len(c_vals)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, c_vals, alpha=0.6)
    plt.text(-0.05, (y_lower + y_upper) / 2, str(c))
    y_lower = y_upper + 10

plt.axvline(np.mean(sil_vals), color="red", linestyle="--", label="mean silhouette")
plt.title(f"Silhouette plot (sample) - KMeans k={K_FINAL}")
plt.xlabel("silhouette value")
plt.ylabel("samples (stacked by cluster)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"internal_silhouette_plot_k{K_FINAL}.png"), dpi=160)
plt.show()

# -------------------------
# 8) t-SNE visualization (sampled)
# -------------------------
n_tsne = min(TSNE_SAMPLE_N, X_reduced.shape[0])
idx2 = rng.choice(X_reduced.shape[0], size=n_tsne, replace=False)

tsne = TSNE(
    n_components=2,
    random_state=RANDOM_STATE,
    init="pca",
    learning_rate="auto",
    perplexity=30,
    max_iter=1000
)
X_tsne = tsne.fit_transform(X_reduced[idx2])

plt.figure(figsize=(11, 7))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=final_labels[idx2], cmap="tab20", s=10, alpha=0.7)
plt.colorbar(label="Cluster")
plt.title(f"t-SNE (sample) colored by KMeans cluster (k={K_FINAL})")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"internal_tsne_kmeans_k{K_FINAL}.png"), dpi=160)
plt.show()

print("\n" + "=" * 60)
print("INTERNAL EVALUATION COMPLETE")
print("=" * 60)
print(f"Figures saved in: {OUTPUT_DIR}/")
print("Next step: External evaluation (ARI, NMI vs BUILDINGID/FLOOR)")