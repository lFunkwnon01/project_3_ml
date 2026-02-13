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
# 10. Elbow Method
# =============================

inertia_values = []
k_range = range(2, 21)

for k in k_range:

    kmeans_tmp = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans_tmp.fit(X_pca)

    inertia_values.append(kmeans_tmp.inertia_)


plt.figure(figsize=(8, 5))

plt.plot(k_range, inertia_values, marker="o")

plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid()

plt.savefig(os.path.join(PLOTS_DIR, f"new_elbow_{timestamp}.png"), dpi=300)
plt.close()


# =============================
# 11. Silhouette Method
# =============================

sil_scores = []

for k in k_range:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels_tmp = model.fit_predict(X_pca)

    score = silhouette_score(X_pca, labels_tmp)

    sil_scores.append(score)


plt.figure(figsize=(8, 5))

plt.plot(k_range, sil_scores, marker="o")

plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")
plt.grid()

plt.savefig(os.path.join(PLOTS_DIR, f"new_silhouette_{timestamp}.png"), dpi=300)
plt.close()


# =============================
# 12. Final KMeans
# =============================

k_clusters = 9

kmeans = KMeans(
    n_clusters=k_clusters,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(X_pca)
df = df.copy()
df["Cluster_ID"] = labels

print("Points per cluster:")
print(df["Cluster_ID"].value_counts().sort_index())


sil = silhouette_score(X_pca, labels)

print(f"Silhouette para k-means: {sil:.4f}")


# =============================
# 13. PCA 2D Visualization
# =============================

pca_2d = PCA(n_components=2, random_state=42)
X_2d = X_pca[:, :2]  # usando solo las 2 primeras componentes de PCA final
# X_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=labels,
    s=8,
    cmap="tab10"
)

plt.title("PCA K-D Clustering")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.savefig(os.path.join(PLOTS_DIR, f"new_pca2d_{timestamp}.png"), dpi=300)
plt.close()


# =============================
# 14. t-SNE Visualization
# =============================

tsne = TSNE(
    n_components=2,
    random_state=42,
    init="pca",
    learning_rate=200,
    perplexity=30
)

X_tsne = tsne.fit_transform(X_pca)

plt.figure(figsize=(10, 7))

for i in range(k_clusters):

    plt.scatter(
        X_tsne[labels == i, 0],
        X_tsne[labels == i, 1],
        s=12,
        alpha=0.7,
        label=f"Cluster {i}"
    )

plt.legend(markerscale=2)

plt.title("t-SNE Visualization of Clusters")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")

plt.savefig(os.path.join(PLOTS_DIR, f"new_tsne_{timestamp}.png"), dpi=300)
plt.close()

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

# Evaluacion de DBSCAN con Silhouette Score
# Filtramos el ruido (-1) porque silhouette_score no lo admite
mask_clean = db_labels != -1
X_clean = X_pca[mask_clean]
labels_clean = db_labels[mask_clean]
unique_clusters = np.unique(labels_clean)

n_points = len(X_clean)
print(f"Puntos totales (original): {len(X_pca)}")
print(f"Puntos a evaluar (sin ruido): {n_points}")
print(f"Clusters encontrados: {unique_clusters}")

if len(unique_clusters) < 2:
    print("ERROR: No se puede calcular Silhouette. Se necesitan al menos 2 clusters formados (además del ruido).")
    print("Ajusta EPS o Beta.")
else:
    print("Calculando Matriz de Distancias (esto puede tomar unos segundos/minutos)...")
    start_time = time.time()
    
    # 2. Matriz de Distancias (Paso pesado pero necesario para hacerlo manual con 19k datos)
    # D[i, j] es la distancia entre el punto i y el punto j
    # IMPORTANTE: Esto usará aprox 2.8 GB de RAM para 19k puntos. Si falla por memoria, avísame.
    D = cdist(X_clean, X_clean, metric='euclidean')
    
    print(f"Matriz calculada en {time.time() - start_time:.2f} segundos.")

    # Arrays para guardar a(i) y b(i) de cada punto
    A = np.zeros(n_points)
    B = np.full(n_points, np.inf)

    print("Calculando a(i) y b(i) cluster por cluster...")

    # 3. Iteramos por CLUSTERS (matemáticamente equivalente a iterar por puntos, pero optimizado)
    for cluster_id in unique_clusters:
        # Máscara booleana para los puntos que pertenecen al cluster actual 'cluster_id'
        in_cluster_mask = (labels_clean == cluster_id)
        
        # --- CÁLCULO DE a(i) ---
        # Fórmula: Promedio de distancia a puntos del MISMO cluster
        # Extraemos submatriz de distancias solo entre miembros del cluster
        dists_in = D[np.ix_(in_cluster_mask, in_cluster_mask)]
        
        n_members = len(dists_in)
        if n_members > 1:
            # Sumamos fila y dividimos entre (n-1) para no contarse a sí mismo (distancia 0)
            A[in_cluster_mask] = np.sum(dists_in, axis=1) / (n_members - 1)
        else:
            A[in_cluster_mask] = 0.0

        # --- CÁLCULO DE b(i) ---
        # Fórmula: Mínimo promedio de distancia a puntos de OTROS clusters
        for other_cluster_id in unique_clusters:
            if cluster_id == other_cluster_id:
                continue
            
            # Máscara de puntos del otro cluster
            other_cluster_mask = (labels_clean == other_cluster_id)
            
            # Submatriz: Filas = cluster actual, Columnas = otro cluster
            dists_out = D[np.ix_(in_cluster_mask, other_cluster_mask)]
            
            # Promedio de distancias hacia ese otro cluster específico
            mean_dists_out = np.mean(dists_out, axis=1)
            
            # Nos quedamos con el mínimo encontrado hasta ahora para cada punto
            B[in_cluster_mask] = np.minimum(B[in_cluster_mask], mean_dists_out)

    # 4. Cálculo final de s(i)
    # s(i) = (b(i) - a(i)) / max(a(i), b(i))
    denom = np.maximum(A, B)
    
    # Manejo de división por cero (si a=0 y b=0, s=0)
    sil_values = np.zeros(n_points)
    valid_denom = denom > 0
    sil_values[valid_denom] = (B[valid_denom] - A[valid_denom]) / denom[valid_denom]

    final_score = np.mean(sil_values)
    
    print("-" * 30)
    print(f"Ev Interna con Silhouette Score Promedio: {final_score:.4f}")
    print("-" * 30)

    # 5. Guardar Histograma (Evidencia visual del cálculo punto a punto)
    plt.figure(figsize=(10, 6))
    plt.hist(sil_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=final_score, color='red', linestyle='--', linewidth=2, label=f'Promedio: {final_score:.2f}')
    plt.title(f"Distribución de Silhouette Score\nTotal Puntos: {n_points}")
    plt.xlabel("Valor de Silhouette s(i)")
    plt.ylabel("Cantidad de Puntos")
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    plt.savefig(os.path.join(PLOTS_DIR, f"manual_silhouette_distribution_{timestamp}.png"), dpi=300)
    print("Gráfico de distribución guardado.")
    plt.close()

# ==========================================
# 17. CÁLCULO DEL DAVIES-BOULDIN INDEX (DBI)
# ==========================================
# Aprovechamos que ya tenemos X_clean y labels_clean definidos arriba
from sklearn.metrics import davies_bouldin_score

print("\n" + "="*50)
print("INICIANDO CÁLCULO DE DAVIES-BOULDIN INDEX")
print("="*50)

if len(unique_clusters) < 2:
    print("ERROR: Se necesitan al menos 2 clusters para calcular DBI.")
else:
    # --- PASO 1: Calcular Centroides ---
    centroids = []
    for k in unique_clusters:
        cluster_points = X_clean[labels_clean == k]
        centroid = cluster_points.mean(axis=0)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    
    # --- PASO 2: Calcular Dispersión Intra-Cluster (s_i) ---
    s = []
    for i, k in enumerate(unique_clusters):
        cluster_points = X_clean[labels_clean == k]
        # Distancia de cada punto a su centroide
        dists = cdist(cluster_points, [centroids[i]], metric='euclidean')
        mean_dist = np.mean(dists)
        s.append(mean_dist)
    
    s = np.array(s)
    
    # --- PASO 3: Calcular Separación Inter-Cluster (d_ij) ---
    centroid_dists = cdist(centroids, centroids, metric='euclidean')
    
    # --- PASO 4: Calcular Ratio R_ij ---
    R = np.zeros((len(unique_clusters), len(unique_clusters)))
    
    for i in range(len(unique_clusters)):
        for j in range(len(unique_clusters)):
            if i != j:
                R[i, j] = (s[i] + s[j]) / centroid_dists[i, j]

    # --- PASO 5: Calcular DBI Final ---
    max_R = np.max(R, axis=1)
    dbi_manual = np.mean(max_R)

    print(f"Davies-Bouldin Index (MANUAL):  {dbi_manual:.5f}")
    
    # Verificación con Sklearn
    dbi_sklearn = davies_bouldin_score(X_clean, labels_clean)
    print(f"Davies-Bouldin Index (SKLEARN): {dbi_sklearn:.5f}")
    
    print("-" * 30)
    if dbi_manual < 1.0:
        print("INTERPRETACIÓN: Excelente separación (Clusters compactos).")
    elif dbi_manual < 1.5:
        print("INTERPRETACIÓN: Buena separación.")
    else:
        print("INTERPRETACIÓN: Separación moderada o solapada.")


# ==========================================
# 18. CÁLCULO DEL DUNN INDEX (MANUAL)
# ==========================================
print("\n" + "="*50)
print("INICIANDO CÁLCULO DE DUNN INDEX")
print("="*50)

if len(unique_clusters) < 2:
    print("ERROR: Se necesitan al menos 2 clusters para calcular Dunn Index.")
else:
    start_time_dunn = time.time()
    
    # --- PASO 1: Calcular Diámetros (Denominador) ---
    # Diámetro = Máxima distancia intra-cluster
    max_diameters = []
    
    for k in unique_clusters:
        cluster_points = X_clean[labels_clean == k]
        if len(cluster_points) < 2:
            max_diameters.append(0.0)
            continue
        # cdist intra-cluster
        dists = cdist(cluster_points, cluster_points, metric='euclidean')
        max_diameters.append(np.max(dists))
    
    max_intra_cluster_dist = np.max(max_diameters)

    # --- PASO 2: Calcular Distancias Inter-Cluster (Numerador) ---
    # Minima distancia entre cualquier punto de C_i y C_j
    min_inter_cluster_dist = np.inf
    
    n_clus = len(unique_clusters)
    for i in range(n_clus):
        for j in range(i + 1, n_clus):
            c1 = unique_clusters[i]
            c2 = unique_clusters[j]
            
            points_c1 = X_clean[labels_clean == c1]
            points_c2 = X_clean[labels_clean == c2]
            
            # Distancia entre grupos
            dists_between = cdist(points_c1, points_c2, metric='euclidean')
            current_min = np.min(dists_between)
            
            if current_min < min_inter_cluster_dist:
                min_inter_cluster_dist = current_min

    # --- PASO 3: Calcular Dunn Index ---
    if max_intra_cluster_dist > 0:
        dunn_index = min_inter_cluster_dist / max_intra_cluster_dist
    else:
        dunn_index = 0.0

    print(f"Dunn Index Final: {dunn_index:.5f}")
    print(f"Tiempo cálculo Dunn: {time.time() - start_time_dunn:.2f} s")
    print("-" * 30)
    print("INTERPRETACIÓN: Cuanto más alto, mejor (más separación vs compactación).")
