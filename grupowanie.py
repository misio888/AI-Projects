import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from mpl_toolkits.mplot3d import Axes3D

# Wczytanie danych
iris = datasets.load_iris()
X = iris.data
feature_names = iris.feature_names

# Skalowanie
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

all_results = []

# ========= KMeans ===========
print("%%%%%%  K-MEANS: RÓŻNE K  %%%%%%\n")
fig2d, axes2d = plt.subplots(2, 3, figsize=(15, 8))
fig3d = plt.figure(figsize=(15, 10))
kmeans_results = []

for idx, k in enumerate(range(2, 7)):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    kmeans_results.append((k, sil, db, ch))
    all_results.append(('KMeans', f'k={k}', sil, db, ch))

    print(f"k={k}: Silhouette={sil:.4f}, DB={db:.4f}, CH={ch:.4f}")

    ax2d = axes2d[idx // 3, idx % 3]
    ax2d.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='Set1', alpha=0.7)
    ax2d.set_title(f'K-Means (k={k})')
    ax2d.set_xlabel(feature_names[0])
    ax2d.set_ylabel(feature_names[1])

    ax3d = fig3d.add_subplot(2, 3, idx + 1, projection='3d')
    ax3d.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=labels, cmap='Set1', alpha=0.7)
    ax3d.set_title(f'3D Clustering (k={k})')
    ax3d.set_xlabel(feature_names[0])
    ax3d.set_ylabel(feature_names[1])
    ax3d.set_zlabel(feature_names[2])

plt.tight_layout()
plt.show()

# Metryki KMeans
k_vals, sil_scores, db_scores, ch_scores = zip(*kmeans_results)
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(k_vals, sil_scores, marker='o')
for i, score in zip(k_vals, sil_scores):
    plt.text(i, score, f"{score:.2f}", ha='center', va='bottom')
plt.title('Silhouette Score (KMeans)')
plt.xlabel('Liczba klastrów (k)')

plt.subplot(1, 3, 2)
plt.plot(k_vals, db_scores, marker='o', color='orange')
for i, score in zip(k_vals, db_scores):
    plt.text(i, score, f"{score:.2f}", ha='center', va='top')
plt.title('Davies-Bouldin Index (KMeans)')
plt.xlabel('Liczba klastrów (k)')

plt.subplot(1, 3, 3)
plt.plot(k_vals, ch_scores, marker='o', color='green')
for i, score in zip(k_vals, ch_scores):
    plt.text(i, score, f"{score:.2f}", ha='center', va='bottom')
plt.title('Calinski-Harabasz Index (KMeans)')
plt.xlabel('Liczba klastrów (k)')
plt.tight_layout()
plt.show()

# ========= DBSCAN ===========
print("\n%%%%%%  DBSCAN: eps  %%%%%%\n")
eps_values = [1.4, 0.5, 0.46, 0.42, 0.35]
dbscan_results = []
fig3d = plt.figure(figsize=(15, 8))
fig2d, axes2d = plt.subplots(2, 3, figsize=(15, 8))

for idx, eps in enumerate(eps_values):
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    ax3d = fig3d.add_subplot(2, 3, idx + 1, projection='3d')
    ax3d.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=labels, cmap='Set2', alpha=0.7)
    ax3d.set_title(f'DBSCAN (eps={eps})')

    ax2d = axes2d[idx // 3, idx % 3]
    ax2d.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='Set2', alpha=0.7)
    ax2d.set_title(f'DBSCAN (eps={eps})')
    ax2d.set_xlabel(feature_names[0])
    ax2d.set_ylabel(feature_names[1])

    if n_clusters > 1:
        sil = silhouette_score(X_scaled, labels)
        dbi = davies_bouldin_score(X_scaled, labels)
        chi = calinski_harabasz_score(X_scaled, labels)
        dbscan_results.append((eps, sil, dbi, chi))
        all_results.append(('DBSCAN', f'eps={eps}', sil, dbi, chi))
        print(f"eps={eps}: Silhouette={sil:.4f}, DB={dbi:.4f}, CH={chi:.4f}")
    else:
        dbscan_results.append((eps, None, None, None))
        all_results.append(('DBSCAN', f'eps={eps}', None, None, None))
        print(f"eps={eps}: Niewystarczająca liczba klastrów")

plt.tight_layout()
plt.show()
# Wykresy metryk DBSCAN
valid_dbscan = [res for res in dbscan_results if res[1] is not None]
if valid_dbscan:
    eps_vals, sil_scores, db_scores, ch_scores = zip(*valid_dbscan)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(eps_vals, sil_scores, marker='o')
    for x, y in zip(eps_vals, sil_scores):
        axes[0].text(x, y, f"{y:.2f}", ha='center', va='bottom')
    axes[0].set_title('Silhouette Score (DBSCAN)')
    axes[0].set_xlabel('eps')
    axes[1].plot(eps_vals, db_scores, marker='o', color='orange')
    for x, y in zip(eps_vals, db_scores):
        axes[1].text(x, y, f"{y:.2f}", ha='center', va='top')
    axes[1].set_title('Davies-Bouldin Index (DBSCAN)')
    axes[1].set_xlabel('eps')
    axes[2].plot(eps_vals, ch_scores, marker='o', color='green')
    for x, y in zip(eps_vals, ch_scores):
        axes[2].text(x, y, f"{y:.2f}", ha='center', va='bottom')
    axes[2].set_title('Calinski-Harabasz Index (DBSCAN)')
    axes[2].set_xlabel('eps')

    plt.tight_layout()
    plt.show()

# ========= Agglomerative ===========
print("\n%%%%%%  AGGLOMERATIVE CLUSTERING  %%%%%%\n")
agglo_results = []
fig3d = plt.figure(figsize=(15, 8))
fig2d, axes2d = plt.subplots(2, 3, figsize=(15, 8))

for idx, k in enumerate([2, 3, 4, 5, 6]):
    agg = AgglomerativeClustering(n_clusters=k)
    labels = agg.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)
    chi = calinski_harabasz_score(X_scaled, labels)
    agglo_results.append((k, sil, dbi, chi))
    all_results.append(('Agglomerative', f'n_clusters={k}', sil, dbi, chi))

    print(f"k={k}: Silhouette={sil:.4f}, DB={dbi:.4f}, CH={chi:.4f}")

    ax3d = fig3d.add_subplot(2, 3, idx + 1, projection='3d')
    ax3d.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=labels, cmap='Accent', alpha=0.7)
    ax3d.set_title(f'Agglomerative (k={k})')

    ax2d = axes2d[idx // 3, idx % 3]
    ax2d.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='Accent', alpha=0.7)
    ax2d.set_title(f'Agglomerative (k={k})')
    ax2d.set_xlabel(feature_names[0])
    ax2d.set_ylabel(feature_names[1])

plt.tight_layout()
plt.show()

# Wykresy metryk Agglomerative
k_vals, sil_scores, db_scores, ch_scores = zip(*agglo_results)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(k_vals, sil_scores, marker='o')
for x, y in zip(k_vals, sil_scores):
    axes[0].text(x, y, f"{y:.2f}", ha='center', va='bottom')
axes[0].set_title('Silhouette Score (Agglomerative)')
axes[0].set_xlabel('Liczba klastrów (k)')
axes[1].plot(k_vals, db_scores, marker='o', color='orange')
for x, y in zip(k_vals, db_scores):
    axes[1].text(x, y, f"{y:.2f}", ha='center', va='top')
axes[1].set_title('Davies-Bouldin Index (Agglomerative)')
axes[1].set_xlabel('Liczba klastrów (k)')
axes[2].plot(k_vals, ch_scores, marker='o', color='green')
for x, y in zip(k_vals, ch_scores):
    axes[2].text(x, y, f"{y:.2f}", ha='center', va='bottom')
axes[2].set_title('Calinski-Harabasz Index (Agglomerative)')
axes[2].set_xlabel('Liczba klastrów (k)')

plt.tight_layout()
plt.show()

# ========= ZAPIS DO PLIKU ===========
with open('clustering_stats.txt', 'w') as f:
    f.write("Wyniki metryk dla różnych modeli i parametrów\n")
    f.write("-------------------------------------------------------------\n")
    f.write(f"{'Model':<15}{'Parametr':<15}{'Silhouette':<12}{'Davies-Bouldin':<17}{'Calinski-Harabasz'}\n")

    for model, param, sil, db, ch in all_results:
        sil_str = f"{sil:.4f}" if sil is not None else "N/A"
        db_str = f"{db:.4f}" if db is not None else "N/A"
        ch_str = f"{ch:.4f}" if ch is not None else "N/A"
        f.write(f"{model:<15}{param:<15}{sil_str:<12}{db_str:<17}{ch_str}\n")

print("\nWszystkie statystyki zapisane do pliku 'clustering_stats.txt'")
