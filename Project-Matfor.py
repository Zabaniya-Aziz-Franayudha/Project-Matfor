import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# =========================
# 1. DATA
# =========================
df = pd.read_csv("Datamahasiswa.csv", sep=';')

nama = df['Nama'].tolist()

data = []
for _, row in df.iterrows():
    data.append([
        row['Matfor'],
        row['Strukdat'],
        row['Inggris'],
        row['Litik'],
        row['Pancasila'],
        row['StraPem']
    ])


# TAMBAHAN PRAPROSES (Validasi & Pembersihan)
for i in range(len(data)):
    for k in range(len(data[i])):
        # Mengatasi data jika ada yang kosong (isi dengan 3)
        if pd.isna(data[i][k]):
            data[i][k] = 3
        # Memastikan data di range 1-5
        if data[i][k] < 1: data[i][k] = 1
        if data[i][k] > 5: data[i][k] = 5
print("Data berhasil divalidasi ke rentang 1-5.")

# =========================
# 2. CENTROID AWAL
# =========================
centroids = [
    [1, 5, 1, 3, 1, 2],
    [4, 3, 3, 3, 3, 3],
    [3, 3, 5, 3, 5, 5]
]

kluster=3
iterasi = 1
max_iter = 100

# =========================
# 3. LOOP K-MEANS
# =========================
while iterasi <= max_iter:

    print("\n" + "="*90)
    print(f"ITERASI {iterasi}")
    print("="*90)

    # =========================
    # HITUNG JARAK
    # =========================
    jarak = []

    for i in range(len(data)):
        row = []
        for c in centroids:
            total = 0
            for k in range(6):
                selisih = data[i][k] - c[k]
                total += selisih * selisih   # kuadrat
            jarak_akhir = total ** 0.5       # akar manual
            row.append(jarak_akhir)
        jarak.append(row)

    # =========================
    # CLUSTER
    # =========================
    labels = []
    for i in range(len(jarak)):
        min_val = jarak[i][0]
        cluster = 0
        for j in range(1, 3):
            if jarak[i][j] < min_val:
                min_val = jarak[i][j]
                cluster = j
        labels.append(cluster)

    # =========================
    # CENTROID BARU
    # =========================
    new_centroids = []

    for c in range(3):
        anggota = []
        for i in range(len(data)):
            if labels[i] == c:
                anggota.append(data[i])
        if len(anggota) == 0:
            new_centroids.append(centroids[c])
            continue
        centroid_baru = []
        for k in range(6):
            total = 0
            for a in anggota:
                total += a[k]
            centroid_baru.append(total / len(anggota))
        new_centroids.append(centroid_baru)

    # =========================
    # OUTPUT CENTROID
    # =========================
    print("\n[POSISI CENTROID]")
    print("-"*90)
    for i in range(3):
        print(f"C{i+1} : ", end="")
        for v in centroids[i]:
            print(f"{v:7.2f}", end=" ")
        print()

    # =========================
    # DETAIL JARAK
    # =========================
    print("\n[DETAIL JARAK & CLUSTER]")
    print("-"*90)
    print(f"{'No':<4}{'Nama':<28}{'C1':>10}{'C2':>10}{'C3':>10}{'Cluster':>10}")
    print("-"*90)

    for i in range(len(data)):
        print(
            f"{i+1:<4}"
            f"{nama[i][:28]:<28}"
            f"{jarak[i][0]:>10.2f}"
            f"{jarak[i][1]:>10.2f}"
            f"{jarak[i][2]:>10.2f}"
            f"{'C'+str(labels[i]+1):>10}"
        )

    # =========================
    # REKAP
    # =========================
    count = [0, 0, 0]
    for l in labels:
        count[l] += 1
    print("\n[REKAP CLUSTER]")
    print("-"*90)
    for i in range(3):
        print(f"Cluster {i+1} : {count[i]} mahasiswa")

    # =========================
    # CENTROID BARU
    # =========================
    print("\n[CENTROID BARU]")
    print("-"*90)
    for i in range(3):
        print(f"C{i+1} : ", end="")
        for v in new_centroids[i]:
            print(f"{v:7.2f}", end=" ")
        print()

    # =========================
    # KONVERGENSI
    # =========================
    sama = True
    for i in range(3):
        for j in range(6):
            if round(centroids[i][j], 4) != round(new_centroids[i][j], 4):
                sama = False
    if sama:
        print("\n=> KONVERGENSI TERCAPAI")
        break
    centroids = new_centroids
    iterasi += 1

# =========================
# 4. HASIL AKHIR
# =========================
print("\n" + "="*90)
print("HASIL AKHIR CLUSTERING")
print("="*90)
rata = []
for c in centroids:
    total = 0
    for v in c:
        total += v
    rata.append(total / len(c))
urutan = sorted(range(3), key=lambda i: rata[i])

mapping = {
    urutan[0]: "MUDAH",
    urutan[1]: "SEDANG",
    urutan[2]: "SUSAH"
}

print(f"{'No':<4}{'Nama':<28}{'Cluster':<10}{'Kategori'}")
print("-"*90)
for i in range(len(data)):
    print(
        f"{i+1:<4}"
        f"{nama[i][:27]:<28}"
        f"C{labels[i]+1:<10}"
        f"{mapping[labels[i]]}"
    )
print("-"*90)
print(f"Total iterasi: {iterasi}")

# ==================================================
# VISUALISASI BAR CHART (Distribusi Cluster)
# ==================================================
def show_bar_chart(labels, mapping):
    counts = [labels.count(i) for i in range(3)]
    kategori = [mapping[i] for i in range(3)]
    
    plt.figure(figsize=(8, 5))
    plt.bar(kategori, counts, color=['red', 'blue', 'green'])
    plt.title("Distribusi Jumlah Mahasiswa per Cluster")
    plt.xlabel("Kategori")
    plt.ylabel("Jumlah Mahasiswa")
    plt.show()
show_bar_chart(labels, mapping)

# ==================================================
# VISUALISASI PCA (Reduksi Dimensi)
# ==================================================

def show_pca_plot(data, labels, mapping):
    # Konversi data dan labels ke format yang dimengerti sklearn (numpy array)
    data_array = np.array(data)
    labels_array = np.array(labels)
    # PCA
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_array)
    # Visualisasi
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'green']
    for i in range(3):
        # Menggunakan boolean indexing pada numpy array
        plt.scatter(data_pca[labels_array == i, 0], 
                    data_pca[labels_array == i, 1], 
                    c=colors[i], label=mapping[i], s=80, alpha=0.7)
    
    plt.title("Visualisasi PCA (Reduksi 6 Dimensi ke 2D)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()
# Panggil fungsi setelah loop K-Means
show_pca_plot(data, labels, mapping)

# ==================================================
# FUNGSI: SILHOUETTE SCORE (TIAP & SELURUH)
# ==================================================
def hitung_jarak(p1, p2):
    return sum((a - b)**2 for a, b in zip(p1, p2))**0.5

def evaluasi_silhouette_manual(data, labels):
    n = len(data)
    sil_scores = []
    
    for i in range(n):
        cluster_i = labels[i]
        # Hitung a(i): jarak rata-rata ke anggota cluster sendiri
        anggota_i = [data[j] for j in range(n) if labels[j] == cluster_i]
        if len(anggota_i) > 1:
            a_i = sum(hitung_jarak(data[i], p) for p in anggota_i) / (len(anggota_i) - 1)
        else:
            a_i = 0
        # Hitung b(i): jarak rata-rata terpendek ke cluster lain
        cluster_lain = [c for c in range(3) if c != cluster_i]
        b_i = float('inf')
        for c in cluster_lain:
            anggota_c = [data[j] for j in range(n) if labels[j] == c]
            if len(anggota_c) > 0:
                dist = sum(hitung_jarak(data[i], p) for p in anggota_c) / len(anggota_c)
                if dist < b_i:
                    b_i = dist
        
        # Hitung s(i)
        if max(a_i, b_i) == 0:
            sil_scores.append(0)
        else:
            sil_scores.append((b_i - a_i) / max(a_i, b_i))
            
    return sil_scores

# ==================================================
# IMPLEMENTASI EVALUASI
# ==================================================
sil_list = evaluasi_silhouette_manual(data, labels)

# Mengelompokkan skor siluet berdasarkan cluster
cluster_groups = {0: [], 1: [], 2: []}
for i in range(len(labels)):
    cluster_groups[labels[i]].append(sil_list[i])
print("\n" + "="*50)
print("EVALUASI SILHOUETTE SCORE")
print("="*50)
# Print Siluet tiap cluster
for i in range(3):
    avg_c = sum(cluster_groups[i]) / len(cluster_groups[i]) if cluster_groups[i] else 0
    print(f"Cluster {i+1} ({mapping[i]}): {avg_c:.4f}")

# Print Siluet seluruh data
total_avg = sum(sil_list) / len(sil_list)
print("-" * 50)
print(f"Rata-rata Siluet Keseluruhan: {total_avg:.4f}")
print("="*50)

# =========================
# 1. DATA & PREPROCESSING
# =========================
df = pd.read_csv("Datamahasiswa.csv", sep=';')
nama = df['Nama'].tolist()
data = df.iloc[:, 1:].values.tolist()
for i in range(len(data)):
    for k in range(6):
        data[i][k] = 3 if pd.isna(data[i][k]) else max(1, min(5, data[i][k]))

# =========================
# 2. K-MEANS MANUAL
# =========================
centroids = [[1,5,1,3,1,2], [4,3,3,3,3,3], [3,3,5,3,5,5]]
for iterasi in range(1, 101):
    labels = [min(range(3), key=lambda i: sum((data[j][k]-centroids[i][k])**2 for k in range(6))) for j in range(len(data))]
    new_c = [[sum(col)/len(g) for col in zip(*g)] if (g := [data[j] for j, l in enumerate(labels) if l == i]) else centroids[i] for i in range(3)]
    if all(round(centroids[i][k], 4) == round(new_c[i][k], 4) for i in range(3) for k in range(6)): break
    centroids = new_c

# =========================
# PLOT DISTRIBUSI TIAP FITUR
# =========================
def plot_distribusi_per_fitur(data, labels):
    fitur = ['Matfor', 'Strukdat', 'Inggris', 'Litik', 'Pancasila', 'StraPem']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for k in range(6):
        for i in range(3):
            nilai = [data[j][k] for j in range(len(labels)) if labels[j] == i]
            axes[k].hist(nilai, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], alpha=0.5, label=f"C{i+1}")
        axes[k].set_title(fitur[k])
        axes[k].legend()
    plt.tight_layout()
    plt.savefig("distribusi_fitur.png", dpi=300)
    plt.show()

plot_distribusi_per_fitur(data, labels)
