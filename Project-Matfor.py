import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# 1. MEMBACA DATA
# Pastikan file nilaimahasiswa.csv ada di folder yang sama
df = pd.read_csv('nilaimahasiswa.csv', sep=';', engine='python')
data = df[['Matfor', 'Strukdat', 'Inggris', 'Litik', 'Pancasila', 'StraPem']].values
nama = df['Nama'].values

# 2. CENTROID AWAL (1, 3, 5)
centroids = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
                      [3.0, 3.0, 3.0, 3.0, 3.0, 3.0], 
                      [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]])

iterasi = 1
max_iter = 100

print("--- ISI SELURUH DATA MAHASISWA ---")
print(df)


print("\n--- ANALISIS K-MEANS LENGKAP ---")
while iterasi < max_iter:
    # A. Hitung jarak Euclidean
    jarak = cdist(data, centroids, metric='euclidean')
    labels = np.argmin(jarak, axis=1)

    # B. Tampilkan Output
    print(f"--- ITERASI {iterasi} ---")

    # 1. Posisi centroid saat ini
    print("\n[POSISI CENTROID]")
    df_centroid = pd.DataFrame(
        centroids,
        index=['Cluster 1', 'Cluster 2', 'Cluster 3'],
        columns=['Matfor', 'Strukdat', 'Inggris',
                 'Litik', 'Pancasila', 'StraPem']
    )
    print(df_centroid.round(3).to_string())

    # 2. Detail jarak mahasiswa
    df_detail = pd.DataFrame(
        jarak,
        columns=['Cluster 1',
                 'Cluster 2',
                 'Cluster 3']
    )

    df_detail.insert(0, 'Nama', nama)

    df_detail['Cluster'] = pd.Series(labels).replace({
        0: 'Cluster 1',
        1: 'Cluster 2',
        2: 'Cluster 3'
    })

    # 3. Rekap jumlah mahasiswa
    rekap = df_detail['Cluster'].value_counts().reindex(
        ['Cluster 1', 'Cluster 2', 'Cluster 3'],
        fill_value=0
    )

    print("\n[REKAP JUMLAH MAHASISWA]")
    print(rekap.to_string())

    # 4. Detail lengkap
    print("\n[DETAIL JARAK & KLASTER MAHASISWA]")
    df_detail.insert(0, 'No', range(1, len(df_detail)+1))
    print(df_detail.round(2).to_string(index=False))

    # C. Hitung centroid baru
    new_centroids = np.array([
        data[labels == i].mean(axis=0)
        if len(data[labels == i]) > 0
        else centroids[i]
        for i in range(len(centroids))
    ])

    # D. Tampilkan centroid baru
    print("\n[CENTROID BARU]")
    df_new_centroid = pd.DataFrame(
        new_centroids,
        index=['Cluster 1', 'Cluster 2', 'Cluster 3'],
        columns=['Matfor', 'Strukdat', 'Inggris',
                 'Litik', 'Pancasila', 'StraPem']
    )
    print(df_new_centroid.round(3).to_string())

    # E. Cek konvergensi
    if np.allclose(centroids, new_centroids):
        print(f"\n=> KONVERGEN pada iterasi ke-{iterasi}.")
        break

    # F. Update centroid
    centroids = new_centroids
    iterasi += 1

# Cari rata-rata tiap centroid
rata_centroid = centroids.mean(axis=1)

# Urutkan centroid dari kecil ke besar
urutan = np.argsort(rata_centroid)

# Mapping nama cluster
mapping_nama = {
    urutan[0]: 'Mudah',
    urutan[1]: 'Sedang',
    urutan[2]: 'Sulit'
}

# Tambahkan kategori
df_detail['Kategori'] = [
    mapping_nama[label]
    for label in labels
]

# =========================
# 5. HASIL AKHIR
# =========================
print("\n--- HASIL AKHIR CLUSTER ---")

print(
    df_detail[
        ['No', 'Nama', 'Cluster', 'Kategori']
    ].to_string(index=False)
)

print(f"\nTotal iterasi: {iterasi}")

# Tambahkan ini di bawah "5. HASIL AKHIR"
print("\n[REKAP JUMLAH MAHASISWA]")
rekap_akhir = df_detail['Cluster'].value_counts().reindex(
    ['Cluster 1', 'Cluster 2', 'Cluster 3'],
    fill_value=0
)
print(rekap_akhir.to_string())

print("\n--- PROSES SELESAI ---")