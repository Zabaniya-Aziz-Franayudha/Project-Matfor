import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# 1. MEMBACA DATA
# Pastikan file nilaimahasiswa.csv ada di folder yang sama
df = pd.read_csv('Datamahasiswa.csv', sep=';', engine='python')
data = df[['Matfor', 'Strukdat', 'Inggris', 'Litik', 'Pancasila', 'StraPem']].values
nama = df['Nama'].values

centroids = np.array([
    [1.0, 5.0, 1.0, 3.0, 1.0, 2.0], # C1: Moh Raditya Ridwansyah
    [4.0, 3.0, 3.0, 3.0, 3.0, 3.0], # C2: Anas
    [3.0, 3.0, 5.0, 3.0, 5.0, 5.0]  # C3: Azka Ramadhan
])
iterasi = 1
max_iter = 100

print("--- ISI SELURUH DATA MAHASISWA ---")
print(df)

print("\n--- ANALISIS K-MEANS LENGKAP ---")
while iterasi < max_iter:
    # A. Hitung jarak  
    jarak = cdist(data, centroids, metric='euclidean')
    labels = np.argmin(jarak, axis=1)

    # B. Tampilkan Output
    print(f"--- ITERASI {iterasi} ---")

    # 1. Posisi centroid saat ini
    print("\n[POSISI CENTROID]")
    df_centroid = pd.DataFrame(
        centroids,
        index=['Centeroid 1', 'Centeroid 2', 'Centeroid 3'],
        columns=['Matfor', 'Strukdat', 'Inggris',
                 'Litik', 'Pancasila', 'StraPem']
    )
    print(df_centroid.round(3).to_string())


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


    print("\n[CENTROID BARU]")
    df_new_centroid = pd.DataFrame(
        new_centroids,
        index=['Cluster 1', 'Cluster 2', 'Cluster 3'],
        columns=['Matfor', 'Strukdat', 'Inggris',
                 'Litik', 'Pancasila', 'StraPem']
    )
    print(df_new_centroid.round(3).to_string())


    if np.allclose(centroids, new_centroids):
        print(f"\n=> KONVERGEN pada iterasi ke-{iterasi}.")
        break

    centroids = new_centroids
    iterasi += 1

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


print("\n--- HASIL AKHIR CLUSTER ---")

print(
    df_detail[
        ['No', 'Nama', 'Cluster', 'Kategori']
    ].to_string(index=False)
)

print(f"\nTotal iterasi: {iterasi}")

print("\n[REKAP JUMLAH MAHASISWA]")
rekap_akhir = df_detail['Cluster'].value_counts().reindex(
    ['Cluster 1', 'Cluster 2', 'Cluster 3'],
    fill_value=0
)
print(rekap_akhir.to_string())

print("\n--- PROSES SELESAI ---")