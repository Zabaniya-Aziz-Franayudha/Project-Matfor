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

iterasi = 0
max_iter = 100

print("--- ISI SELURUH DATA MAHASISWA ---")
print(df)

print("--- ANALISIS K-MEANS LENGKAP ---")
while iterasi < max_iter:
    # A. Hitung jarak Euclidean
    jarak = cdist(data, centroids, metric='euclidean')
    labels = np.argmin(jarak, axis=1)

    # B. Tampilkan Output
    print(f"\n{'='*80}")
    print(f"--- ITERASI {iterasi} ---")

    # 1. Posisi centroid saat ini
    print("\n[POSISI CENTROID]")
    df_centroid = pd.DataFrame(
        centroids,
        index=['Mudah', 'Sedang', 'Sulit'],
        columns=['Matfor', 'Strukdat', 'Inggris',
                 'Litik', 'Pancasila', 'StraPem']
    )
    print(df_centroid.round(3).to_string())

    # 2. Detail jarak mahasiswa
    df_detail = pd.DataFrame(
        jarak,
        columns=['Jarak_Mudah(1)',
                 'Jarak_Sedang(3)',
                 'Jarak_Sulit(5)']
    )

    df_detail.insert(0, 'Nama', nama)

    df_detail['Cluster'] = pd.Series(labels).replace({
        0: 'Mudah',
        1: 'Sedang',
        2: 'Sulit'
    })

    # 3. Rekap jumlah mahasiswa
    rekap = df_detail['Cluster'].value_counts().reindex(
        ['Mudah', 'Sedang', 'Sulit'],
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
        index=['Mudah', 'Sedang', 'Sulit'],
        columns=['Matfor', 'Strukdat', 'Inggris',
                 'Litik', 'Pancasila', 'StraPem']
    )
    print(df_new_centroid.round(3).to_string())

    print(f"{'='*80}")

    # E. Cek konvergensi
    if np.allclose(centroids, new_centroids):
        print(f"\n=> KONVERGEN pada iterasi ke-{iterasi}.")
        break

    # F. Update centroid
    centroids = new_centroids
    iterasi += 1
print("\n--- PROSES SELESAI ---")

print("\n--- HASIL AKHIR CLUSTER ---")
print(df_detail[['No', 'Nama', 'Cluster']].to_string(index=False))

print(f"\nTotal iterasi: {iterasi}")

print("\n--- PROSES SELESAI ---")