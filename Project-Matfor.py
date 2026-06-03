import pandas as pd

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

# =========================
# 2. CENTROID AWAL
# =========================
centroids = [
    [1, 5, 1, 3, 1, 2],
    [4, 3, 3, 3, 3, 3],
    [3, 3, 5, 3, 5, 5]
]

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
    # HITUNG JARAK (MANUAL TANPA MATH)
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