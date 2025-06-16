
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Data
waktu_belajar = np.array([1, 2, 3, 4]).reshape(-1, 1)  # Waktu belajar (jam)
nilai_ujian = np.array([60, 65, 70, 75])  # Nilai ujian

# Membuat dan melatih model
model = LinearRegression()
model.fit(waktu_belajar, nilai_ujian)

# Mendapatkan koefisien
kemiringan = model.coef_[0]
intersep = model.intercept_

# Membuat prediksi
prediksi_nilai = model.predict(waktu_belajar)

# Menghitung metrik
mse = mean_squared_error(nilai_ujian, prediksi_nilai)
r2 = r2_score(nilai_ujian, prediksi_nilai)

# Prediksi untuk 3,5 jam belajar
prediksi_x = 3.5
prediksi_y = model.predict([[prediksi_x]])[0]

# Menampilkan hasil
print("Masalah 2: Analisis Waktu Belajar vs Nilai Ujian")
print("=" * 52)
print(f"Titik Data:")
print(f"Waktu Belajar (X): {waktu_belajar.flatten()}")
print(f"Nilai Ujian (Y): {nilai_ujian}")
print()
print(f"Persamaan Regresi: Y = {intersep:.2f} + {kemiringan:.2f}X")
print(f"Kemiringan (b): {kemiringan:.2f}")
print(f"Intersep (a): {intersep:.2f}")
print(f"R-squared: {r2:.4f}")
print(f"Mean Squared Error: {mse:.2f}")
print()
print(f"Peningkatan rata-rata per jam: {kemiringan:.2f} poin")
print(f"Prediksi untuk {prediksi_x} jam belajar: {prediksi_y:.2f} poin")

# Membuat visualisasi
plt.figure(figsize=(10, 6))

# Plot titik data
plt.scatter(waktu_belajar, nilai_ujian, color='blue', alpha=0.7, s=100,
           label='Titik Data', zorder=5)

# Plot garis regresi
garis_x = np.linspace(waktu_belajar.min(), waktu_belajar.max(), 100).reshape(-1, 1)
garis_y = model.predict(garis_x)
plt.plot(garis_x, garis_y, color='red', linewidth=2,
         label=f'Y = {intersep:.1f} + {kemiringan:.1f}X', zorder=3)

# Menambahkan titik prediksi
plt.scatter([prediksi_x], [prediksi_y], color='green', s=150,
           marker='*', label=f'Prediksi: {prediksi_x}j → {prediksi_y:.1f} poin', zorder=10)

# Menyesuaikan plot
plt.xlabel('Waktu Belajar (jam)', fontsize=12)
plt.ylabel('Nilai Ujian', fontsize=12)
plt.title('Masalah 2: Regresi Linear Waktu Belajar vs Nilai Ujian\n' +
          f'Peningkatan rata-rata: {kemiringan:.1f} poin per jam', fontsize=14, pad=20)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Menambahkan kotak teks dengan persamaan
teks_persamaan = f'Y = {kemiringan:.1f}X + {intersep:.1f}\nPeningkatan: {kemiringan:.1f} poin/jam\nR² = {r2:.4f}'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
plt.text(0.05, 0.95, teks_persamaan, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.show()

# Verifikasi perhitungan manual
print("\nVerifikasi Perhitungan Manual:")
print("-" * 40)
waktu_rata = np.mean(waktu_belajar.flatten())
nilai_rata = np.mean(nilai_ujian)

print(f"X̄ (rata-rata waktu belajar): {waktu_rata}")
print(f"Ȳ (rata-rata nilai ujian): {nilai_rata}")

# Menghitung kemiringan secara manual
pembilang = sum((waktu_belajar.flatten() - waktu_rata) * (nilai_ujian - nilai_rata))
penyebut = sum((waktu_belajar.flatten() - waktu_rata) ** 2)
kemiringan_manual = pembilang / penyebut

print(f"Kemiringan (manual): {kemiringan_manual}")
print(f"Intersep (manual): {nilai_rata - kemiringan_manual * waktu_rata}")

# Menampilkan perhitungan detail
print("\nPerhitungan Detail:")
for i in range(len(waktu_belajar.flatten())):
    xi_diff = waktu_belajar.flatten()[i] - waktu_rata
    yi_diff = nilai_ujian[i] - nilai_rata
    print(f"Titik {i+1}: ({waktu_belajar.flatten()[i]}j, {nilai_ujian[i]}) -> ({xi_diff:+.1f}) * ({yi_diff:+.1f}) = {xi_diff * yi_diff:+.2f}")

print(f"\nPembilang: {pembilang}")
print(f"Penyebut: {penyebut}")
print(f"Kemiringan = {pembilang}/{penyebut} = {kemiringan_manual}")
print(f"Peningkatan rata-rata nilai ujian per jam belajar: {kemiringan_manual:.1f} poin")
