
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Data
jumlah_pelanggan = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)  # Jumlah pelanggan
pendapatan = np.array([200, 400, 600, 800, 1000])  # Pendapatan (dalam ribuan)

# Membuat dan melatih model
model = LinearRegression()
model.fit(jumlah_pelanggan, pendapatan)

# Mendapatkan koefisien
kemiringan = model.coef_[0]
intersep = model.intercept_

# Membuat prediksi
prediksi_pendapatan = model.predict(jumlah_pelanggan)

# Menghitung metrik
mse = mean_squared_error(pendapatan, prediksi_pendapatan)
r2 = r2_score(pendapatan, prediksi_pendapatan)

# Prediksi untuk 35 pelanggan
prediksi_x = 35
prediksi_y = model.predict([[prediksi_x]])[0]

# Menampilkan hasil
print("Masalah 1: Analisis Pelanggan vs Pendapatan")
print("=" * 50)
print(f"Titik Data:")
print(f"Pelanggan (X): {jumlah_pelanggan.flatten()}")
print(f"Pendapatan (Y): {pendapatan}")
print()
print(f"Persamaan Regresi: Y = {intersep:.2f} + {kemiringan:.2f}X")
print(f"Kemiringan (b): {kemiringan:.2f}")
print(f"Intersep (a): {intersep:.2f}")
print(f"R-squared: {r2:.4f}")
print(f"Mean Squared Error: {mse:.2f}")
print()
print(f"Prediksi untuk {prediksi_x} pelanggan: {prediksi_y:.2f} ribu")

# Membuat visualisasi
plt.figure(figsize=(10, 6))

# Plot titik data
plt.scatter(jumlah_pelanggan, pendapatan, color='blue', alpha=0.7, s=100,
           label='Titik Data', zorder=5)

# Plot garis regresi
garis_x = np.linspace(jumlah_pelanggan.min(), jumlah_pelanggan.max(), 100).reshape(-1, 1)
garis_y = model.predict(garis_x)
plt.plot(garis_x, garis_y, color='red', linewidth=2,
         label=f'Y = {intersep:.0f} + {kemiringan:.0f}X', zorder=3)

# Menambahkan titik prediksi
plt.scatter([prediksi_x], [prediksi_y], color='green', s=150,
           marker='*', label=f'Prediksi: X={prediksi_x}, Y={prediksi_y:.0f}', zorder=10)

# Menyesuaikan plot
plt.xlabel('Jumlah Pelanggan', fontsize=12)
plt.ylabel('Pendapatan (ribuan)', fontsize=12)
plt.title('Masalah 1: Regresi Linear Pelanggan vs Pendapatan\n' +
          f'R² = {r2:.4f}, MSE = {mse:.2f}', fontsize=14, pad=20)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Menambahkan kotak teks dengan persamaan
teks_persamaan = f'Y = {kemiringan:.0f}X + {intersep:.0f}\nR² = {r2:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.05, 0.95, teks_persamaan, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.show()

# Verifikasi perhitungan manual
print("\nVerifikasi Perhitungan Manual:")
print("-" * 40)
pelanggan_rata = np.mean(jumlah_pelanggan.flatten())
pendapatan_rata = np.mean(pendapatan)

print(f"X̄ (rata-rata pelanggan): {pelanggan_rata}")
print(f"Ȳ (rata-rata pendapatan): {pendapatan_rata}")

# Menghitung kemiringan secara manual
pembilang = sum((jumlah_pelanggan.flatten() - pelanggan_rata) * (pendapatan - pendapatan_rata))
penyebut = sum((jumlah_pelanggan.flatten() - pelanggan_rata) ** 2)
kemiringan_manual = pembilang / penyebut

print(f"Kemiringan (manual): {kemiringan_manual}")
print(f"Intersep (manual): {pendapatan_rata - kemiringan_manual * pelanggan_rata}")

# Menampilkan perhitungan detail
print("\nPerhitungan Detail:")
for i in range(len(jumlah_pelanggan.flatten())):
    xi_diff = jumlah_pelanggan.flatten()[i] - pelanggan_rata
    yi_diff = pendapatan[i] - pendapatan_rata
    print(f"Titik {i+1}: ({jumlah_pelanggan.flatten()[i]}, {pendapatan[i]}) -> ({xi_diff:+.1f}) * ({yi_diff:+.1f}) = {xi_diff * yi_diff:+.1f}")

print(f"\nPembilang: {pembilang}")
print(f"Penyebut: {penyebut}")
print(f"Kemiringan = {pembilang}/{penyebut} = {kemiringan_manual}")
