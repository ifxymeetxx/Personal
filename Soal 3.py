
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Data
umur = np.array([1, 3, 5, 7, 9]).reshape(-1, 1)  # Umur (tahun)
harga_jual = np.array([150, 120, 90, 60, 30])  # Harga jual (juta)

# Membuat dan melatih model
model = LinearRegression()
model.fit(umur, harga_jual)

# Mendapatkan koefisien
kemiringan = model.coef_[0]
intersep = model.intercept_

# Membuat prediksi
prediksi_harga = model.predict(umur)

# Menghitung metrik
mse = mean_squared_error(harga_jual, prediksi_harga)
r2 = r2_score(harga_jual, prediksi_harga)

# Prediksi untuk umur 4 tahun
prediksi_x = 4
prediksi_y = model.predict([[prediksi_x]])[0]

# Menampilkan hasil
print("Masalah 3: Analisis Umur vs Harga Jual")
print("=" * 45)
print(f"Titik Data:")
print(f"Umur (X): {umur.flatten()}")
print(f"Harga Jual (Y): {harga_jual}")
print()
print(f"Persamaan Regresi: Y = {intersep:.2f} + ({kemiringan:.2f})X")
print(f"Disederhanakan: Y = {intersep:.0f} - {abs(kemiringan):.0f}X")
print(f"Kemiringan (b): {kemiringan:.2f}")
print(f"Intersep (a): {intersep:.2f}")
print(f"R-squared: {r2:.4f}")
print(f"Mean Squared Error: {mse:.2f}")
print()
print(f"Prediksi untuk umur {prediksi_x} tahun: {prediksi_y:.0f} juta")

# Membuat visualisasi
plt.figure(figsize=(10, 6))

# Plot titik data
plt.scatter(umur, harga_jual, color='blue', alpha=0.7, s=100,
           label='Titik Data', zorder=5)

# Plot garis regresi
garis_x = np.linspace(umur.min(), umur.max(), 100).reshape(-1, 1)
garis_y = model.predict(garis_x)
plt.plot(garis_x, garis_y, color='red', linewidth=2,
         label=f'Y = {intersep:.0f} - {abs(kemiringan):.0f}X', zorder=3)

# Menambahkan titik prediksi
plt.scatter([prediksi_x], [prediksi_y], color='green', s=150,
           marker='*', label=f'Prediksi: {prediksi_x} tahun → {prediksi_y:.0f} juta', zorder=10)

# Menyesuaikan plot
plt.xlabel('Umur (tahun)', fontsize=12)
plt.ylabel('Harga Jual (juta)', fontsize=12)
plt.title('Masalah 3: Regresi Linear Umur vs Harga Jual\n' +
          f'Harga turun {abs(kemiringan):.0f} juta per tahun', fontsize=14, pad=20)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Menambahkan kotak teks dengan persamaan
teks_persamaan = f'Y = {intersep:.0f} - {abs(kemiringan):.0f}X\nPenurunan: {abs(kemiringan):.0f}jt/tahun\nR² = {r2:.4f}'
props = dict(boxstyle='round', facecolor='lightcoral', alpha=0.8)
plt.text(0.05, 0.95, teks_persamaan, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.show()

# Verifikasi perhitungan manual
print("\nVerifikasi Perhitungan Manual:")
print("-" * 40)
umur_rata = np.mean(umur.flatten())
harga_rata = np.mean(harga_jual)

print(f"X̄ (rata-rata umur): {umur_rata}")
print(f"Ȳ (rata-rata harga): {harga_rata}")

# Menghitung kemiringan secara manual
pembilang = sum((umur.flatten() - umur_rata) * (harga_jual - harga_rata))
penyebut = sum((umur.flatten() - umur_rata) ** 2)
kemiringan_manual = pembilang / penyebut

print(f"Kemiringan (manual): {kemiringan_manual}")
print(f"Intersep (manual): {harga_rata - kemiringan_manual * umur_rata}")

# Menampilkan perhitungan detail
print("\nPerhitungan Detail:")
for i in range(len(umur.flatten())):
    xi_diff = umur.flatten()[i] - umur_rata
    yi_diff = harga_jual[i] - harga_rata
    print(f"Titik {i+1}: ({umur.flatten()[i]} tahun, {harga_jual[i]}jt) -> ({xi_diff:+.1f}) * ({yi_diff:+.1f}) = {xi_diff * yi_diff:+.1f}")

print(f"\nPembilang: {pembilang}")
print(f"Penyebut: {penyebut}")
print(f"Kemiringan = {pembilang}/{penyebut} = {kemiringan_manual}")
print(f"Artinya harga turun {abs(kemiringan_manual):.0f} juta per tahun bertambahnya umur")
