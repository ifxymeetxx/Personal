import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Data
jam_internet = np.array([5, 10, 15, 20]).reshape(-1, 1)  # Jam internet
biaya_data = np.array([25000, 50000, 75000, 100000])  # Biaya data (Rupiah)

# Membuat dan melatih model
model = LinearRegression()
model.fit(jam_internet, biaya_data)

# Mendapatkan koefisien
kemiringan = model.coef_[0]
intersep = model.intercept_

# Membuat prediksi
prediksi_biaya = model.predict(jam_internet)

# Menghitung metrik
mse = mean_squared_error(biaya_data, prediksi_biaya)
r2 = r2_score(biaya_data, prediksi_biaya)

# Prediksi untuk 18 jam
prediksi_x = 18
prediksi_y = model.predict([[prediksi_x]])[0]

# Menampilkan hasil
print("Masalah 4: Analisis Jam Internet vs Biaya Data")
print("=" * 53)
print(f"Titik Data:")
print(f"Jam Internet (X): {jam_internet.flatten()}")
print(f"Biaya Data (Y): {biaya_data}")
print()
print(f"Persamaan Regresi: Y = {intersep:.2f} + {kemiringan:.2f}X")
print(f"Disederhanakan: Y = {kemiringan:.0f}X + {intersep:.0f}")
print(f"Kemiringan (b): {kemiringan:.2f}")
print(f"Intersep (a): {intersep:.2f}")
print(f"R-squared: {r2:.4f}")
print(f"Mean Squared Error: {mse:.2f}")
print()
print(f"Prediksi untuk {prediksi_x} jam: Rp {prediksi_y:,.0f}")
print(f"Biaya per jam: Rp {kemiringan:,.0f}")

# Membuat visualisasi
plt.figure(figsize=(10, 6))

# Plot titik data
plt.scatter(jam_internet, biaya_data, color='blue', alpha=0.7, s=100,
           label='Titik Data', zorder=5)

# Plot garis regresi
garis_x = np.linspace(jam_internet.min(), jam_internet.max(), 100).reshape(-1, 1)
garis_y = model.predict(garis_x)
plt.plot(garis_x, garis_y, color='red', linewidth=2,
         label=f'Y = {kemiringan:.0f}X + {intersep:.0f}', zorder=3)

# Menambahkan titik prediksi
plt.scatter([prediksi_x], [prediksi_y], color='green', s=150,
           marker='*', label=f'Prediksi: {prediksi_x}j → Rp {prediksi_y:,.0f}', zorder=10)

# Menyesuaikan plot
plt.xlabel('Jam Internet', fontsize=12)
plt.ylabel('Biaya Data (Rupiah)', fontsize=12)
plt.title('Masalah 4: Regresi Linear Jam Internet vs Biaya Data\n' +
          f'Biaya: Rp {kemiringan:,.0f} per jam', fontsize=14, pad=20)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Format sumbu y untuk menampilkan nilai dalam ribuan
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rp {x/1000:.0f}rb'))

plt.tight_layout()

# Menambahkan kotak teks dengan persamaan
teks_persamaan = f'Y = {kemiringan:.0f}X\nBiaya: Rp {kemiringan:,.0f}/jam\nR² = {r2:.4f}'
props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
plt.text(0.05, 0.95, teks_persamaan, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.show()

# Verifikasi perhitungan manual
print("\nVerifikasi Perhitungan Manual:")
print("-" * 40)
jam_rata = np.mean(jam_internet.flatten())
biaya_rata = np.mean(biaya_data)

print(f"X̄ (rata-rata jam): {jam_rata}")
print(f"Ȳ (rata-rata biaya): Rp {biaya_rata:,.0f}")

# Menghitung kemiringan secara manual
pembilang = sum((jam_internet.flatten() - jam_rata) * (biaya_data - biaya_rata))
penyebut = sum((jam_internet.flatten() - jam_rata) ** 2)
kemiringan_manual = pembilang / penyebut

print(f"Kemiringan (manual): {kemiringan_manual:,.0f}")
print(f"Intersep (manual): {biaya_rata - kemiringan_manual * jam_rata:.0f}")

# Menampilkan perhitungan detail
print("\nPerhitungan Detail:")
for i in range(len(jam_internet.flatten())):
    xi_diff = jam_internet.flatten()[i] - jam_rata
    yi_diff = biaya_data[i] - biaya_rata
    print(f"Titik {i+1}: ({jam_internet.flatten()[i]}j, Rp {biaya_data[i]:,}) -> ({xi_diff:+.1f}) * ({yi_diff:+,.0f}) = {xi_diff * yi_diff:+,.0f}")

print(f"\nPembilang: {pembilang:,.0f}")
print(f"Penyebut: {penyebut}")
print(f"Kemiringan = {pembilang:,.0f}/{penyebut} = {kemiringan_manual:,.0f}")
print(f"Artinya biaya adalah Rp {kemiringan_manual:,.0f} per jam penggunaan internet")

# Analisis tambahan
print(f"\nAnalisis Tambahan:")
print(f"- Hubungan linear sempurna (R² = {r2:.4f})")
print(f"- Tidak ada biaya tetap (intersep ≈ {intersep:.0f})")
print(f"- Struktur biaya: Murni variabel sebesar Rp {kemiringan:,.0f} per jam")
