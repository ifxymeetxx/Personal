import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Tetap diimpor untuk visualisasi meskipun manual calculation diubah
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Data
hari = [1, 2, 3, 4, 5]
X1 = np.array([50, 60, 70, 80, 90])  # Pengunjung
X2 = np.array([2, 3, 4, 5, 6])       # Item menu baru
y = np.array([400, 480, 560, 640, 720])  # Penjualan (ribuan)

# Persiapkan data untuk regresi berganda (untuk model sklearn)
X = np.column_stack((X1, X2))

# Buat dan latih model sklearn (ini berhasil karena sklearn menangani multicollinearity)
model = LinearRegression()
model.fit(X, y)

# Dapatkan koefisien dari model sklearn
b1, b2 = model.coef_
intersep = model.intercept_

# Buat prediksi menggunakan model sklearn
y_prediksi = model.predict(X)

# Hitung metrik menggunakan model sklearn
mse = mean_squared_error(y, y_prediksi)
r2 = r2_score(y, y_prediksi)

# Prediksi untuk 75 pengunjung dan 4 item menu baru menggunakan model sklearn
prediksi_x1, prediksi_x2 = 75, 4
prediksi_y = model.predict([[prediksi_x1, prediksi_x2]])[0]

# Cetak hasil dari model sklearn
print("Masalah 5: Analisis Regresi Berganda")
print("Pengunjung dan Item Menu Baru vs Penjualan")
print("=" * 50)
print(f"Titik Data:")
print(f"Hari: {hari}")
print(f"Pengunjung (X1): {X1}")
print(f"Menu Baru (X2): {X2}")
print(f"Penjualan (Y): {y}")
print()
print(f"Persamaan Regresi Berganda (dari Sklearn): Y = {intersep:.2f} + {b1:.2f}X1 + {b2:.2f}X2")
print(f"Disederhanakan: Y = {intersep:.0f} + {b1:.0f}X1 + {b2:.0f}X2")
print(f"Intersep (a): {intersep:.2f}")
print(f"Koefisien b1 (Pengunjung): {b1:.2f}")
print(f"Koefisien b2 (Menu Baru): {b2:.2f}")
print(f"R-kuadrat: {r2:.4f}")
print(f"Mean Squared Error: {mse:.2f}")
print()
print(f"Prediksi untuk {prediksi_x1} pengunjung dan {prediksi_x2} item menu baru: {prediksi_y:.0f} ribu")

# Buat visualisasi 3D (menggunakan hasil dari model sklearn)
fig = plt.figure(figsize=(14, 10))

# Plot scatter 3D
ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(X1, X2, y, color='blue', alpha=0.8, s=100, label='Titik Data')

# Buat bidang regresi menggunakan koefisien dari model sklearn
rentang_X1 = np.linspace(X1.min()-5, X1.max()+5, 10)
rentang_X2 = np.linspace(X2.min()-1, X2.max()+1, 10)
mesh_X1, mesh_X2 = np.meshgrid(rentang_X1, rentang_X2)
mesh_Y = intersep + b1 * mesh_X1 + b2 * mesh_X2

ax1.plot_surface(mesh_X1, mesh_X2, mesh_Y, alpha=0.3, color='red', label='Bidang Regresi')

# Tambahkan titik prediksi
ax1.scatter([prediksi_x1], [prediksi_x2], [prediksi_y],
           color='green', s=200, marker='*',
           label=f'Prediksi: ({prediksi_x1}, {prediksi_x2}) → {prediksi_y:.0f}')

ax1.set_xlabel('Pengunjung (X1)', fontsize=10)
ax1.set_ylabel('Menu Baru (X2)', fontsize=10)
ax1.set_zlabel('Penjualan (ribuan)', fontsize=10)
ax1.set_title('Regresi Berganda 3D\nPengunjung & Menu vs Penjualan', fontsize=12)

# Plot 2D (menggunakan hasil dari model sklearn)
# Pengunjung vs Penjualan
ax2 = fig.add_subplot(222)
ax2.scatter(X1, y, color='blue', alpha=0.7, s=100, label='Titik Data')
garis_X1 = np.linspace(X1.min(), X1.max(), 100)
# Perhatikan bahwa untuk plot 2D dari regresi berganda, kita memprediksi dengan nilai rata-rata variabel lain
garis_y_x1 = intersep + b1 * garis_X1 + b2 * np.mean(X2)
ax2.plot(garis_X1, garis_y_x1, color='red', linewidth=2,
         label=f'Y = {intersep:.0f} + {b1:.0f}X1 + {b2:.0f}*{np.mean(X2):.1f}')
ax2.set_xlabel('Pengunjung (X1)')
ax2.set_ylabel('Penjualan (ribuan)')
ax2.set_title('Pengunjung vs Penjualan\n(Menu tetap pada rata-rata)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Menu Baru vs Penjualan
ax3 = fig.add_subplot(223)
ax3.scatter(X2, y, color='blue', alpha=0.7, s=100, label='Titik Data')
garis_X2 = np.linspace(X2.min(), X2.max(), 100)
# Perhatikan bahwa untuk plot 2D dari regresi berganda, kita memprediksi dengan nilai rata-rata variabel lain
garis_y_x2 = intersep + b1 * np.mean(X1) + b2 * garis_X2
ax3.plot(garis_X2, garis_y_x2, color='red', linewidth=2,
         label=f'Y = {intersep:.0f} + {b1:.0f}*{np.mean(X1):.0f} + {b2:.0f}X2')
ax3.set_xlabel('Item Menu Baru (X2)')
ax3.set_ylabel('Penjualan (ribuan)')
ax3.set_title('Menu Baru vs Penjualan\n(Pengunjung tetap pada rata-rata)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot residual (menggunakan hasil dari model sklearn)
ax4 = fig.add_subplot(224)
residual = y - y_prediksi
ax4.scatter(y_prediksi, residual, color='purple', alpha=0.7, s=100)
ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
ax4.set_xlabel('Penjualan Prediksi')
ax4.set_ylabel('Residual')
ax4.set_title('Plot Residual')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# --- Perhitungan manual diubah untuk menggunakan hanya X1 ---
# Ini untuk mendemonstrasikan metode matriks pada kasus yang tidak singular
print("\nPerhitungan Manual Menggunakan Metode Matriks (Hanya dengan X1):")
print("-" * 60)

# Buat matriks desain untuk hanya X1 (dengan kolom intersep)
# matriks_X = np.column_stack((np.ones(len(X1)), X1, X2)) # Baris asli
matriks_X_manual = np.column_stack((np.ones(len(X1)), X1)) # Baris diubah untuk hanya X1
print("Matriks Desain X (Manual Calculation):")
print("   1   X1")
for i in range(len(X1)):
    # print(f"{i+1}: 1   {X1[i]}   {X2[i]}") # Baris asli
    print(f"{i+1}: 1   {X1[i]}") # Baris diubah

print(f"\nVektor Y: {y}")

# Hitung koefisien: β = (X'X)^(-1)X'y menggunakan matriks_X_manual
XtX_manual = matriks_X_manual.T @ matriks_X_manual
print("\nMatriks XtX (Manual Calculation):")
print(XtX_manual)

# Cek determinan sebelum inversi (opsional, tapi bagus untuk debugging)
det_XtX_manual = np.linalg.det(XtX_manual)
print(f"\nDeterminan XtX (Manual Calculation): {det_XtX_manual}")


# Sekarang matriks XtX_manual tidak singular, inversi akan berhasil
try:
    XtX_inv_manual = np.linalg.inv(XtX_manual)
    print("\nMatriks (XtX)^-1 (Manual Calculation):")
    print(XtX_inv_manual)

    Xty_manual = matriks_X_manual.T @ y
    print("\nMatriks Xty (Manual Calculation):")
    print(Xty_manual)

    koefisien_manual = XtX_inv_manual @ Xty_manual

    print(f"\nKoefisien yang dihitung (Manual Calculation):")
    # Karena hanya ada X1, koefisien adalah intersep dan koefisien X1
    print(f"Intersep (a): {koefisien_manual[0]:.2f}")
    print(f"b1 (Pengunjung): {koefisien_manual[1]:.2f}")

except np.linalg.LinAlgError as e:
    print(f"\nError saat menghitung invers (Manual Calculation): {e}")
    print("Pastikan kolom-kolom dalam matriks desain tidak saling berkorelasi sempurna.")


# Verifikasi dengan hasil sklearn (untuk regresi sederhana hanya dengan X1)
# Latih model sklearn terpisah untuk hanya X1 vs Y
model_simple = LinearRegression()
model_simple.fit(X1.reshape(-1, 1), y)
intersep_simple = model_simple.intercept_
b1_simple = model_simple.coef_[0]

print(f"\n--- Perbandingan dengan Sklearn (Simple Regression X1 vs Y) ---")
print(f"Intersep (a) Sklearn: {intersep_simple:.2f}")
print(f"b1 (Pengunjung) Sklearn: {b1_simple:.2f}")

# Interpretasi (dari model sklearn berganda asli)
print(f"\nInterpretasi (berdasarkan model Sklearn berganda):")
print("-" * 20)
print(f"• Penjualan dasar (tanpa pengunjung, tanpa menu baru): {intersep:.0f} ribu")
print(f"• Setiap pengunjung tambahan meningkatkan penjualan sebesar: {b1:.0f} ribu")
print(f"• Setiap item menu baru meningkatkan penjualan sebesar: {b2:.0f} ribu")
print(f"• Model menjelaskan {r2*100:.1f}% dari variasi penjualan")
print(f"• Untuk {prediksi_x1} pengunjung dan {prediksi_x2} item menu baru:")
print(f"  Penjualan = {intersep:.0f} + {b1:.0f}×{prediksi_x1} + {b2:.0f}×{prediksi_x2} = {prediksi_y:.0f} ribu")
