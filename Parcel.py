import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

mean=5
std_dev=2
sample_size=1000
heights= np.random.normal(mean,std_dev,sample_size)

df=pd.DataFrame(heights,columns=['Waktu Pengiriman'])
print(df.head())

plt.figure(figsize=(8,6))
sns.histplot(df['Waktu Pengiriman'], kde=True, color='blue')  
plt.title('Distribusi Waktu Pengiriman dalam Kabupaten')       
plt.xlabel('Waktu pengiriman')                               
plt.ylabel('Frekuensi')                                  
plt.show()

prob = np.sum((df['Waktu Pengiriman']>=3)&(df['Waktu Pengiriman']<=7))/sample_size
print(f"Probaibilitas Waktu Pengiriman dalam 3 dan 7:{prob:.2f}")

plt.figure(figsize=(8,6))
sns.ecdfplot(df['Waktu Pengiriman'], color='green')
plt.title('Probabilitas Kumulatif Waktu pengiriman')
plt.xlabel('Waktu pengiriman(Hari)')
plt.ylabel('Probabilitas Kumulatif')
plt.show()

print(df['Waktu Pengiriman'].describe())
