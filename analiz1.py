import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial import cKDTree

# CSV dosyalarını yükle
microplastic_df = pd.read_csv('microplastic.csv')
climate_df = pd.read_csv('filtered_data.csv')

# Sayısal olmayan ('--') değerleri NaN ile değiştir
microplastic_df.replace('--', np.nan, inplace=True)
climate_df.replace('--', np.nan, inplace=True)

# Verileri sayısal hale getir
microplastic_df = microplastic_df.apply(pd.to_numeric, errors='coerce')
climate_df = climate_df.apply(pd.to_numeric, errors='coerce')

# Eksik (NaN) değerleri kaldır
microplastic_df.dropna(inplace=True)
climate_df.dropna(inplace=True)

# Eşleşen boylam ve enlem ile birleştir
tolerance = 1  # Tolerans aralığı

# Kd-tree kullanarak yakın komşuları bul
climate_tree = cKDTree(climate_df[['lat', 'lon']].values)
distances, indices = climate_tree.query(microplastic_df[['latitude', 'longitude']].values, distance_upper_bound=tolerance)
valid_indices = indices[~np.isinf(distances)]

# Eşleşen verileri birleştir
merged_list = []
for idx in valid_indices:
    merged_row = pd.concat([microplastic_df.iloc[idx], climate_df.iloc[indices[idx]]])
    merged_list.append(merged_row)

# Birleştirilmiş DataFrame
merged_df = pd.DataFrame(merged_list)

# Korelasyon analizi
if not merged_df.empty:
    correlation = merged_df.corr()
    print("Korelasyon Matrisi:\n", correlation[['mp_concentration']])

    # Kullanmak istediğiniz değişken
    x_variable = 'R90P'  # Bu satırı değiştirin

    # Scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=merged_df, x=x_variable, y='mp_concentration')
    plt.title(f'{x_variable} vs Microplastic Concentration')
    plt.xlabel(f'{x_variable} (Değişken)')
    plt.ylabel('Mikroplastik Konsantrasyonu')
    plt.grid()
    plt.show()
else:
    print("Eşleşen değer bulunamadı.")
