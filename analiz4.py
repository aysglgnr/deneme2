import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

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

print(climate_df.columns)


# İklim koordinatlarını al
climate_coords = climate_df[['lat', 'lon']].values

if climate_coords.shape[0] == 0:
    print("Climate coordinates are empty!")
else:
    # En yakın komşuları bulmak için model oluştur
    nbrs = NearestNeighbors(radius=1.0).fit(climate_coords)

    # Microplastic verileri ile eşleşen iklim verilerini bul
    microplastic_coords = microplastic_df[['latitude', 'longitude']].values
    distances, indices = nbrs.radius_neighbors(microplastic_coords)

    # Eşleşen verileri birleştir
    merged_list = []
    for i, idx in enumerate(indices):
        if len(idx) > 0:  # Eğer eşleşen iklim verisi varsa
            microplastic_row = microplastic_df.iloc[i]
            for climate_idx in idx:
                climate_row = climate_df.iloc[climate_idx]
                merged_row = pd.concat([microplastic_row, climate_row], axis=0)
                merged_list.append(merged_row)

    # Birleştirilmiş DataFrame
    merged_df = pd.DataFrame(merged_list)

    # Korelasyon analizi
    if not merged_df.empty:
        correlation = merged_df.corr()
        print("Korelasyon Matrisi:\n", correlation[['mp_concentration']])

        # Kullanmak istediğiniz değişken
        x_variable = 'TX90p'  # Bu satırı değiştirin

        # Scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=merged_df, x=x_variable, y='mp_concentration')
        plt.title(f'{x_variable} vs Microplastic Concentration')
        plt.xlabel(f'{x_variable} (Variable)')
        plt.ylabel('Microplastic Concentration')
        plt.grid()
        plt.show()
    else:
        print("No matching values found.")
