import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("D:/song_predicter/data/Spotify_Dataset_V3.csv", sep = ';')
print(df.iloc[:5,4:10])
#print(df.columns)
#print(df.info())

scaling_table = ['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Valence']

X = df[scaling_table]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns = scaling_table)

df[scaling_table] = X_scaled_df

print(df.iloc[:5,4:10])

df.to_csv("D:/song_predicter/data/Spotify_Dataset_V3_scaled.csv", sep=';', index=False)
