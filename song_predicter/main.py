import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

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

X = df[scaling_table]
Y = df['Continent']

X_train, X_test, Y_train, Y_test = train_test_split (X,Y, test_size = 0.2, random_state = 42)

model = RandomForestClassifier(random_state = 42)
model.fit(X_train, Y_train)

accuracy = model.score(X_test, Y_test)
print(accuracy)

Y_pred = model.predict(X_test)
print(classification_report(Y_test, Y_pred))

joblib.dump(model, 'D:/song_predicter/data/model.pkl')

joblib.dump(scaler, 'scaler.pkl')


