import joblib
import pandas as pd

model = joblib.load('D:\song_predicter\data\model.pkl')
scaler = joblib.load('scaler.pkl')

scaling_table = ['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Valence']


new_track = [[0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]]  # Пример данных в том же порядке, что у твоего X
new_track_df = pd.DataFrame(new_track, columns=scaling_table)

new_track_scaled = scaler.transform(new_track_df)
new_track_scaled_df = pd.DataFrame(new_track_scaled, columns=scaling_table)

prediction = model.predict(new_track_scaled_df)
print(f"Predicted continent: {prediction[0]}")