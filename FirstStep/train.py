import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier


music_data = pd.read_csv('FirstStep/music.csv')
X = music_data.drop(columns=['genre'])
Y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(X, Y)
joblib.dump(model, "music-recommender.joblib")  