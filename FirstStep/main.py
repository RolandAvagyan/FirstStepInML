import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

music_data = pd.read_csv('FirstStep/music.csv')
X = music_data.drop(columns=['genre'])
Y = music_data['genre']

model = joblib.load("music-recommender.joblib")
prediction = model.predict([[21, 1]])

tree.export_graphviz(model, out_file="music-recommender.dot",
                     feature_names=['age', 'gender'],
                     class_names=sorted(Y.unique()),
                     label="all",
                     rounded=True,
                     filled=True)
