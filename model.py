import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

feat1 = pd.read_pickle("features.pkl")
label = pd.read_pickle("label.pkl")

feat1.sort_index(inplace=True)
label.sort_index(inplace=True)

X = []
Y = []
for col in feat1.columns:
    X.append( feat1[col].values.tolist() )
Y = label['like_count_36_hour'].values.tolist()

# X = X[:, None]
Xp = np.array(X)
print(Xp.shape)
print(Xp.T)
X = Xp.T
# print(len(X), len(X[0]))
# print(len(Y))
clf = RandomForestClassifier()
# clf = GaussianNB()
# clf = LogisticRegression(max_iter=10000)
clf.fit(X, Y)

pickle.dump(clf, open("model.pkl", "wb"))
# print(feat1)