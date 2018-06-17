import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

df = pd.read_csv('train.csv')
with open('config_file.json','rb') as fi:
    config_dict = json.load(fi)

num_cols = ['B','C','H','K','N','O']

num_data = df[num_cols]
y = df['P']

X_train, X_test, y_train, y_test = train_test_split(num_data, y, test_size = 0.2, random_state=42, stratify=y)


for key,value in config_dict.items():
    clf = Pipeline([
        ('imputer', Imputer()),
        ('model',eval(value))
    ])
    clf.fit(X_train,y_train)
    # y_pred = clf.predict(X_test)
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy= clf.score(X_test, y_test)
    print(key,'Train acc:',train_accuracy)
    print(key,'Test acc:',test_accuracy)