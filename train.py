import pandas as pd
import numpy as np
import re
import json
import pickle
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from create_config import config_file

df = pd.read_csv('train.csv')

# df = df.fillna(method='ffill')

# for column in df.columns:
#     if df[column].dtype == type(object):
#         le = preprocessing.LabelEncoder()
#         df[column] = le.fit_transform(df[column])

num_cols = ['B','C','H','K','N','O']
text_cols = ['A','D','E','F','G','I','J','L','M']

# print(df[num_cols].head(1))
# print(df[text_cols].head(1))
# print(id.head(1))

id = df['id']
X = df.drop('id', axis=1)
# print(X.head())


num_data = df[num_cols]
text_data = df[text_cols]
# num_data = num_data.fillna(method='ffill')
# no missing values but inf values
# print(np.isinf(num_data))
# print(num_data.isnull().any())

# num_data = df[num_cols].values
# text_data = df[text_cols].values

y = df['P'].values
# X = df.drop('P', axis=1).values


X_train, X_test, y_train, y_test = train_test_split(num_data,y, test_size = 0.2, random_state=42, stratify=y)

with open('config_file.json','rb') as fi:
    config_dict = json.load(fi)

# print(config_dict.keys())

# target_names = ['class 0','class 1']

# drop id
# try numeric col only
# combine all text col
# feature union for all cols


for key,value in config_dict.items():
    # clf = value
    clf = Pipeline([
        ('imputer', Imputer()),
        ('model',eval(value))
    ])
    # clf = eval(value)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy= clf.score(X_test, y_test)
    print(key,train_accuracy)
    print(key, test_accuracy)
#     with open('model-' + name + '.pickle', 'wb') as fo:
#         pickle.dump(clf,fo)
#     with open('classifier-'+name+'.txt','wb') as f:
        # f.write(name)
        # f.write(train_accuracy)
        # f.write(test_accuracy)
        # f.write(confusion_matrix(y_test, y_pred))
        # f.write(classification_report(y_test, y_pred, target_names=target_names))
