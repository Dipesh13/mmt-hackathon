import pandas as pd
import numpy as np
import re
import json
import pickle
import sys
from sklearn.base import TransformerMixin
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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
# from create_config import config_file

class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

df = pd.read_csv('train.csv')
with open('config_file.json','rb') as fi:
    config_dict = json.load(fi)

num_cols = ['B','C','H','K','N','O']
text_cols = ['A','D','E','F','G','I','J','L','M']

def combine_text_columns(data_frame):
    # next line handles missing text values.
    data_frame.fillna("", inplace=True)
    return data_frame.apply(lambda x: " ".join(x), axis=1)

y = df['P']
X = df.drop('P', axis=1)
X = X.drop('id', axis=1)
num_data = df[num_cols]
text_data = df[text_cols]
# text_data = combine_text_columns(text_data)

X_train, X_test, y_train, y_test = train_test_split(df[['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']], df['P'], test_size = 0.2, random_state=42, stratify=y)

get_numeric_data = FunctionTransformer(lambda x:x[['B','C','H','K','N','O']],validate=False)
get_text_data = FunctionTransformer(lambda x:combine_text_columns(x[['A','D','E','F','G','I','J','L','M']]),validate=False)

numeric_pipeline = Pipeline([
    ('selector', get_numeric_data),
    ('imputer', Imputer())
])

text_pipeline = Pipeline([
    ('selector', get_text_data),
    # ('to_dense', DenseTransformer()),
    ('vectorizer',CountVectorizer())
])

for key,value in config_dict.items():
    if key != "Gaussian Process" and key!="Naive Bayes" and key!="QDA":
        clf = Pipeline([
            ('union', FeatureUnion([
                ('num', numeric_pipeline),
                ('text', text_pipeline)
            ])),
            ('model',eval(value))
        ])
        # clf = eval(value)
        clf.fit(X_train,y_train)
        # y_pred = clf.predict(X_test)
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