import pandas as pd
import numpy as np
import json
from sklearn.base import TransformerMixin
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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion

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

text_cols = ['A','D','E','F','G','I','J','L','M']

def combine_text_columns(data_frame):
    # next line handles missing text values.
    text_data.fillna("", inplace=True)
    return text_data.apply(lambda x: " ".join(x), axis=1)


text_data = df[text_cols]
text_data = combine_text_columns(text_data)
# print(text_data)
y = df['P']

X_train, X_test, y_train, y_test = train_test_split(text_data, y, test_size = 0.2, random_state=42, stratify=y)

for key,value in config_dict.items():
    clf = Pipeline([
        ('vec', CountVectorizer()),
        ('to_dense', DenseTransformer()),
        ('model',eval(value))
    ])
    clf.fit(X_train,y_train)
    # y_pred = clf.predict(X_test)
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy= clf.score(X_test, y_test)
    print(key,'Train acc:',train_accuracy)
    print(key,'Test acc:',test_accuracy)