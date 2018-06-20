import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

list_df = [[1,5,'a','bob',1],[2,6,'an','foo',0],[3,7,'a','foobar',0],[4,8,'and','bob',1]]
df= pd.DataFrame(list_df)
# print(df.info())

num_cols = [0,1]
text_cols = [2,3]

def combine_text_columns(data_frame):
    # next line handles missing text values.
    data_frame.fillna("", inplace=True)
    return data_frame.apply(lambda x: " ".join(x), axis=1)


# num_data = df[num_cols]
# text_data = df[text_cols]
# text_data = combine_text_columns(text_data)
# print(text_data)
# y = df[4]
# X= df[num_cols+text_cols]
# print(X)
X_train, X_test, y_train, y_test = train_test_split(df[[0,1,2,3]], df[4], test_size = 0.4, random_state=42)

get_numeric_data = FunctionTransformer(lambda x:x[[0,1]],validate=False)
get_text_data = FunctionTransformer(lambda x:combine_text_columns(x[[2,3]]),validate=False)

numeric_pipeline = Pipeline([
    ('selector', get_numeric_data),
    ('imputer', Imputer())
])

text_pipeline = Pipeline([
    ('selector', get_text_data),
    ('vectorizer',CountVectorizer())
])

clf = Pipeline([
    ('union', FeatureUnion([
        ('num', numeric_pipeline),
        ('text', text_pipeline)
    ])),
    ('model',LogisticRegression())
])

# clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train,y_train)
train_accuracy = clf.score(X_train, y_train)
test_accuracy= clf.score(X_test, y_test)
print('Train acc:',train_accuracy)
print('Test acc:',test_accuracy)