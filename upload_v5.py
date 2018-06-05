import pandas as pd
import re
import json
import pickle
import argparse
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
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


# address the problem of null values
df = pd.read_csv('train.csv')
df = df.fillna(method='ffill')

# address the problems of string to float/int conversion
for column in df.columns:
    if df[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        df[column] = le.fit_transform(df[column])

y = df['P'].values
X = df.drop('P', axis=1).values


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42, stratify=y)

classifiers = [
    KNeighborsClassifier(2),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

names = ["Nearest Neighbors", "Linear SVM",
         "RBF SVM", "Gaussian Process","Decision Tree",
         "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

target_names = ['class 0','class 1']

for clf,name in zip(classifiers,names):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy= clf.score(X_test, y_test)
    with open('model-' + name + '.pickle', 'wb') as fo:
        pickle.dump(clf,fo)
    with open('classifier-'+name+'.txt','wb') as f:
        f.write("Train and Test accuracy is {} and {} resp.".format(train_accuracy,test_accuracy))
        # f.write(name)
        # f.write(train_accuracy)
        # f.write(test_accuracy)
        # f.write(confusion_matrix(y_test, y_pred))
        # f.write(classification_report(y_test, y_pred, target_names=target_names))

config_file = dict(zip(names,classifiers))
with open('config_file.json','wb') as fi:
    fi.dump(config_file)

parser = argparse.ArgumentParser(description='model name to load the pickle file')
parser.add_argument('-modelname', help='name of the classifier',default="Linear SVM")
args = parser.parse_args()
model_name = args.modelname

def model_pred(model_name):
    """Test based on model name (defaults to Linear SVM) to generate the pred.csv file

    Following models are available

    "Nearest Neighbors"
    "Linear SVM"
    "RBF SVM"
    "Gaussian Process"
    "Decision Tree"
    "Random Forest"
    "Neural Net"
    "AdaBoost"
    "Naive Bayes"
    "QDA"

    """
    with open('model-'+model_name+'.pickle', 'rb') as clf_file:
        clf_test = pickle.load(clf_file)

    df_test = pd.read_csv("test.csv")
    df_test = df_test.fillna(method='ffill')

    for column in df_test.columns:
        if df_test[column].dtype == type(object):
            le = preprocessing.LabelEncoder()
            df_test[column] = le.fit_transform(df_test[column])

    y_test_output = df_test.values
    y_pred_output = clf_test.predict(y_test_output)
    output = []
    for a,b in zip(y_test_output,y_pred_output):
        output.append([int(a[0]),b])

    out_df = pd.DataFrame(output,columns = ['id','P'])
    out_df.to_csv('pred-'+model_name+'.csv',index = False)

if __name__ == '__main__':
    model_pred(model_name)