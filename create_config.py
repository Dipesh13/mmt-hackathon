import json
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
    "KNeighborsClassifier(2)",
    "SVC(kernel='linear', C=0.025)",
    "SVC(gamma=2, C=1)",
    "GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)",
    "DecisionTreeClassifier(max_depth=5)",
    "RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)",
    "MLPClassifier(alpha=1)",
    "AdaBoostClassifier()",
    "GaussianNB()",
    "QuadraticDiscriminantAnalysis()"]

names = ["Nearest Neighbors", "Linear SVM",
         "RBF SVM", "Gaussian Process","Decision Tree",
         "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

config_file = dict(zip(names,classifiers))
with open('config_file.json','wb') as fo:
    json.dump(config_file,fo, sort_keys=True, indent=4)