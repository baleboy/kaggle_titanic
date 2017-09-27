import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib
import sys
import preprocessing as pp # Local module

# Pick the selected features and apply the preprocessing transformations.
# Return the feature matrix, target vector (if present) and passenger IDs.
# Works on both training and testing data.
def get_data(traindatafile):

    data = pd.read_csv(traindatafile)
    y = data['Survived']
    data.drop(['Survived'], axis=1, inplace=True)

    X = pp.process_training_data(data)

    return X, y

# Return a list of pipelines with different transforms
# and the same model
def model_pipelines(model, name):

    pipelines=[]
    pipelines.append({'name': name, 'pipe': make_pipeline(model)})
    pipelines.append({'name': name + "(poly2)", 'pipe': make_pipeline(PolynomialFeatures(degree=2, interaction_only=True),
    model)})

    return pipelines

# main

trainfile = sys.argv[1]
X, y = get_data(trainfile)

models = [{'name': 'logreg', 'model': LogisticRegressionCV()},
          {'name': 'forest', 'model': RandomForestClassifier(n_estimators=100)},
          {'name': 'neuralnet', 'model': MLPClassifier(max_iter = 1000)},
          {'name': 'svc', 'model': SVC()}]

pipelines = []

for m in models:
    pipelines.extend(model_pipelines(m['model'], m['name']))

best_pipeline = None
best_score = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state = 42)

for p in pipelines:
    print("*** " + p['name'])
    # score = cross_val_score(p['pipe'], X_train, y, scoring='accuracy').mean()
    p['pipe'].fit(X_train, y_train)
    print("Training score: {}".format(p['pipe'].score(X_train, y_train)))
    score = p['pipe'].score(X_test, y_test)
    if score > best_score:
        best_score = score
        best_pipeline = p
    print("CV score: {}".format(score))

if (best_pipeline != None):
    print("Best pipeline: " + best_pipeline['name'] + ", score: {}".format(best_score))
    joblib.dump(best_pipeline['pipe'], 'titanic.pkl')
