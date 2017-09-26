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

import preprocessing as pp# Local module

# Pick the selected features and apply the preprocessing transformations.
# Return the feature matrix, target vector (if present) and passenger IDs.
# Works on both training and testing data.
def get_data(traindatafile, testdatafile):

    train_data = pd.read_csv(traindatafile)
    y = train_data['Survived']
    train_data.drop(['Survived'], axis=1, inplace=True)
    test_data = pd.read_csv(testdatafile)
    data = pd.concat([train_data, test_data], keys = ['train', 'test'])

    X_train, X_test = pp.process_data(data)

    pid_test = test_data['PassengerId']

    return X_train, X_test, pid_test, y

# Return a list of pipelines with different transforms
# and the same model
def model_pipelines(model, name):

    pipelines=[]
    pipelines.append({'name': name, 'pipe': make_pipeline(model)})
    pipelines.append({'name': name + "(poly2)", 'pipe': make_pipeline(PolynomialFeatures(degree=2, interaction_only=True),
    model)})

    return pipelines

# main
result = pd.DataFrame()
X, X_target, result['PassengerId'], y = get_data('train.csv', 'test.csv')

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
    best_pipeline['pipe'].fit(X, y) #re-train on full training data
    result['Survived'] = best_pipeline['pipe'].predict(X_target)
    result.to_csv('result.csv', index=False)
    joblib.dump(best_pipeline['pipe'], 'titanic.pkl')
