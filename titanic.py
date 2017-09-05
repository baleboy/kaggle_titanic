import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def fill_missing_data(data):
    filled_data = data
    filled_data['Age'].fillna(filled_data['Age'].mean(), inplace = True)
    filled_data['Fare'].fillna(filled_data['Fare'].mean(), inplace = True)
    filled_data['Embarked'].fillna(0, inplace = True)
    return filled_data

def encode_data(data):
    encoded_data = data
    encoded_data['Sex'] = encoded_data['Sex'].map({'male':1,'female':0})
    encoded_data['Embarked'] = encoded_data['Embarked'].map({'S':0,'C':1, 'Q':2})
    return encoded_data

# Pick the selected features and apply the transformations above.
# Return the feature matrix, target vector (if present) and passenger IDs.
# Works on both training and testing data.
def get_prepared_data(filename):
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Parch', 'SibSp', 'Embarked']
    data = pd.read_csv(filename)
    data = encode_data(data)
    data = fill_missing_data(data)
    pid = data['PassengerId']
    if 'Survived' in data.columns:
        y = data['Survived']
    else:
        y = None
    X = data[list(features)].values
    return X,y,pid

# Return a list of pipelines with different transforms
# and the same model
def model_pipelines(model, name):

    pipelines=[]
    pipelines.append({'name': name, 'pipe': make_pipeline(model)})
    pipelines.append({'name': name + "(enc)", 'pipe': make_pipeline(OneHotEncoder(categorical_features = [0,6]),
                                   model)})
    pipelines.append({'name': name + "(enc, poly)", 'pipe': make_pipeline(OneHotEncoder(categorical_features = [0,6]),
                                   FunctionTransformer(lambda x: x.todense(), accept_sparse=True),
                                   PolynomialFeatures(degree=2, interaction_only=True),
                                   model)})
    pipelines.append({'name': name + "(enc, poly, scale)", 'pipe': make_pipeline(OneHotEncoder(categorical_features = [0,6]),
                                   FunctionTransformer(lambda x: x.todense(), accept_sparse=True),
                                   PolynomialFeatures(degree=2, interaction_only=True),
                                   StandardScaler(),
                                   model)})
    return pipelines

# main
X,y,pid = get_prepared_data('train.csv')

models = [{'name': 'logreg', 'model': LogisticRegressionCV()},
          {'name': 'dectree', 'model': DecisionTreeClassifier()},
          {'name': 'forest', 'model': RandomForestClassifier(n_estimators=100)},
          {'name': 'neuralnet', 'model': MLPClassifier(max_iter = 1000)}]

pipelines = []

for m in models:
    pipelines.extend(model_pipelines(m['model'], m['name']))

best_pipeline = None
best_score = 0

for p in pipelines:
    score = cross_val_score(p['pipe'], X, y).mean()
    if score > best_score:
        best_score = score
        best_pipeline = p
    print(p['name'] + ": {}".format(score))

if (best_pipeline != None):
    print("Best pipeline: " + best_pipeline['name'])
    best_pipeline['pipe'].fit(X, y) #re-train on full training data
    result = pd.DataFrame()
    print("Training accuracy: {}".format(best_pipeline['pipe'].score(X, y)))
    X_test, y_hat, result['PassengerId'] = get_prepared_data('test.csv')
    result['Survived'] = best_pipeline['pipe'].predict(X_test)
    result.to_csv('result.csv', index=False)
