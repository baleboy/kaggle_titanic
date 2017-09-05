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

def add_model_pipelines(pipelines, model):
    pipelines.append(make_pipeline(model))
    pipelines.append(make_pipeline(OneHotEncoder(categorical_features = [0,6]),
                                   model))
    pipelines.append(make_pipeline(OneHotEncoder(categorical_features = [0,6]),
                                   FunctionTransformer(lambda x: x.todense(), accept_sparse=True),
                                   PolynomialFeatures(2),
                                   model))
    pipelines.append(make_pipeline(OneHotEncoder(categorical_features = [0,6]),
                                   FunctionTransformer(lambda x: x.todense(), accept_sparse=True),
                                   PolynomialFeatures(2),
                                   StandardScaler(),
                                   model))
# main
X,y,pid = get_prepared_data('train.csv')

models = [LogisticRegressionCV(),
          DecisionTreeClassifier(),
          RandomForestClassifier(n_estimators=100),
          MLPClassifier(max_iter = 1000)]

pipelines = []

for model in models:
    add_model_pipelines(pipelines, model)

best_pipeline = None
best_score = 0

for pipe in pipelines:
    score = cross_val_score(pipe, X, y).mean()
    if score > best_score:
        best_score = score
        best_pipeline = pipe
    print("Cross-validation score: {}".format(score))

if (best_pipeline != None):
    print(best_pipeline)
    best_pipeline.fit(X, y) #re-train on full training data
    result = pd.DataFrame()
    print("Cross-validation score: {}".format(cross_val_score(best_pipeline, X, y).mean()))
    X_test, y_hat, result['PassengerId'] = get_prepared_data('test.csv')
    result['Survived'] = best_pipeline.predict(X_test)
    result.to_csv('result.csv', index=False)
