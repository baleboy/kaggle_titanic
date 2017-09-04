import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

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

def get_data(filename):
    features = ["Pclass", "Sex", "Age", "Fare", "Parch", "SibSp", "Embarked"]
    data = pd.read_csv(filename)
    data = encode_data(data)
    data = fill_missing_data(data)
    X = data[list(features)].values
    y = data["Survived"] # Empty if not training data
    return X,y

def print_score(model, X, y):
    scores = cross_val_score(model, X, y)
    print("Cross-validation score: {}".format(scores.mean()))

# main
X,y = get_data("train.csv")

est = [('model', LogisticRegressionCV())]
pipe = Pipeline(est)
print_score(pipe, X, y)

est.insert(0, ('encode', OneHotEncoder(categorical_features = [0,6])))
pipe = Pipeline(est)
print_score(pipe, X, y)

est.insert(len(est) - 1, ('densify', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)))
est.insert(len(est) - 1, ('polynomial', PolynomialFeatures(2)))
pipe = Pipeline(est)
print_score(pipe, X, y)

est.insert(len(est) - 1, ('normalize', Normalizer()))
pipe = Pipeline(est)
print_score(pipe, X, y)
