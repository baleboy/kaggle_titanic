import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

def cleanup_data(data):

    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Fare'].fillna(data['Fare'].mean(), inplace=True)
    data['Sex'] = data['Sex'].map({'male':1,'female':0})
    data['Embarked'] = data['Embarked'].map({'S':0,'C':1, 'Q':2})
    data['Embarked'].fillna(0, inplace=True)

def train(model, X, y):
    model = model.fit(X, y)
    print("Training accuracy: {}".format(model.score(X, y)))
    return model

def predict_result(model, X):
    y_hat = model.predict(X)
    return y_hat

train_data = pd.read_csv('train.csv')
cleanup_data(train_data)
test_data = pd.read_csv('test.csv')
cleanup_data(test_data)

model = LogisticRegression()
target = train_data["Survived"]
result = pd.DataFrame()
result["PassengerId"] = test_data["PassengerId"]

print("*** Step 0. Predict no survivors")

result["Survived"] = 0
result.to_csv("step0_result.csv", index=False)

features = ["Pclass", "Sex", "Age", "Fare"]

print("*** Step 1. Features: " +", ".join(features))

model = train(model, train_data[list(features)].values, target)
result["Survived"] = predict_result(model, test_data[list(features)].values)
result.to_csv("step1_result.csv", index=False)

features = ["Pclass", "Sex", "Age", "Fare", "Parch", "SibSp", "Embarked"]

print("*** Step 2. Features: " +", ".join(features))

model = train(model, train_data[list(features)].values, target)
result["Survived"] = predict_result(model, test_data[list(features)].values)
result.to_csv("step1_result.csv", index=False)

print("*** Step 3. Polynomial features (square)")

poly = PolynomialFeatures(2)
X_train = train_data[list(features)].values
X_train = poly.fit_transform(X_train)
model = train(model, X_train, target)

X_test = test_data[list(features)].values
X_test = poly.fit_transform(X_test)
result["Survived"] = predict_result(model, X_test)
result.to_csv("step3_result.csv", index=False)
