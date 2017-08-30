import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

def cleanup_data(data):
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Fare'].fillna(data['Fare'].mean(), inplace=True)
    data['Sex'] = data['Sex'].map({'male':1,'female':0})
    data['Embarked'] = data['Embarked'].map({'S':0,'C':1, 'Q':2})
    data['Embarked'].fillna(0, inplace=True)

def train(model, data, features):
    X = data[list(features)].values
    y = train_data['Survived']
    model = model.fit(X, y)
    print("Training accuracy: {}".format(model.score(X, y)))
    return model

def predict_result(model, data, features):
    X = data[list(features)].values
    y = model.predict(X)
    result = pd.DataFrame()
    result["PassengerId"] = data["PassengerId"]
    result["Survived"] = y
    return result


train_data = pd.read_csv('train.csv')
cleanup_data(train_data)
test_data = pd.read_csv('test.csv')
cleanup_data(test_data)
model = LogisticRegression()

features = ["Pclass", "Sex", "Age", "Fare"]

print("*** Step 1. Features: " +", ".join(features))

model = train(model, train_data, features)
result = predict_result(model, test_data, features)
result.to_csv("step1_result.csv", index=False)

features = ["Pclass", "Sex", "Age", "Fare", "Parch", "SibSp", "Embarked"]

print("*** Step 2. Features: " +", ".join(features))

model = train(model, train_data, features)
result = predict_result(model, test_data, features)
result.to_csv("step2_result.csv", index=False)
