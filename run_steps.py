import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

def cleanup_data_step1(data):
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Fare'].fillna(data['Fare'].mean(), inplace=True)
    data['Sex'] = data['Sex'].map({'male':1,'female':0})

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

features = ["Pclass", "Sex", "Age", "Fare"]

print("*** Step 1. Features: " +", ".join(features))

cleanup_data_step1(train_data)

# Extract feature matrix and target vector
X_train = train_data[list(features)].values
y_train = train_data['Survived']
model = LogisticRegression()
model = model.fit(X_train, y_train)
print("Training accuracy: {}".format(model.score(X_train, y_train)))

cleanup_data_step1(test_data)

X_test = test_data[list(features)].values
y_test = model.predict(X_test)
result = pd.DataFrame()
result["PassengerId"] = test_data["PassengerId"]
result["Survived"] = y_test
result.to_csv("step1_result.csv", index=False)
