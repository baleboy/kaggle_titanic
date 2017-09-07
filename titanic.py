import pandas as pd
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def add_title(data):
    data['Title'] = data['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())

def add_deck(data):
    data['Deck'] = data['Cabin'].map(lambda x: str(x)[0])
    data['Deck'].fillna('U', inplace = True)

def process_age(data):
    # Fill in missing age with the average on a given title/sex
    age_by_title_sex = data.groupby(['Title', 'Sex'])['Age'].mean()
    data['Age'] = data.apply(
        lambda row:
            age_by_title_sex[row['Title']][row['Sex']] if np.isnan(row['Age'])
            else row['Age'], axis=1
        )
    data['Age'] = StandardScaler().fit_transform(data['Age'].values.reshape(-1, 1))

def process_fare(data):
    fare_by_class = data.groupby(['Pclass'])['Fare'].mean()
    data['Fare'] = data.apply(
        lambda row:
            fare_by_class[row['Pclass']] if np.isnan(row['Fare'])
            else row['Fare'], axis=1
        )

    data['Fare'] = StandardScaler().fit_transform(data['Fare'].values.reshape(-1, 1))

def process_embarked(data):
    most_frequent_port = data['Embarked'].value_counts().idxmax()
    data['Embarked'].fillna(most_frequent_port, inplace = True)

def process_sibsp(data):
    data['SibSp'] = StandardScaler().fit_transform(data['SibSp'].values.reshape(-1, 1))

def process_parch(data):
    data['Parch'] = StandardScaler().fit_transform(data['Parch'].values.reshape(-1, 1))

# Pick the selected features and apply the transformations above.
# Return the feature matrix, target vector (if present) and passenger IDs.
# Works on both training and testing data.
def get_data(traindatafile, testdatafile):

    train_data = pd.read_csv(traindatafile)
    y = train_data['Survived']
    train_data.drop(['Survived'], axis=1, inplace=True)
    test_data = pd.read_csv(testdatafile)
    data = pd.concat([train_data, test_data], keys = ['train', 'test'])

    add_title(data)
    add_deck(data)
    process_age(data)
    process_fare(data)
    process_embarked(data)
    process_sibsp(data)
    process_parch(data)

    data.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
    data = pd.get_dummies(data, columns=['Pclass', 'Sex', 'Embarked', 'Title', 'Deck'])

    X_train = data.ix['train'].values
    X_test = data.ix['test'].values
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
X_train, X_test, result['PassengerId'], y = get_data('train.csv', 'test.csv')

models = [{'name': 'logreg', 'model': LogisticRegressionCV()},
          {'name': 'forest', 'model': RandomForestClassifier(n_estimators=100)},
          {'name': 'neuralnet', 'model': MLPClassifier(max_iter = 1000)},
          {'name': 'svc', 'model': SVC()}]

pipelines = []

for m in models:
    pipelines.extend(model_pipelines(m['model'], m['name']))

best_pipeline = None
best_score = 0

for p in pipelines:
    score = cross_val_score(p['pipe'], X_train, y, scoring='f1').mean()
    if score > best_score:
        best_score = score
        best_pipeline = p
    print(p['name'] + ": {}".format(score))

if (best_pipeline != None):
    print("Best pipeline: " + best_pipeline['name'])
    best_pipeline['pipe'].fit(X_train, y) #re-train on full training data
    print("Training accuracy: {}".format(best_pipeline['pipe'].score(X_train, y)))

    result['Survived'] = best_pipeline['pipe'].predict(X_test)
    result.to_csv('result.csv', index=False)
