import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def process_data(data):
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

    return X_train, X_test

def process_training_data(data):
    add_title(data)
    add_deck(data)
    process_age(data)
    process_fare(data)
    process_embarked(data)
    process_sibsp(data)
    process_parch(data)

    data.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
    data = pd.get_dummies(data, columns=['Pclass', 'Sex', 'Embarked', 'Title', 'Deck'])

    return data.values

def add_title(data):
    data['Title'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

def add_deck(data):
    data['Deck'] = data['Cabin'].apply(lambda x: str(x)[0])
    data['Deck'].fillna('U', inplace = True)

def process_age(data):
    # Fill in missing age with the average on a given title/sex
    age_by_title = data.groupby(['Title'])['Age'].mean()
    data['Age'] = data.apply(
        lambda row:
            age_by_title[row['Title']] if np.isnan(row['Age'])
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
