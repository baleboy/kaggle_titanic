import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pdb import set_trace

def process_training_data(data):

    helpers = {}

    add_title(data)
    add_deck(data)
    helpers['age_by_title'], helpers['age_scaler'] = process_age(data)
    helpers['fare_by_class'], helpers['fare_scaler'] = process_fare(data)
    helpers['default_port'] = process_embarked(data)
    helpers['sibsp_scaler'] = process_sibsp(data)
    helpers['parch_scaler'] = process_parch(data)

    data.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
    data = pd.get_dummies(data, columns=['Pclass', 'Sex', 'Embarked', 'Title', 'Deck'])
    helpers['columns'] = data.columns
    return data.values, helpers

def process_test_data(data, helpers):

    add_title(data)
    add_deck(data)
    process_age(data, helpers['age_by_title'], helpers['age_scaler'])
    process_fare(data, helpers['fare_by_class'], helpers['fare_scaler'])
    process_embarked(data, helpers['default_port'])
    process_sibsp(data, helpers['sibsp_scaler'])
    process_parch(data, helpers['parch_scaler'])

    data.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True, errors='ignore')
    data = pd.get_dummies(data, columns=['Pclass', 'Sex', 'Embarked', 'Title', 'Deck'])

    # Fill in missing dummy values if needed
    trained_cols = helpers['columns']
    missing_cols = set( trained_cols ) - set( data.columns )
    for c in missing_cols:
        data[c] = 0
    #reorder columns to match training set
    data = data[ trained_cols]

    return data.values

def add_title(data):
    data['Title'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

def add_deck(data):
    data['Deck'] = data['Cabin'].apply(lambda x: str(x)[0])
    data['Deck'].fillna('U', inplace = True)

def add_missing_ages(data, age_by_title):
    data['Age'] = data.apply(
        lambda row:
            age_by_title[row['Title']] if np.isnan(row['Age'])
            else row['Age'], axis=1
        )

def process_age(data, age_by_title = None, scaler = None):
    # Fill in missing age with the average on a given title/sex
    if age_by_title is None:
        age_by_title = data.groupby(['Title'])['Age'].mean()
    add_missing_ages(data, age_by_title)
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(data['Age'].values.reshape(-1, 1))

    data['Age'] = scaler.transform(data['Age'].values.reshape(-1, 1))

    return age_by_title, scaler

def add_missing_fares(data, fare_by_class):
    data['Fare'] = data.apply(
        lambda row:
            fare_by_class[row['Pclass']] if np.isnan(row['Fare'])
            else row['Fare'], axis=1
        )

def process_fare(data, fare_by_class = None, scaler = None):
    if fare_by_class is None:
        fare_by_class = data.groupby(['Pclass'])['Fare'].mean()
    add_missing_fares(data, fare_by_class)
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(data['Fare'].values.reshape(-1, 1))
    data['Fare'] = scaler.transform(data['Fare'].values.reshape(-1, 1))
    return fare_by_class, scaler

def process_embarked(data, default_port = None):
    if default_port is None:
        default_port =  data['Embarked'].value_counts().idxmax()
    data['Embarked'].fillna(default_port, inplace = True)
    return default_port

def process_sibsp(data, scaler = None):
    if scaler is None:
        scaler = StandardScaler().fit(data['SibSp'].values.reshape(-1, 1))
    data['SibSp'] = scaler.transform(data['SibSp'].values.reshape(-1, 1))
    return scaler

def process_parch(data, scaler = None):
    if scaler is None:
        scaler = StandardScaler().fit(data['Parch'].values.reshape(-1, 1))
    data['Parch'] = scaler.transform(data['Parch'].values.reshape(-1, 1))
    return scaler
