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
from sys import argv
import preprocessing as pp # Local module
from pdb import set_trace

def get_data(traindatafile):

    data = pd.read_csv(traindatafile)
    y = data['Survived']
    data.drop(['Survived'], axis=1, inplace=True)

    X, helpers = pp.process_training_data(data)

    return X, y, helpers

def main(argv):
    trainfile = argv[1]
    X, y, helpers = get_data(trainfile)

    models = [{'name': 'logreg', 'model': LogisticRegressionCV()},
              {'name': 'logreg + poly2', 'model': make_pipeline(PolynomialFeatures(degree=2, interaction_only=True), LogisticRegressionCV())},
              {'name': 'forest', 'model': RandomForestClassifier(n_estimators=100)},
              {'name': 'neuralnet', 'model': MLPClassifier(max_iter = 1000)},
              {'name': 'svc', 'model': SVC()}]

    best_model = None
    best_score = 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state = 42)

    for m in models:
        print("*** " + m['name'])

        m['model'].fit(X_train, y_train)
        print("Training score: {}".format(m['model'].score(X_train, y_train)))
        cv_score = m['model'].score(X_test, y_test)
        if cv_score > best_score:
            best_score = cv_score
            best_model = m
        print("CV score: {}".format(cv_score))

    if (best_model != None):
        print("Best model: " + best_model['name'] + ", score: {}".format(best_score))
        helpers['model'] = best_model['model']
        joblib.dump(helpers, 'titanic.pkl')

if __name__ == "__main__":
    main(argv)
