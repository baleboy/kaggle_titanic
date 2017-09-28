from sklearn.externals import joblib
from sys import argv
import preprocessing as pp
import pandas as pd
from pdb import set_trace

def main(argv):

    helpers = joblib.load('titanic.pkl')
    data = pd.read_csv(argv[1])
    result = pd.DataFrame()
    result['PassengerId'] = data['PassengerId']
    X = pp.process_test_data(data, helpers)
    result['Survived'] = helpers['model'].predict(X)
    result.to_csv('result.csv', index=False)

    print("Prediction saved to file result.csv")
    
if __name__ == "__main__":
    main(argv)
