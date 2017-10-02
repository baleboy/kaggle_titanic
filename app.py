from flask import Flask, jsonify
from flask import request
import pandas as pd
from sklearn.externals import joblib

import preprocessing as pp # local module

app = Flask(__name__)

@app.route('/isAlive')
def index():
    return "true"

@app.route('/titanic/api/v1.0/survived', methods=['GET'])

def get_prediction():

    helpers = joblib.load('titanic.pkl')
    model = helpers['model']

    passenger = {};

    passenger['Name'] = request.args.get('n')
    passenger['Sex'] = request.args.get('s')
    passenger['Age'] = int(float(request.args.get('a')))
    passenger['Fare'] = float(request.args.get('f'))
    passenger['Pclass'] = int(float(request.args.get('c')))
    passenger['SibSp'] = int(float(request.args.get('si')))
    passenger['Parch'] = int(float(request.args.get('p')))
    passenger['Embarked'] = request.args.get('e')
    passenger['Cabin'] = request.args.get('ca')

    data = pd.DataFrame(passenger, index=[0])

    X = pp.process_test_data(data, helpers)
    survived = model.predict(X)

    survived = 'yes' if survived else 'no'
    return jsonify({'survived': survived})

if __name__ == '__main__':
    app.run(port=5000,host='0.0.0.0')
