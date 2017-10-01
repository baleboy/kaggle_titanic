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

    data = pd.DataFrame(request.json, index=[0])

    X = pp.process_test_data(data, helpers)
    survived = model.predict(X)

    survived = 'yes' if survived else 'no'
    return jsonify({'survived': survived})

if __name__ == '__main__':
    app.run(port=5000,host='0.0.0.0')
