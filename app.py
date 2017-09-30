from flask import Flask
from flask import request
import pandas as pd
from sklearn.externals import joblib

import preprocessing as pp # local module

app = Flask(__name__)

@app.route('/isAlive')
def index():
    return "true"

@app.route('/titanic/predict', methods=['GET', 'POST'])

# {
#   name: "Caio, Mr. Tizio",
#   class: 2,
#   sex: "male",
#   age: 42,
#   sibsp: 2,
#   parch: 1,
#   fare: 23.45,
#   cabin: "A123",
#   embarked: "Q"
# }
def get_prediction():

    helpers = joblib.load('titanic.pkl')
    model = helpers['model']

    data = pd.DataFrame(request.json, index=[0])

    X = pp.process_test_data(data, helpers)
    survived = model.predict(X)

    return 'yes' if survived else 'no'

if __name__ == '__main__':
    app.run(port=5000,host='0.0.0.0')
