from keras.models import load_model

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))

from flask import Flask
app = Flask(__name__)

import analysis

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/predict', method='POST')
def predict(target_data = [], features = [], labels = [], dataset_path = 'models/data.csv', model_name = 'fraud_model'):
  return analysis.predict(target_data, features, labels, dataset_path, model_name)