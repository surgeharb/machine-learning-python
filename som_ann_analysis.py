from deep_learning.minisom import MiniSom

# Importing the SK-Learn Preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Keras libraries and packages
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense

from pylab import bone, pcolor, colorbar, plot, show

def parse_columns(dataset, columns):
  dataArray = []

  for column in columns:
    data = np.array(dataset[column].tolist()).reshape(-1, 1)
    
    if len(dataArray):
      dataArray = np.concatenate((dataArray, data), axis=1)
    else:
      dataArray = data

  return dataArray

def load_data(features, labels, dataset_path):
  dataset = pd.read_csv(dataset_path)

  featuresData = parse_columns(dataset, features)
  labelsData = parse_columns(dataset, labels)

  return { "features": featuresData, "labels": labelsData }

def create_and_train_som(data, sigma = 1.0, learning_rate = 0.5):
  som = MiniSom(x = 10, y = 10, input_len = len(data[0]), sigma = sigma, learning_rate = learning_rate)
  som.random_weights_init(data)
  som.train_random(data=data, num_iteration=1000)

  return som

def visualize_som(som, dataset):
  bone()
  pcolor(som.distance_map().T)
  colorbar()
  ##markers = ['o', 's']
  ##colors = ['r', 'g']
  ##for i, x in enumerate(dataset):
    ##  w = som.winner(x)
     ## plot(w[0] + 0.5,
      ##    w[1] + 0.5,
        ##  markers[y[i]],
        ##  markeredgecolor = colors[y[i]],
        ##  markerfacecolor = 'None',
         ## markersize = 5,
         ## markeredgewidth = 2) 
  show()

def get_som_winners(som, data, threshold = 0.5):
  mappings = som.win_map(data)
  winners = []

  for i, row in enumerate(som.distance_map()):
    for j, cell in enumerate(row):
      if mappings[(i,j)] and len(mappings[(i,j)]) and cell >= threshold:
        if len(winners):
          winners = np.concatenate((winners, mappings[(i,j)]), axis=0)
        else:
          winners = mappings[(i, j)]

  return winners

def generate_label(data, winners, id_index = 0):
  is_potential_winner = np.zeros(len(data))

  for i in range(len(data)):
    if data[i, id_index] in winners:
      is_potential_winner[i] = 1

  return is_potential_winner.reshape(-1, 1)

def create_ann_classifier(features, labels, standard_scaler, batch_size = 1, epochs = 3):
  classifier = Sequential()

  # Adding the input layer and the first hidden layer
  classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(features[0])))

  # Adding the output layer
  classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

  # Compiling the ANN
  classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  
  # Feature Scaling
  features = standard_scaler.fit_transform(features)

  # Fitting the ANN to the Training set
  classifier.fit(features, labels, batch_size, epochs)

  score = classifier.evaluate(features, labels, verbose=0)
  print("TRAINED MODEL %s: %.2f%%" % (classifier.metrics_names[1], score[1] * 100))
  
  return classifier

def predict(target_data, features, labels, dataset_path, model_name, retrain):
  pred = []
  classifier = {}
  original_data = []

  model_name = 'models/' + model_name
  saved_model = model_name
  
  if retrain==True:
      saved_model = 'NONE'

  try:
    classifier = load_model(saved_model + '.h5')
  except:
    data = load_data(features, labels, dataset_path)["features"]
    original_data = data

    minMaxScaler = MinMaxScaler(feature_range=(0, 1))
    data = minMaxScaler.fit_transform(data)

    data_min = np.asarray(minMaxScaler.data_min_)
    data_max = np.asarray(minMaxScaler.data_max_)
    np.save('models/data_min', data_min)
    np.save('models/data_max', data_max)

    som = create_and_train_som(data)
    som_winners = get_som_winners(som, data)
    som_winners = minMaxScaler.inverse_transform(som_winners)
    som_label = generate_label(data, som_winners)

    visualize_som(som, data)

    standard_scaler = StandardScaler()
    classifier = create_ann_classifier(data, som_label, standard_scaler)
    classifier.save(model_name + '.h5')

    # Predicting the probabilities of frauds
    pred = classifier.predict(data)

  data_min = np.load('models/data_min.npy')
  data_max = np.load('models/data_max.npy')

  original_data = target_data
  target_data = (target_data - data_min) / ((data_max - data_min) + 1e-7)
  pred = np.concatenate((original_data, pred), axis = 1)
  pred = pred[pred[:, -1].argsort()]

  prediction = classifier.predict(target_data)
  return { "prediction": prediction, "pred": pred }

