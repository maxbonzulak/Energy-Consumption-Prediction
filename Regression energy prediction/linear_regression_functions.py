import numpy as np
import pandas as pd
import os
from os.path import join
import glob
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.pyplot import figure
import datetime

from load_and_process_data import process_data
from sklearn import preprocessing

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


def learn_parameters(train_data, test_data, lr=0.00003, batch_size = 256 ):
    # Make a model:
    linear_regression_model = keras.models.Sequential([
        layers.Dense(units=1) # Linear Model
    ])
    
    # Choose loss function and optimization method and metric
    loss = keras.losses.MeanSquaredError() # MeanSquaredError
    #loss = keras.losses.MeanSquaredLogarithmicError()
    optim = keras.optimizers.Adam(lr=lr) 
    metrics=tf.keras.metrics.RootMeanSquaredError()
    
    # Compile model:
    linear_regression_model.compile(optimizer=optim, loss=loss)#, metrics=metrics)
    
    # Save history of learning:
    history = linear_regression_model.fit(
        train_data[0], train_data[1],
        batch_size=batch_size,
        epochs=15,
        verbose=1,
        # Calculate validation results on 10% of the training data (roughly 1 month out of 11 months for training)
        validation_split = 0.2)
    # Evaluate model on test
    linear_regression_model.evaluate(
        test_data[0],
        test_data[1], verbose=1)
    return linear_regression_model, history


def plot_loss_history(history):
    figure(figsize =(7,4))
    plt.plot(history.history['loss'], label='training', color = 'green') 
    plt.plot(history.history['val_loss'], label='validation', color = 'purple') 
    plt.xlabel('Epoch') 
    plt.ylabel('Loss') 
    plt.title('Mean squared error loss for normalized electricity consumption') 
    plt.legend() 
    plt.show()
    
    
    
def plot_test_and_prediction(test_data, timestamp_test, linear_regression_model, feature_encoding = 'polynomial'):
    test_features = test_data[0]
    test_labels = test_data[1] 
    test_features_new = test_features.copy() 
    y = linear_regression_model.predict(test_features_new) 
    y_norm= pd.DataFrame(y, columns = ['prediction']) 
    y_norm['timestamp'] = np.array(timestamp_test) 
    test_features_new['meter_reading']  = test_labels 
    test_features_new['timestamp']  = np.array(timestamp_test)   
    
    figure(figsize =(15,7)) 
    plt.plot(test_features_new.groupby(by = "timestamp").mean().filter(["timestamp", "meter_reading"]), color = 'C1', label = 'data') 
    plt.plot(y_norm.groupby(by = "timestamp").mean().filter(["timestamp", "prediction"]), color = 'gray', label = 'prediction') 
    plt.xlabel('timestamp', fontsize = 12) 
    plt.ylabel('Normalized electricity consumption', fontsize = 12) 
    plt.title('Prediction of electricity consumption for June using ' + feature_encoding + ' feature encoding', fontsize = 14) 
    len_ts = len(timestamp_test.values) 
    plt.xticks([timestamp_test.values[0],timestamp_test.values[int(len_ts/2)],timestamp_test.values[-1]], fontsize = '12') 
    plt.yticks(fontsize = 12) 
    plt.legend() 
    plt.show()
    
    
def plot_linear_regression_weights(train_data,linear_regression_model, feature_encoding = 'polynomial'): 
    train_features = train_data[0] 
    figure(figsize =(7,5)) 
    plt.bar(x = range(len(train_features.columns)),
        height=linear_regression_model.layers[0].kernel[:,0].numpy()) 
    axis = plt.gca() 
    axis.set_xticks(range(len(train_features.columns))) 
    axis.tick_params( labelsize=12) 
    _ = axis.set_xticklabels(train_features.columns, rotation=90) 
    plt.title('Linear regression weights with ' + feature_encoding + ' feature encoding ')
    plt.show()
  
  
  
  
  
  