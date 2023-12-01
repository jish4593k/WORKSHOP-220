# Import necessary libraries for machine learning and data mining
import os
import json
import time
import math
import matplotlib.pyplot as plt
import numpy as np  # Adding NumPy for array manipulation
import pandas as pd  # Adding pandas for data manipulation
from sklearn.model_selection import train_test_split  # Adding scikit-learn for machine learning

# Import libraries for deep learning with TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

# Import custom modules for data processing and modeling
from core.data_processor import DataLoader
from core.model import Model

# Import libraries for data visualization
import matplotlib.pyplot as plt

# Import libraries for AI and machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import libraries for GUI and turtle graphics
import turtle
from tkinter import Tk, Label, Button

# Define a function to plot the predicted and true data
def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

# Define a function to plot multiple sets of predicted data along with true data
def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to its correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

# Main function for training and testing the model
def main():
    # Load configurations from a JSON file
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']):
        os.makedirs(configs['model']['save_dir'])

    # Load and preprocess data using a custom DataLoader
    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    # Create an instance of the Model class
    model = Model()
    model.build_model(configs)
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    # Out-of-memory generative training using a generator
    steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
            normalise=configs['data']['normalise']
        ),
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_dir=configs['model']['save_dir']
    )

    # Get test data
    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    # Generate predictions using the trained model
    predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])

    # Plot multiple sets of predicted data along with true data
    plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])

if __name__ == '__main__':
    main()
