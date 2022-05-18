from turtle import width
from unittest import result
import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd

import plotly
import cufflinks

import plotly.express as px
import plotly.graph_objects as go
import scipy
from keras import layers
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf

from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer
from utils import read_yaml, convert_data_format, calculate_return, plot_observation_price, process_data
from LinearOptimization import portpolio_optimization
from SenarioClassification import SenarioClassifier

# def senario_classification(VNIndex_path, thresshold = 0.5):
#     VNIndex = pd.read_csv("/content/VNIndex.csv")

#     VNIndex.volumn = VNIndex.volumn.apply(lambda x: x.replace('.', '').replace('K','000').replace('M','000000'))
#     VNIndex.volumn = VNIndex.volumn.astype(int)
#     VNIndex['tic'] = ['VNIndex'] * len(VNIndex)

#     fe = FeatureEngineer(
#                         use_technical_indicator=True,
#                         tech_indicator_list = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma'],
#                         use_turbulence=True,
#                         use_vix=True,
#                         user_defined_feature = False)

#     VNIndex = fe.preprocess_data(VNIndex)

#     VNI_value = VNIndex.iloc[-15:][['close','open','high','low','volumn','macd','macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma']].to_numpy().astype(np.float64)

#     model = Sequential()
#     model.add(layers.BatchNormalization(axis = -1))
#     model.add(layers.LSTM(50, input_shape=(X_train.shape[1],X_train.shape[2])))
#     model.add(layers.Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'], )

#     model.load_weights("/content/checkpoint/")
#     scores = model.predict(np.expand_dims(VNI_value, axis=0))[0][0]
#     print("scores:", scores)
#     if scores >= thresshold:
#         print("Good - Probability = ", scores * 100)
#         return 1, scores * 100
#     else:
#         print("Bad - Probability = ", (1-scores) * 100)
#         return 0, (1-scores) * 100

def calculate_action(config):
    data = process_data(config["DATA_DIR"])

    config["NUM_TIC"] = len(data.tic.unique()) 
    data_plot = data.iloc[-config["NUM_DAY_PLOT"] * config["NUM_TIC"]:]
    observation_data = data.iloc[-config["OBSERVATION_STEPS"] * config["NUM_TIC"]:]

    observation_data = convert_data_format(observation_data)
    total_return = (observation_data[-1,1:] - observation_data[0,1:]) / observation_data[0,1:]
    return_data = calculate_return(observation_data).round(decimals=3)

    plot_observation_price(0, config["NUM_DAY_PLOT"], config["NUM_TIC"], data_plot)

    portpolio_proportion = portpolio_optimization(return_data, total_return, config["NUM_TIC"], config['LAMBDA'])

    prices = observation_data[-1,1:].reshape((config["NUM_TIC"],1))
    min_cash = (prices[0] / max(portpolio_proportion))
    stock_batch = min_cash * portpolio_proportion / prices

    stock = (config["INIT_CASH"] // (min_cash * (1 + config["BUY_FEE"]))) * stock_batch
    stock = stock // 10 * 10 #round to 10
    cash = config["INIT_CASH"] - np.dot(stock.T, prices) * (1+config["BUY_FEE"]) #buy stocks
    trading_fee = np.dot(stock.T, prices) * (config["BUY_FEE"]) + np.dot(stock.T, prices) * (config["BUY_FEE"])

    action = []
    for tic, stock in zip(data.tic.unique(), stock[:,0]):
        if stock > 0:
            print( "BUY {}: {} shares".format(tic,stock))
            action.append("BUY {}: {} shares".format(tic,stock))
    action
    
if __name__ == '__main__':
    config_path = "config_LP.yaml"
    config = read_yaml(config_path)
    classifier = SenarioClassifier(config, False)
    senario, score = classifier.predict()
    if senario == 0:
        print("Market is downtrend - Probability = ", score)
    else:
        action = calculate_action(config)
