from xml.dom.minidom import Document
import yaml
import numpy as np
import os

def read_yaml(config_weight):
    try:
        with open(config_weight) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except ValueError:
        print("No config file!")
    return config

def convert_data_format(data):
    converted_data = data.date.unique()
    for tic in data.tic.unique():
        converted_data = np.vstack((converted_data, data[data.tic == tic]['close'].to_numpy()))
    return converted_data.T

def calculate_return(data):
    data = data[:,1:] #Drop date column
    return (np.diff(data, axis = 0) / data[1,:]).astype(np.float32)

def plot_observation_price(t, observation_steps,num_tic, data):
    data_plot = data.iloc[t * observation_steps * num_tic: (t+1) * observation_steps * num_tic,:]
    fig = px.line(data_plot,x='date',y='close', title='', color='tic', width=2048, height=780)
    fig.write_image('price.png')