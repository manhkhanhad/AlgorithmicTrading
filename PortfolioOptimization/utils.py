from xml.dom.minidom import Document
import yaml
import numpy as np
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

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

def visualize_action(data_raw, actions, config):
    plot_data = data_raw[(data_raw.date >= config['START_DAY']) & (data_raw.date <= config['END_DAY'])]
    date = list(plot_data.date.unique())

    fig = make_subplots()
    max_prices = 0
    for tic in plot_data.tic.unique():
        tic_price_historical = list(plot_data[plot_data.tic == tic]['close'])
        fig.add_trace( go.Scatter(x=date, y=tic_price_historical, name=tic), secondary_y=False,)
        max_prices = max(max_prices, max(tic_price_historical))

    annotations = []
    color = 'DarkGreen'
    for date, cash, stock, return_value, is_profit, portfolio_value in actions:
        
        hovertext = 'Date: {}<br>' \
                'Rerturn: {} <br>' \
                'Portfolio Value: {} <br>' \
                'Cash: {} <br>'.format(date, return_value, portfolio_value, cash)
        for index, tic in enumerate(plot_data.tic.unique()):
            hovertext += '{}: {}<br>'.format(tic,stock[index][0])
        
        if is_profit:
            color = 'DarkGreen'
            ay = 15
        else:
            color = 'FireBrick'
            ay = -15

        annotations += [go.layout.Annotation(x = date, y = max_prices + 0.25 * ay, 
                                            ax=0, ay= ay, xref='x1', yref='y1',showarrow=True,
                                            arrowhead=2, arrowcolor=color, arrowwidth=4,
                                            arrowsize=0.8, hovertext=hovertext, opacity=0.6,
                                            hoverlabel=dict(bgcolor=color))]

    fig.layout.annotations += tuple(annotations)
    #fig.show()
    fig.write_image(config['SAVE_DIR'] + '/action.png')
    fig.write_html(config['SAVE_DIR'] + '/action.html')
