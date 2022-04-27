from ast import While
from distutils.command.config import config
from tkinter import W
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from finrl import config as config_finrl

#from finrl.finrl_meta.env_stock_trading.env_stocktrading import StockTradingEnv
#from finrl.drl_agents.elegantrl.models import DRLAgent as DRLAgent_erl

from Model.elegantrlAgent import DRLAgent as DRLAgent_erl
from Model.stablebaselines3 import DRLAgent as DRLAgent_stablebaselines
from Model.rllib import DRLAgent as DRLAgent_rllib


from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.finrl_meta.data_processor import DataProcessor
from Model.enviroment import StockTradingEnv, RealtimeTradingEnv
from Utils.visualize import visualize, visualize_trading_action
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from pprint import pprint
import sys
sys.path.append("../FinRL-Library")
import itertools
import numpy as np
import os
import gym
from datetime import datetime

from Utils.utils import read_yaml, df_to_array, update_realtime_data
from Model.rllib import get_trained_agent

def main(config):
    processed = pd.read_csv(config["DATA_PATH"])
    processed = processed.sort_values(['date','tic'])
    
    num_updated_data = 0

    #Setup Agent and Enviroment:
    price_array, tech_array, turbulence_array = df_to_array(processed, config["TECHNICAL_INDICATORS_LIST"], if_vix= True)
    print("price_array: ", price_array[-2:,:].shape)
    print("tech_array: ", tech_array[-2:,:].shape)
    print("turbulence_array: ", turbulence_array[-2:].shape)
    env_config = {'price_array':price_array[-2:,:],
                'tech_array':tech_array[-2:,:],
                'turbulence_array':turbulence_array[-2:],
                'if_train':False}

    env_instance = RealtimeTradingEnv(config=env_config)
    state = env_instance.reset()
    max_episode = max(os.listdir(config["TRAINED_MODEL_FOLDER"] + 'ddpg'))
    max_episode = int(max_episode.split('_')[-1])
    agent_path= config["TRAINED_MODEL_FOLDER"] + 'ddpg' + "/checkpoint_{}/checkpoint-{}".format("0"*(6-len(str(max_episode))) + str(max_episode), str(max_episode))

    agent = get_trained_agent('ddpg', RealtimeTradingEnv, agent_path, price_array, tech_array, turbulence_array)
    while True:
        #Trading
        action = agent.compute_single_action(state)
        state, reward, done, sell_buy_actions, _ = env_instance.step(action)
        print(sell_buy_actions)


        #Get current date_time
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        if current_time > "15:00:00":
            update_realtime_data()
            num_updated_data += 1
        
        if num_updated_data > 0:
            #Continue training agent with new updated data
            pass
        
        break

if __name__ == '__main__':
    config_path = "config.yaml"
    config = read_yaml(config_path)
    main(config)


