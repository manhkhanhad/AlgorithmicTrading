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
from Model.enviroment import StockTradingEnv

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from pprint import pprint
import sys
sys.path.append("../FinRL-Library")
import itertools
import numpy as np
import os
import gym

from Utils.utils import read_yaml, df_to_array

def train_elegantrl(config):
    # Load data
    processed = pd.read_csv(config['DATA_PATH'])
    processed = processed.sort_values(['date','tic'])
    
    for scenario in config['SCENARIOS']:
        begin_trade, end_trade = config['PERIODS'][scenario]
        train = data_split(processed, config['BEGIN_DAY'], begin_trade)   #yyyy-mm-dd
        trade = data_split(processed, begin_trade, end_trade)
        # Process data

        for agent_name in config['AGENTS']:

            env = StockTradingEnv
            
            train_price_array, train_tech_array, train_turbulence_array = df_to_array(train, tech_indicator_list= config_finrl.TECHNICAL_INDICATORS_LIST, if_vix= True)
            
            agent = DRLAgent_erl(env = env,
                                price_array = train_price_array,
                                tech_array=train_tech_array,
                                turbulence_array=train_turbulence_array,
                                config = config)
            model = agent.get_model(agent_name, model_kwargs = config['ERL_PARAMS'])

            trained_model = agent.train_model(model=model,
                                            cwd=config["TRAINED_MODEL_FOLDER"] + scenario + '/' + agent_name,
                                            total_timesteps=config['BREAK_STEP'])

def train_stable_baselines(config):
    # Load data
    processed = pd.read_csv(config['DATA_PATH'])
    processed = processed.sort_values(['date','tic'])
    for scenario in config['SCENARIOS']:
        begin_trade, end_trade = config['PERIODS'][scenario]
        train = data_split(processed, config['BEGIN_DAY'], begin_trade)   #yyyy-mm-dd
        trade = data_split(processed, begin_trade, end_trade)
        # Process data
        for agent_name in config['AGENTS']:

            agent = DRLAgent_stablebaselines(df = train, env = StockTradingEnv, config = config)
            model = agent.get_model(agent_name, model_kwargs = config['SB3_PARAMS'])

def train_rllib(config):
    # Load data
    processed = pd.read_csv(config['DATA_PATH'])
    processed = processed.sort_values(['date','tic'])
    
    for scenario in config['SCENARIOS']:
        begin_trade, end_trade = config['PERIODS'][scenario]
        train = data_split(processed, config['BEGIN_DAY'], begin_trade)   #yyyy-mm-dd
        trade = data_split(processed, begin_trade, end_trade)
        # Process data

        for agent_name in config['AGENTS']:

            env = StockTradingEnv
            
            train_price_array, train_tech_array, train_turbulence_array = df_to_array(train, tech_indicator_list= config_finrl.TECHNICAL_INDICATORS_LIST, if_vix= True)
            
            agent = DRLAgent_rllib(env = env,
                                price_array = train_price_array,
                                tech_array=train_tech_array,
                                turbulence_array=train_turbulence_array)
            model, model_config = agent.get_model(agent_name)

            trained_model = agent.train_model(model=model,
                                            model_name=agent_name,
                                            cwd = config["TRAINED_MODEL_FOLDER"] + scenario + '/' + agent_name,
                                            model_config=model_config)


if __name__ == "__main__":
    config_path = "config.yaml"
    config = read_yaml(config_path)
    if config['RLLIB'] == "elegantrl":
        train_elegantrl(config)
    elif config['RLLIB'] == "stable_baselines":
        train_stable_baselines(config)
    elif config['RLLIB'] == "ray":
        train_rllib(config)
    else:
        raise ValueError("Please choose elegantrl or stable_baselines or ray")