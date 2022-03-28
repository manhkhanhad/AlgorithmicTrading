import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from finrl import config as config_finrl

#from finrl.finrl_meta.env_stock_trading.env_stocktrading import StockTradingEnv
#from finrl.drl_agents.elegantrl.models import DRLAgent as DRLAgent_erl
from Model.elegantrl import DRLAgent as DRLAgent_erl

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

from utils.utils import read_yaml, df_to_array

def train(config):
    # Load data
    processed = pd.read_csv(config['DATA_PATH'])
    processed = processed.sort_values(['date','tic'])

    for scenario in config['SCENARIOS']:
        begin_trade, end_trade = config['PERIODS'][scenario]
        train = data_split(processed, config['BEGIN_DAY'], begin_trade)   #yyyy-mm-dd
        trade = data_split(processed, begin_trade, end_trade)

        # Process data
        train_price_array, train_tech_array, train_turbulence_array = df_to_array(train, tech_indicator_list= config_finrl.TECHNICAL_INDICATORS_LIST, if_vix= True)

        for agent_name in config['AGENTS']:
            env = StockTradingEnv

            agent = DRLAgent_erl(env = env,
                                price_array = train_price_array,
                                tech_array=train_tech_array,
                                turbulence_array=train_turbulence_array)

            model = agent.get_model(agent_name, model_kwargs = config['ERL_PARAMS'])

            trained_model = agent.train_model(model=model,
                                            cwd=config["TRAINED_MODEL_FOLDER"] + scenario + '/' + agent_name,
                                            total_timesteps=config['BREAK_STEP'])

if __name__ == "__main__":
    config_path = "config.yaml"
    config = read_yaml(config_path)
    train(config)