import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from finrl import config

#from finrl.finrl_meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.drl_agents.elegantrl.models import DRLAgent as DRLAgent_erl
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.finrl_meta.data_processor import DataProcessor
from model.enviroment import StockTradingEnv
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from pprint import pprint
import sys
sys.path.append("../FinRL-Library")
import itertools
import numpy as np
import os
import gym

from utils.utils import read_yaml, df_to_array
from utils.visualize import visualize
def test(config):
    # Load data
    processed = pd.read_csv(config['DATA_PATH'])
    processed = processed.sort_values(['date','tic'])

    for scenario in config['SCENARIOS']:
        #Process data
        print("Testing scenario: ", scenario)
        begin_trade, end_trade = config['PERIODS'][scenario]
        train = data_split(processed, config['BEGIN_DAY'], begin_trade)   #yyyy-mm-dd
        trade = data_split(processed, begin_trade, end_trade)
        test_price_array, test_tech_array, test_turbulence_array = df_to_array(trade, tech_indicator_list= config['TECHNICAL_INDICATORS_LIST'], if_vix= True)

        #Create Environment
        env_config = {'price_array':test_price_array,
                'tech_array':test_tech_array,
                'turbulence_array':test_turbulence_array,
                'if_train':False}
        env = StockTradingEnv
        env_instance = env(config=env_config)

        for agent_name in config['AGENTS']:
            #Test trained model
            episode_total_assets = DRLAgent_erl.DRL_prediction(model_name=agent_name,
                                                    cwd=config["TRAINED_MODEL_FOLDER"] + scenario + '/' + agent_name,
                                                    net_dimension=config['ERL_PARAMS']['net_dimension'],
                                                    environment=env_instance)
        
            date_list = pd.DataFrame(trade['date'].unique(), columns=['date'])
            account_value_erl = pd.DataFrame({'date':date_list['date'],'account_value':episode_total_assets[0:len(episode_total_assets)]})

            #Save the account value during trading period
            os.makedirs(config["RESULT_FOLDER"] + scenario + '/' + agent_name, exist_ok=True)
            account_value_erl.to_csv(config["RESULT_FOLDER"] + scenario + '/' + agent_name + "/account_value.csv")

            #Print the results
            print("==============Get Backtest Results of {}===========".format(agent_name))
            now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

            perf_stats_all = backtest_stats(account_value=account_value_erl)
            perf_stats_all = pd.DataFrame(perf_stats_all)
        
    #Visulize the results
        visualize(config["RESULT_FOLDER"] + '/' + scenario, config['AGENTS'], with_VNI= True)

if __name__ == "__main__":
    config_path = "config.yaml"
    config = read_yaml(config_path)
    test(config)