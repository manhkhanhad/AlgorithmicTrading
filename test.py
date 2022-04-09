import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from finrl import config

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
from Utils.visualize import visualize, visualize_trading_action
def test_elegantrl(config):
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
            episode_total_assets, episode_sell_buy, rewards, action_values = DRLAgent_erl.DRL_prediction(model_name=agent_name,
                                                    cwd=config["TRAINED_MODEL_FOLDER"] + scenario + '/' + agent_name,
                                                    net_dimension=config['ERL_PARAMS']['net_dimension'],
                                                    environment=env_instance, devices = config['ERL_PARAMS']['learner_gpus'],)
        
            date_list = pd.DataFrame(trade['date'].unique(), columns=['date'])
            account_value_erl = pd.DataFrame({'date':date_list['date'],'account_value':episode_total_assets[0:len(episode_total_assets)]})
            action_values_pd = pd.DataFrame({'date':date_list['date'],'-1':action_values[:,0],'-0.5':action_values[:,1],'0':action_values[:,2],'0.5':action_values[:,3],'1':action_values[:,4]})

            print("len rewards: ", len(rewards))

            episode_sell_buy_df = {'date':date_list['date'], 'reward': rewards}

            for i,tic in enumerate(trade['tic'].unique()):
                episode_sell_buy_df[tic] = episode_sell_buy[:,i]
                print(len(episode_sell_buy_df[tic]))
            episode_sell_buy_df = pd.DataFrame(episode_sell_buy_df)

            #Save the account value during trading period
            os.makedirs(config["RESULT_FOLDER"] + scenario + '/' + agent_name, exist_ok=True)
            account_value_erl.to_csv(config["RESULT_FOLDER"] + scenario + '/' + agent_name + "/account_value.csv")
            episode_sell_buy_df.to_csv(config["RESULT_FOLDER"] + scenario + '/' + agent_name + "/sell_buy.csv")
            action_values_pd.to_csv(config["RESULT_FOLDER"] + scenario + '/' + agent_name + "/action_values.csv")
            #Print the results
            print("==============Get Backtest Results of {}===========".format(agent_name))
            now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

            perf_stats_all = backtest_stats(account_value=account_value_erl)
            perf_stats_all = pd.DataFrame(perf_stats_all)
        
    #Visulize the results
        visualize(config,scenario, with_Baseline= True)

        if config['VISUALIZE_TRADING_ACTION']:
            visualize_trading_action(begin_trade, end_trade,scenario,config)

def test_rllib(config):
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

        for agent_name in config['AGENTS']:
            #Test trained model
            env = StockTradingEnv
            agent_path= config["TRAINED_MODEL_FOLDER"] + scenario + '/' + agent_name + "/checkpoint_000100/checkpoint-100"
            episode_total_assets, episode_sell_buy, rewards, action_values = DRLAgent_rllib.DRL_prediction(agent_name,env, test_price_array, test_tech_array, test_turbulence_array,agent_path)
        
            date_list = pd.DataFrame(trade['date'].unique(), columns=['date'])
            account_value_erl = pd.DataFrame({'date':date_list['date'],'account_value':episode_total_assets[0:len(episode_total_assets)]})
            #action_values_pd = pd.DataFrame({'date':date_list['date'],'-1':action_values[:,0],'-0.5':action_values[:,1],'0':action_values[:,2],'0.5':action_values[:,3],'1':action_values[:,4]})
            
            #Save the action of agent
            episode_sell_buy_dict = {'date':date_list['date']}
            for i, tic in enumerate(trade['tic'].unique()):
                episode_sell_buy_dict[tic] = episode_sell_buy[:,i]
            episode_sell_buy_df = pd.DataFrame(episode_sell_buy_dict)

            #Save the account value during trading period
            os.makedirs(config["RESULT_FOLDER"] + scenario + '/' + agent_name, exist_ok=True)
            account_value_erl.to_csv(config["RESULT_FOLDER"] + scenario + '/' + agent_name + "/account_value.csv")
            episode_sell_buy_df.to_csv(config["RESULT_FOLDER"] + scenario + '/' + agent_name + "/sell_buy.csv")
            #action_values_pd.to_csv(config["RESULT_FOLDER"] + scenario + '/' + agent_name + "/action_values.csv")
            #Print the results
            print("==============Get Backtest Results of {}===========".format(agent_name))
            now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

            perf_stats_all = backtest_stats(account_value=account_value_erl)
            perf_stats_all = pd.DataFrame(perf_stats_all)
        
    #Visulize the results
        visualize(config,scenario, with_Baseline= True)

        if config['VISUALIZE_TRADING_ACTION']:
            visualize_trading_action(begin_trade, end_trade,scenario,config)
        
if __name__ == "__main__":
    config_path = "config.yaml"
    config = read_yaml(config_path)
    if config['RLLIB'] == "elegantrl":
        test_elegantrl(config)
    #elif config['RLLIB'] == "stable_baselines":
    #    test_stable_baselines(config)
    elif config['RLLIB'] == "ray":
        test_rllib(config)
    else:
        raise ValueError("Please choose elegantrl or stable_baselines or ray")