from ast import While
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
from Utils.visualize import visualize, visualize_trading_action
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from pprint import pprint
import sys
sys.path.append("../FinRL-Library")
import itertools
import numpy as np
import os
import gym

from Utils.utils import read_yaml, df_to_array
from Model.rllib import get_action

def train(agent_name,config, train_data, t, total_trained_episode):
    env = StockTradingEnv

    train_price_array, train_tech_array, train_turbulence_array = df_to_array(train_data, tech_indicator_list= config_finrl.TECHNICAL_INDICATORS_LIST, if_vix= True)
    
    agent = DRLAgent_rllib(env = env,
                        price_array = train_price_array,
                        tech_array=train_tech_array,
                        turbulence_array=train_turbulence_array,
                        config=config)
    model, model_config = agent.get_model(agent_name)

    if t == 0:
        trained_agent_path = None # if t == 0 mean this is the first time to train
    else:
        trained_agent_path = config["TRAINED_MODEL_FOLDER"] + agent_name + "/checkpoint_{}/checkpoint-{}".format("0"*(6-len(str(total_trained_episode))) + str(total_trained_episode), str(total_trained_episode))

    train_episode = min(20, 50 - 5*t) #Decrease training episode for each training round
    trained_model = agent.train_model(model=model,
                                    model_name=agent_name,
                                    cwd = config["TRAINED_MODEL_FOLDER"] + '/' + agent_name,
                                    model_config=model_config,
                                    trained_agent_path = trained_agent_path,
                                    total_episodes=train_episode)
    return total_trained_episode + train_episode
def test(agent_name,config, trade, total_trained_episode, initial_capital):
    test_price_array, test_tech_array, test_turbulence_array = df_to_array(trade, tech_indicator_list= config['TECHNICAL_INDICATORS_LIST'], if_vix= True)
    env = StockTradingEnv
    agent_path= config["TRAINED_MODEL_FOLDER"] + agent_name + "/checkpoint_{}/checkpoint-{}".format("0"*(6-len(str(total_trained_episode))) + str(total_trained_episode), str(total_trained_episode))
    print("agent_path: {}".format(agent_path))
    episode_total_assets, episode_sell_buy, rewards, action_values = DRLAgent_rllib.DRL_prediction(agent_name,env, test_price_array, test_tech_array, test_turbulence_array, agent_path, config['MAX_STOCK'],initial_capital)
    return episode_total_assets, episode_sell_buy, rewards, action_values, \
            episode_total_assets[-1] # - initial_capital

def saving_test_result(config,date_list, episode_total_assets, episode_sell_buy, agent_name, tic_list):
    #date_list = pd.DataFrame(trade['date'].unique(), columns=['date'])
    account_value_erl = pd.DataFrame({'date':date_list['date'],'account_value':episode_total_assets[0:len(episode_total_assets)]})
    #action_values_pd = pd.DataFrame({'date':date_list['date'],'-1':action_values[:,0],'-0.5':action_values[:,1],'0':action_values[:,2],'0.5':action_values[:,3],'1':action_values[:,4]})
    
    #Save the action of agent
    episode_sell_buy_dict = {'date':date_list['date']}
    for i, tic in enumerate(tic_list):
        episode_sell_buy_dict[tic] = episode_sell_buy[:,i]
    episode_sell_buy_df = pd.DataFrame(episode_sell_buy_dict)

    #Save the account value during trading period
    os.makedirs(config["RESULT_FOLDER"] + agent_name, exist_ok=True)
    account_value_erl.to_csv(config["RESULT_FOLDER"] + agent_name + "/account_value.csv")
    episode_sell_buy_df.to_csv(config["RESULT_FOLDER"] + agent_name + "/sell_buy.csv")
    #action_values_pd.to_csv(config["RESULT_FOLDER"] + scenario + '/' + agent_name + "/action_values.csv")
    #Print the results
    print("==============Get Backtest Results of {}===========".format(agent_name))
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

    perf_stats_all = backtest_stats(account_value=account_value_erl)
    perf_stats_all = pd.DataFrame(perf_stats_all)
        
    #Visulize the results
    result_folder = config["RESULT_FOLDER"]
    visualize(config,result_folder, with_Baseline= True)

    if config['VISUALIZE_TRADING_ACTION']:
         visualize_trading_action(date_list['date'][0], str(pd.to_datetime(date_list['date'].iloc[-1]) + pd.DateOffset(1)), "", config)

def backtesting_rllib(config):


    for agent_name in config['AGENTS']:
        processed = pd.read_csv(config['DATA_PATH'])
        processed = processed.sort_values(['date','tic'])
        train_data = data_split(processed, config['BEGIN_DAY'], config['END_TRAIN'])

        trade_data = processed[processed.date >= config['END_TRAIN']]
        trade_date_list = pd.DataFrame(trade_data['date'].unique(), columns=['date'])

        num_tic = len(processed.tic.unique())
        num_train_day = int(len(train_data) / num_tic)
        num_test_day = int(len(trade_data) / num_tic)
        total_trained_episode = 0

        capital = config['INITIAL_CAPITAL']

        #Logging
        total_episode_total_assets = [] 
        total_episode_sell_buy = np.array([], dtype=np.float64).reshape(0,num_tic)
        total_rewards = []

        for t in range(num_test_day//config["WINDOW_SIZE"] + 1):
            trade = processed.iloc[num_train_day * num_tic:(num_train_day + config["WINDOW_SIZE"])*num_tic,:]
            print("train date: {} --> {} length: {}".format(train_data.iloc[0,:]['date'], train_data.iloc[-1,:]['date'], len(train_data) / num_tic))
            print("trade date: {} --> {} length: {}".format(trade.iloc[0,:]['date'],      trade.iloc[-1,:]['date'],      len(trade) / num_tic))
            

            total_trained_episode = train(agent_name,config, train_data, t, total_trained_episode)
            episode_total_assets, episode_sell_buy, rewards, _, capital = test(agent_name,config, trade, total_trained_episode, capital)
            print("\n")
            total_episode_total_assets += episode_total_assets
            total_episode_sell_buy = np.vstack([total_episode_sell_buy, episode_sell_buy])
            total_rewards += rewards

            num_train_day += config["WINDOW_SIZE"]
            train_data = processed.iloc[:num_train_day*num_tic,:]


        saving_test_result(config, trade_date_list, total_episode_total_assets, total_episode_sell_buy, agent_name, trade['tic'].unique())

def get_action_API(config):

    processed = pd.read_csv(config['DATA_PATH'])
    processed = processed.sort_values(['date','tic'])
    train_data = data_split(processed, config['BEGIN_DAY'], config['END_TRAIN'])

    env = StockTradingEnv

    train_price_array, train_tech_array, train_turbulence_array = df_to_array(train_data, tech_indicator_list= config_finrl.TECHNICAL_INDICATORS_LIST, if_vix= True)
    turbulence_bool = (train_turbulence_array > 99).astype(np.float32)
    env_config = {
            "price_array": train_price_array,
            "tech_array": train_tech_array,
            "turbulence_array": train_turbulence_array,
            "if_train": False,
        }
    max_stock = 100,
    initial_capital = 1000000
    env_instance = env(config=env_config, max_stock = max_stock, initial_capital = initial_capital)
    #state = env_instance.reset()
    #print(state.shape)

    print("train_price_array.shape: {}".format(train_price_array.shape))
    print("train_tech_array.shape: {}".format(train_tech_array.shape))
    print("train_turbulence_array.shape: {}".format(train_turbulence_array.shape))


    agent_name = 'ddpg'
    max_episode = max(os.listdir(config["TRAINED_MODEL_FOLDER"] + agent_name))
    max_episode = int(max_episode.split('_')[-1])
    agent_path= config["TRAINED_MODEL_FOLDER"] + agent_name + "/checkpoint_{}/checkpoint-{}".format("0"*(6-len(str(max_episode))) + str(max_episode), str(max_episode))

    initial_stocks = np.zeros(34, dtype=np.float32)
    state = get_state(initial_capital, initial_stocks, train_turbulence_array[-1], turbulence_bool[-1], train_price_array[-1], train_tech_array[-1])
    print(get_action(agent_name,state, env,agent_path, train_price_array, train_tech_array, train_turbulence_array))


    #state = get_state(initial_capital, initial_stocks, turbulence, turbulence_bool, price, tech)

def get_state(amount, stocks, turbulence, turbulence_bool, price, tech):
    # origin code: scale is the fator of 2 (2 ** -6, 2 ** -12), which is difficult to debug, thus, 
    # I change this scale to factor of 1
    amount = np.array(max(amount, 1e4) * (2 ** -12), dtype=np.float32)
    scale = np.array(2 ** -6, dtype=np.float32)
    #amount = np.array(max(self.amount, 1e4) * (10 ** -3), dtype=np.float32)
    #scale = np.array(10 ** -1, dtype=np.float32)
    # return np.hstack((amount,
    #                   self.turbulence_ary[self.day],
    #                   self.turbulence_bool[self.day],
    #                   price * scale,
    #                   self.stocks * scale,
    #                   self.stocks_cd,
    #                   self.tech_ary[self.day],
    #                   ))  # state.astype(np.float32)
    
    stocks_cd = np.zeros_like(stocks)

    return np.hstack((amount,
                        turbulence,
                        turbulence_bool,
                        price * scale,
                        stocks * scale,
                        stocks_cd,
                        tech,
                        ))
        
if __name__ == '__main__':
    config_path = "config.yaml"
    config = read_yaml(config_path)
    # if config['RLLIB'] == "elegantrl":
    #     pass
    #     #train_elegantrl(config)
    # #elif config['RLLIB'] == "stable_baselines":
    # #    train_stable_baselines(config)
    # elif config['RLLIB'] == "ray":
    #     backtesting_rllib(config)
    # else:
    #     raise ValueError("Please choose elegantrl or stable_baselines or ray")
    get_action_API(config)