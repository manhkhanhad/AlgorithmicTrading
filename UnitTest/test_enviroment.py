import numpy as np
import sys
sys.path.append("../AlgorithmicTrading/")
from Utils.utils import read_yaml, df_to_array
from finrl.finrl_meta.preprocessor.preprocessors import data_split
import pandas as pd
from Model.enviroment import StockTradingEnv, VNStockEnv

config_path = "config.yaml"
config = read_yaml(config_path)

#Load data
processed = pd.read_csv('UnitTest/unittest_enviroment.csv')
processed = processed.sort_values(['date','tic'])

#Test arguments
price_array, tech_array, turbulence_array = df_to_array(processed, tech_indicator_list= config['TECHNICAL_INDICATORS_LIST'], if_vix= True)
initial_stocks = np.array([100, 80, 120, 70])
action = np.array([-0.20, +0.30, -0.20, +0.20])
stocks_after_trade = initial_stocks + action*100

print("stock at t:", initial_stocks)
print("price at t:", price_array[0])

print("action at t:", action)

print("stock at t+1:", initial_stocks + action*100)
print("price at t+1:", price_array[1])



#Testing
env_config = {'price_array':price_array,
                'tech_array':tech_array,
                'turbulence_array':turbulence_array,
                'if_train':False,
                'initial_stocks': initial_stocks}


env = VNStockEnv
env_instance = env(config=env_config)
print("buy cost:", env_instance.buy_cost_pct)
print("sell cost:", env_instance.sell_cost_pct)

print("\nexpected result")
trading_cost = (np.where(action > 0, action, 0) @ price_array[1].T) * env_instance.buy_cost_pct - (np.where(action < 0, action, 0) @ price_array[1].T) * env_instance.sell_cost_pct
print("trading_cost:", trading_cost * 100)


print("amount t+1", 1e6 + (np.where(action > 0, - action * 100, 0) @ price_array[1].T) * (1 + env_instance.buy_cost_pct) + (np.where(action < 0, - action * 100, 0) @ price_array[1].T) * (1 - env_instance.sell_cost_pct))
print("total asset at t+1:", stocks_after_trade @ price_array[1].T +  1e6 + (np.where(action > 0, - action * 100, 0) @ price_array[1].T) * (1 + env_instance.buy_cost_pct) + (np.where(action < 0, - action * 100, 0) @ price_array[1].T) * (1 - env_instance.sell_cost_pct))

print("\nactual result")


env_instance.reset()
print("total asset at t:", env_instance.total_asset)
#Do trading action
env_instance.step(action)

#print(buy_sell_actions)
print("amount t+1", env_instance.amount)
print("total asset at t+1:", env_instance.total_asset)
