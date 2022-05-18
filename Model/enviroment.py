from distutils.command import config
import re
import numpy as np
import os
import gym
from numpy import random as rd
from Model.reward_scheme import PenalizedProfit,AnomalousProfit, SimpleProfit


class StockTradingEnv(gym.Env):

    def __init__(self, config, initial_account=1e6,
                 gamma=0.99, turbulence_thresh=99, min_stock_rate=0.1,
                 max_stock=1e2, initial_capital=1e6, buy_cost_pct=1e-3, 
                 sell_cost_pct=1e-3,reward_scaling=2 ** -11,  initial_stocks=None,
                 ):
        price_ary = config['price_array']
        tech_ary = config['tech_array']
        turbulence_ary = config['turbulence_array']
        if_train = config['if_train']
        #initial_stocks = config['initial_stocks']
        n = price_ary.shape[0]
        self.price_ary =  price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        self.turbulence_ary = turbulence_ary
        
        #self.tech_ary = self.tech_ary * 2 ** -7 
        self.tech_ary = self.tech_ary * 1 ** -7

        self.turbulence_bool = (turbulence_ary > turbulence_thresh).astype(np.float32)
        self.turbulence_ary = (self.sigmoid_sign(turbulence_ary, turbulence_thresh) * 2 ** -5).astype(np.float32)

        stock_dim = self.price_ary.shape[1]
        self.gamma = gamma
        self.max_stock = max_stock
        self.min_stock_rate = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital

        self.initial_stocks = np.zeros(stock_dim, dtype=np.float32) if initial_stocks is None else initial_stocks
        self.min_stock_batch = 10 # minimum number of stocks to buy/sell

        # reset()
        self.day = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None

        # environment information
        self.env_name = 'StockEnv'
        # self.state_dim = 1 + 2 + 2 * stock_dim + self.tech_ary.shape[1]
        # # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_ary.shape[1]
        # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.stocks_cd = None
        self.action_dim = stock_dim
        self.max_step = self.price_ary.shape[0] - 1
        self.if_train = if_train
        self.if_discrete = False
        self.target_return = 5.0
        self.episode_return = 0.0
        
        self.observation_space = gym.spaces.Box(low=-3000, high=1000000, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        
        self.total_assets = [] # record the total assets
    def reset(self):
        self.day = 0
        price = self.price_ary[self.day] 
        #price = self.price_ary[self.day] * config['PRICE_SCALER']   # for scaling data
        if self.if_train:
            self.stocks = (self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)).astype(np.float32)
            self.stocks_cd = np.zeros_like(self.stocks)
            self.amount = self.initial_capital * rd.uniform(0.95, 1.05) - (self.stocks * price).sum()
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cd = np.zeros_like(self.stocks)
            self.amount = self.initial_capital

        self.total_asset = self.amount + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        return self.get_state(price)  # state

    def step(self, actions):
        actions = (actions * self.max_stock).astype(int)
        buy_sell_actions = [0] * self.action_dim # Just for loging action
        self.day += 1
        if self.day % 5000 == 0:
            print('day:', self.day)
        price = self.price_ary[self.day]
        self.stocks_cd += 1
        trading_cost = 0
        if self.turbulence_bool[self.day] == 0:
            #min_action = int(self.max_stock * self.min_stock_rate)  # stock_cd
            min_action = 0
            for index in np.where(actions < -min_action)[0]:  # sell_index:
                if price[index] > 0:  # Sell only if current asset is > 0
                    sell_num_shares = min(self.stocks[index], -actions[index])
                    sell_num_shares = sell_num_shares - sell_num_shares % self.min_stock_batch
                    buy_sell_actions[index] = -sell_num_shares
                    self.stocks[index] -= sell_num_shares
                    trading_cost += price[index] * sell_num_shares * self.sell_cost_pct # sell_cost
                    self.amount += price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                    self.stocks_cd[index] = 0
            for index in np.where(actions > min_action)[0]:  # buy_index:
                if price[index] > 0:  # Buy only if the price is > 0 (no missing data in this particular date)
                    buy_num_shares = min(self.amount // (price[index]*(1+self.buy_cost_pct)), actions[index])
                    buy_num_shares = buy_num_shares - buy_num_shares % self.min_stock_batch
                    buy_sell_actions[index] = buy_num_shares
                    self.stocks[index] += buy_num_shares
                    trading_cost += price[index] * buy_num_shares * self.buy_cost_pct # buy_cost
                    self.amount -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)
                    self.stocks_cd[index] = 0
                    
                    if self.amount < 0:
                        raise Exception('Amount: {} < 0 Date: {}, index_stock: {}'.format(self.amount, self.day, index))

        else:  # sell all when turbulence
            self.amount += (self.stocks * price).sum() * (1 - self.sell_cost_pct)
            self.stocks[:] = 0
            self.stocks_cd[:] = 0
        #print("Trading cost: ", trading_cost)
        state = self.get_state(price)
        total_asset = self.amount + (self.stocks * price).sum()
        self.total_assets.append(total_asset)

        #reward = PenalizedProfit(self.initial_capital, total_asset, self.amount, self.day)
        #reward = AnomalousProfit(self.total_assets, self.day)
        #reward = SimpleProfit(self.total_assets, self.day)
        #print("reward", reward)
        reward = (total_asset - self.total_asset) * self.reward_scaling

        self.total_asset = total_asset
        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.day == self.max_step
        if done:
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset

        if not self.if_train:
            return state, reward, done, buy_sell_actions, dict()
        else:
            return state, reward, done, dict()

    def get_state(self, price):
        # origin code: scale is the fator of 2 (2 ** -6, 2 ** -12), which is difficult to debug, thus, 
        # I change this scale to factor of 1
        amount = np.array(max(self.amount, 1e4) * (2 ** -12), dtype=np.float32)
        scale = np.array(2 ** -6, dtype=np.float32)
        #amount = np.array(max(self.amount, 1e4) * (10 ** -3), dtype=np.float32)
        #scale = np.array(10 ** -1, dtype=np.float32)
        return np.hstack((amount,
                          self.turbulence_ary[self.day],
                          self.turbulence_bool[self.day],
                          price * scale,
                          self.stocks * scale,
                          self.stocks_cd,
                          self.tech_ary[self.day],
                          ))  # state.astype(np.float32)
    
    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh

class RealtimeTradingEnv(gym.Env):

    def __init__(self, config, initial_account=1e6,
                 gamma=0.99, turbulence_thresh=99, min_stock_rate=0.1,
                 max_stock=1e2, initial_capital=1e6, buy_cost_pct=1e-3, 
                 sell_cost_pct=1e-3,reward_scaling=2 ** -11,  initial_stocks=None,
                 ):
        price_ary = config['price_array']
        tech_ary = config['tech_array']
        turbulence_ary = config['turbulence_array']
        if_train = config['if_train']

        #initial_stocks = config['initial_stocks']
        
        #initial_stocks = config['initial_stocks']
        n = price_ary.shape[0]
        self.price_ary =  price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        self.turbulence_ary = turbulence_ary
        
        #self.tech_ary = self.tech_ary * 2 ** -7 
        self.tech_ary = self.tech_ary * 1 ** -7

        self.turbulence_bool = (turbulence_ary > turbulence_thresh).astype(np.float32)
        self.turbulence_ary = (self.sigmoid_sign(turbulence_ary, turbulence_thresh) * 2 ** -5).astype(np.float32)

        stock_dim = self.price_ary.shape[1]
        self.gamma = gamma
        self.max_stock = max_stock
        self.min_stock_rate = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital

        self.initial_stocks = np.zeros(stock_dim, dtype=np.float32) if initial_stocks is None else initial_stocks
        self.min_stock_batch = 10 # minimum number of stocks to buy/sell

        # reset()
        self.day = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None

        # environment information
        self.env_name = 'StockEnv'
        # self.state_dim = 1 + 2 + 2 * stock_dim + self.tech_ary.shape[1]
        # # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_ary.shape[1]
        # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.stocks_cd = None
        self.action_dim = stock_dim
        self.max_step = self.price_ary.shape[0] - 1
        self.if_train = if_train
        self.if_discrete = False
        self.target_return = 5.0
        self.episode_return = 0.0
        
        self.observation_space = gym.spaces.Box(low=-3000, high=1000000, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        
        self.total_assets = [] # record the total assets
    
    def reset(self):
        self.day = 0
        price = self.price_ary[self.day] 
        #price = self.price_ary[self.day] * config['PRICE_SCALER']   # for scaling data
        if self.if_train:
            self.stocks = (self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)).astype(np.float32)
            self.stocks_cd = np.zeros_like(self.stocks)
            self.amount = self.initial_capital * rd.uniform(0.95, 1.05) - (self.stocks * price).sum()
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cd = np.zeros_like(self.stocks)
            self.amount = self.initial_capital

        self.total_asset = self.amount + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        return self.get_state(price)  # state
    
    def step(self, actions):
        actions = (actions * self.max_stock).astype(int)
        buy_sell_actions = [0] * self.action_dim # Just for loging action
        self.day += 1
        if self.day % 5000 == 0:
            print('day:', self.day)
        price = self.price_ary[self.day]
        self.stocks_cd += 1
        trading_cost = 0
        if self.turbulence_bool[self.day] == 0:
            #min_action = int(self.max_stock * self.min_stock_rate)  # stock_cd
            min_action = 0
            for index in np.where(actions < -min_action)[0]:  # sell_index:
                if price[index] > 0:  # Sell only if current asset is > 0
                    sell_num_shares = min(self.stocks[index], -actions[index])
                    sell_num_shares = sell_num_shares - sell_num_shares % self.min_stock_batch
                    buy_sell_actions[index] = -sell_num_shares
                    self.stocks[index] -= sell_num_shares
                    trading_cost += price[index] * sell_num_shares * self.sell_cost_pct # sell_cost
                    self.amount += price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                    self.stocks_cd[index] = 0
            for index in np.where(actions > min_action)[0]:  # buy_index:
                if price[index] > 0:  # Buy only if the price is > 0 (no missing data in this particular date)
                    buy_num_shares = min(self.amount // (price[index]*(1+self.buy_cost_pct)), actions[index])
                    buy_num_shares = buy_num_shares - buy_num_shares % self.min_stock_batch
                    buy_sell_actions[index] = buy_num_shares
                    self.stocks[index] += buy_num_shares
                    trading_cost += price[index] * buy_num_shares * self.buy_cost_pct # buy_cost
                    self.amount -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)
                    self.stocks_cd[index] = 0
                    
                    if self.amount < 0:
                        raise Exception('Amount: {} < 0 Date: {}, index_stock: {}'.format(self.amount, self.day, index))

        else:  # sell all when turbulence
            self.amount += (self.stocks * price).sum() * (1 - self.sell_cost_pct)
            self.stocks[:] = 0
            self.stocks_cd[:] = 0
        #print("Trading cost: ", trading_cost)
        state = self.get_state(price)
        total_asset = self.amount + (self.stocks * price).sum()
        self.total_assets.append(total_asset)

        #reward = PenalizedProfit(self.initial_capital, total_asset, self.amount, self.day)
        #reward = AnomalousProfit(self.total_assets, self.day)
        #reward = SimpleProfit(self.total_assets, self.day)
        #print("reward", reward)
        reward = (total_asset - self.total_asset) * self.reward_scaling

        self.total_asset = total_asset
        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.day == self.max_step
        if done:
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset

        if not self.if_train:
            return state, reward, done, buy_sell_actions, dict()
        else:
            return state, reward, done, dict()
        
    def get_action(self, actions, price, turbulence_bool):
        actions = (actions * self.max_stock).astype(int)
        buy_sell_actions = [0] * self.action_dim # Just for loging action
        trading_cost = 0
        amount = self.amount 
        if turbulence_bool == 0:
            min_action = 0
            for index in np.where(actions < -min_action)[0]:  # sell_index:
                if price[index] > 0:  # Sell only if current asset is > 0
                    sell_num_shares = min(self.stocks[index], -actions[index])
                    sell_num_shares = sell_num_shares - sell_num_shares % self.min_stock_batch
                    buy_sell_actions[index] = -sell_num_shares
                    trading_cost += price[index] * sell_num_shares * self.sell_cost_pct # sell_cost
                    amount += price[index] * sell_num_shares * (1 - self.sell_cost_pct)
            for index in np.where(actions > min_action)[0]:  # buy_index:
                if price[index] > 0:  # Buy only if the price is > 0 (no missing data in this particular date)
                    buy_num_shares = min(self.amount // (price[index]*(1+self.buy_cost_pct)), actions[index])
                    buy_num_shares = buy_num_shares - buy_num_shares % self.min_stock_batch
                    buy_sell_actions[index] = buy_num_shares
                    trading_cost += price[index] * buy_num_shares * self.buy_cost_pct # buy_cost
                    amount -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)
              
                    if self.amount < 0:
                        raise Exception('Amount: {} < 0 Date: {}, index_stock: {}'.format(self.amount, self.day, index))

        else:  # sell all when turbulence
            amount += (self.stocks * price).sum() * (1 - self.sell_cost_pct)
            
        return buy_sell_actions
        

    def get_state(self, price):
        # origin code: scale is the fator of 2 (2 ** -6, 2 ** -12), which is difficult to debug, thus, 
        # I change this scale to factor of 1
        amount = np.array(max(self.amount, 1e4) * (2 ** -12), dtype=np.float32)
        scale = np.array(2 ** -6, dtype=np.float32)
        #amount = np.array(max(self.amount, 1e4) * (10 ** -3), dtype=np.float32)
        #scale = np.array(10 ** -1, dtype=np.float32)
        return np.hstack((amount,
                          self.turbulence_ary[self.day],
                          self.turbulence_bool[self.day],
                          price * scale,
                          self.stocks * scale,
                          self.stocks_cd,
                          self.tech_ary[self.day],
                          ))  # state.astype(np.float32)
    
    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh

    def update_realtime_data(self, price_array, tech_array, turbulence_array):
        self.price_ary = price_array
        self.tech_ary = tech_array
        self.turbulence_ary = turbulence_array
        self.turbulence_bool = self.sigmoid_sign(self.turbulence_ary, self.turbulence_thresh)

class VNStockEnv(gym.Env):
    def __init__(self, config, initial_account=1e6,
                 gamma=0.99, turbulence_thresh=99, min_stock_rate=0.1,
                 max_stock=1e2, initial_capital=1e6, buy_cost_pct=1e-3, 
                 sell_cost_pct=1e-3,reward_scaling=2 ** -11,  initial_stocks=None,
                 ):
        price_ary = config['price_array']
        tech_ary = config['tech_array']
        turbulence_ary = config['turbulence_array']
        if_train = config['if_train']
        #initial_stocks = config['initial_stocks']
        n = price_ary.shape[0]
        self.price_ary =  price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        self.turbulence_ary = turbulence_ary
        
        #self.tech_ary = self.tech_ary * 2 ** -7 
        self.tech_ary = self.tech_ary * 1 ** -7

        self.turbulence_bool = (turbulence_ary > turbulence_thresh).astype(np.float32)
        self.turbulence_ary = (self.sigmoid_sign(turbulence_ary, turbulence_thresh) * 2 ** -5).astype(np.float32)

        stock_dim = self.price_ary.shape[1]
        self.gamma = gamma
        self.max_stock = max_stock
        self.min_stock_rate = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital

        self.initial_stocks = np.zeros(stock_dim, dtype=np.float32) if initial_stocks is None else initial_stocks
        self.min_stock_batch = 10 # minimum number of stocks to buy/sell

        # reset()
        self.day = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None

        # environment information
        self.env_name = 'StockEnv'
        # self.state_dim = 1 + 2 + 2 * stock_dim + self.tech_ary.shape[1]
        # # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_ary.shape[1]
        # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.stocks_cd = None
        self.action_dim = stock_dim
        self.max_step = self.price_ary.shape[0] - 1
        self.if_train = if_train
        self.if_discrete = False
        self.target_return = 5.0
        self.episode_return = 0.0
        
        self.observation_space = gym.spaces.Box(low=-3000, high=1000000, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        
        self.total_assets = [] # record the total assets

        self.stock_exchange = np.zeros((3, stock_dim), dtype=np.float32) # stock waiting for returning to portfolio (t+2)
        self.cash_exchange = np.zeros((3, 1), dtype=np.float32) #Cash waiting for returning to wallet (t+2)

    def reset(self):
        self.day = 0
        price = self.price_ary[self.day] 
        #price = self.price_ary[self.day] * config['PRICE_SCALER']   # for scaling data
        if self.if_train:
            self.stocks = (self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)).astype(np.float32)
            self.stocks_cd = np.zeros_like(self.stocks)
            self.amount = self.initial_capital * rd.uniform(0.95, 1.05) - (self.stocks * price).sum()
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cd = np.zeros_like(self.stocks)
            self.amount = self.initial_capital

        self.total_asset = self.amount + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        return self.get_state(price)  # state
    
    def step(self, actions):
        actions = (actions * self.max_stock).astype(int)
        buy_sell_actions = [0] * self.action_dim
        self.day += 1
        price = self.price_ary[self.day]
        self.stocks_cd += 1
        trading_cost = 0
        if self.turbulence_bool[self.day] == 0:
            min_action = 0
            for index in np.where(actions < -min_action)[0]:
                sell_cost, sell_num_shares = self._sell_stock(index, actions[index], price)
                trading_cost += sell_cost
                buy_sell_actions[index] = -sell_num_shares
            
            for index in np.where(actions > min_action)[0]:
                buy_cost, buy_num_shares = self._buy_stock(index, actions[index], price)
                trading_cost += sell_cost
                buy_sell_actions[index] = buy_num_shares

            if self.amount < 0:
                raise Exception('Amount: {} < 0 Date: {}, index_stock: {}'.format(self.amount, self.day, index))

        else:
            self._sell_all_stocks(price)
        
        # update cash and stock exchange
        self.amount += float(self.cash_exchange[2]) #Update cash t+2
        self.cash_exchange[2] = self.cash_exchange[1]
        self.cash_exchange[1] = self.cash_exchange[0]
        self.cash_exchange[0] = 0

        self.stocks += self.stock_exchange[2]
        self.stock_exchange[2] = self.stock_exchange[1]
        self.stock_exchange[1] = self.stock_exchange[0]
        self.stock_exchange[0] = 0

        #Calculate the reward
        state = self.get_state(price)
        total_asset = self.amount + (self.stocks * price).sum()
        self.total_assets.append(total_asset)
        
        reward = (total_asset - self.total_asset) * self.reward_scaling

        self.total_asset = total_asset
        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.day == self.max_step
        if done:
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset

        if not self.if_train:
            return state, reward, done, buy_sell_actions, dict()
        else:
            return state, reward, done, dict()

    def _sell_stock(self, index, action, price):
        if price[index] > 0:
            sell_num_shares = min(self.stocks[index], -action)
            sell_num_shares = sell_num_shares - sell_num_shares % self.min_stock_batch
            self.stocks[index] -= sell_num_shares
            trading_cost = price[index] * sell_num_shares * self.sell_cost_pct
            self.cash_exchange[0] += price[index] * sell_num_shares * (1 - self.sell_cost_pct)

            return trading_cost, sell_num_shares

    def _buy_stock(self, index, action, price):
        if price[index] > 0:
            buy_num_shares = min(self.amount // (price[index]*(1+self.buy_cost_pct)), action)
            buy_num_shares = buy_num_shares - buy_num_shares % self.min_stock_batch
            trading_cost = price[index] * buy_num_shares * self.buy_cost_pct
            self.amount -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)
            self.stocks_cd[index] = 0

            self.stock_exchange[0][index] = buy_num_shares
            return trading_cost, buy_num_shares
    
    def _sell_all_stocks(self,price):
        self.cash_exchange[0] = (self.stocks * price).sum() * (1 - self.sell_cost_pct)
        self.stocks[:] = 0
        self.stocks_cd[:] = 0

    def get_state(self, price):
        # origin code: scale is the fator of 2 (2 ** -6, 2 ** -12), which is difficult to debug, thus, 
        # I change this scale to factor of 1
        amount = np.array(max(self.amount, 1e4) * (2 ** -12), dtype=np.float32)
        scale = np.array(2 ** -6, dtype=np.float32)
        #amount = np.array(max(self.amount, 1e4) * (10 ** -3), dtype=np.float32)
        #scale = np.array(10 ** -1, dtype=np.float32)
        return np.hstack((amount,
                          self.turbulence_ary[self.day],
                          self.turbulence_bool[self.day],
                          price * scale,
                          self.stocks * scale,
                          self.stocks_cd,
                          self.tech_ary[self.day],
                          ))  # state.astype(np.float32)

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh