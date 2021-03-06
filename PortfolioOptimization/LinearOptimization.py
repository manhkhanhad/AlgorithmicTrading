from turtle import width
from unittest import result
import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd

import plotly
import cufflinks

import plotly.express as px
import plotly.graph_objects as go
from utils import visualize_action

from utils import read_yaml, convert_data_format, calculate_return, plot_observation_price, process_data


def portpolio_optimization(data,totol_return, num_tic, lambda_ = 1):
    #mean = np.mean(data, axis=0).round(decimals=3)
    mean = totol_return
    risk = np.mean((data - mean))
    print("mean:", mean, "mean return of all stock", np.mean(mean), "risk:", risk)
    #lambda_ = lambda_ * (abs(risk/(np.mean(mean)) + 0.00001))
    #print("lambda_", lambda_)
    #Defind LP parameters
    c = -opt.matrix((mean * lambda_- risk).astype(np.double))
    A = opt.matrix(1.0, (1, num_tic))
    b = opt.matrix(1.0)
    G = -opt.matrix(np.eye(num_tic))
    h = opt.matrix(0.0, (num_tic ,1))

    #Solve
    sol = solvers.lp(c,G,h,A,b)

    return np.asarray(sol['x'])

# def sell_all_and_buy(stocks_pre, prices, cash, propotion, sell_fee = 0.001, buy_fee = 0.001):
#     cash = cash + np.dot(stocks_pre.T, prices) * (1-sell_fee) #sell all stock
#     stocks_cur = (cash * propotion// (prices * 10)) * 10

#     cash_temp = cash - np.dot(stocks_cur.T, prices) * (1+sell_fee) #buy stocks
#     print(cash_temp)
#     if cash_temp[0,0] <= 0:
#         return stocks_cur, cash[0,0], None, False
#     else:
#         return stocks_cur, cash_temp[0,0], None, True

def sell_all_and_buy(stocks_pre, prices, cash, proportion, sell_fee = 0.001, buy_fee = 0.001):
    cash = cash + np.dot(stocks_pre.T, prices) * (1-sell_fee) #sell all stock
    min_cash = (prices[0] / max(proportion))
    print("min_cash", min_cash)
    stock_batch = min_cash * proportion / prices
    stock = (cash // (min_cash * (1 + buy_fee))) * stock_batch

    stock = stock // 10 * 10 #round to 10

    cash = cash - np.dot(stock.T, prices) * (1+buy_fee) #buy stocks

    trading_fee = np.dot(stock.T, prices) * (sell_fee) + np.dot(stock.T, prices) * (buy_fee)

    portfolio_value = cash + np.dot(stock.T, prices)

    return stock, cash[0,0], None, True, trading_fee, portfolio_value


def trading(stocks_pre, prices, cash, propotion, sell_fee = 0.001, buy_fee = 0.001):
    portpolio_value = np.dot(stocks_pre.T, prices) * (1 - sell_fee) + cash #Actually: portpolio_value <= np.dot(stocks_pre.T, prices_cur) * (1 - sell_fee) + cash, but for easy calculation we consider two term are equal
    stocks_cur = (portpolio_value * propotion // (prices * 10)) * 10
    action = stocks_cur - stocks_pre

    sell_amount = np.dot(np.where(action < 0, abs(action), 0), prices) # amount of cash gain when sell stocks
    buy_amount  = np.dot(np.where(action > 0, action, 0), prices)  # amount of cash need for buy stocks
    trading_fee = sell_amount * sell_fee +  buy_amount * buy_fee

    cash = cash + sell_amount - buy_amount - trading_fee

    return stocks_cur, cash, action


def main(config):
    # #Load data
    # data_raw = pd.read_csv('/mmlabworkspace/WorkSpaces/danhnt/tuyensh/khanhngo/AlgorithmicTrading/Data/VnStock_preprocessed.csv')

    # #Convert data format
    # data = convert_data_format(data_raw)

    # t = 0
    # observation_steps = 60
    # num_tic = 34

    # observation_data = data[observation_steps * t: observation_steps * (t+1)]
    # return_data = calculate_return(observation_data).round(decimals=3)
    # plot_observation_price(t, observation_steps,num_tic, data_raw)
    

    # portpolio = portpolio_optimization(return_data, num_tic, lambda_ = 1)
    # print(portpolio)

    #Load data
    # data_raw = pd.read_csv(config['DATA_PATH'])
    # data = data_raw[(data_raw.date >= config['START_DAY']) & (data_raw.date <= config['END_DAY'])]
    # data = convert_data_format(data)

    data_raw = process_data(config["DATA_DIR"])
    #data_raw = pd.read_csv(config['DATA_PATH'])
    data = data_raw[(data_raw.date >= config['START_DAY']) & (data_raw.date <= config['END_DAY'])]
    data = convert_data_format(data)

    num_days,num_tic = data.shape[0], (data.shape[1] - 1) #minus 1 because of we don't need date column

    total_trading_fee = 0
    actions = []
    #Backtesting
    stock = np.zeros((num_tic,1)) #init stock
    cash = config['INIT_CASH'] #init cash
    previous_portfolio_value = cash
    for t in range(0, (num_days - config["OBSERVATION_STEPS"])//config["WAIT_STEPS"] + 1):
        trading_day = config["OBSERVATION_STEPS"] + t * config["WAIT_STEPS"]
        observation_data = data[trading_day - config["OBSERVATION_STEPS"] : trading_day ,:]

        totol_return = (observation_data[-1,1:] - observation_data[0,1:]) / observation_data[0,1:]
        return_data = calculate_return(observation_data).round(decimals=3)

        date = data[trading_day-1,0]
        prices = data[trading_day-1,1:].reshape((num_tic,1))
        portpolio_proportion = portpolio_optimization(return_data, totol_return, num_tic, config['LAMBDA'])

        print(portpolio_proportion)
        if config['SELL_ALL']:
            stock, cash, action, done, trading_fee, portfolio_value = sell_all_and_buy(stock, prices, cash, portpolio_proportion)
            return_value = (portfolio_value - config['INIT_CASH']) / config['INIT_CASH']
            
            is_profit = ((portfolio_value - previous_portfolio_value) > 0)
            previous_portfolio_value = portfolio_value

            total_trading_fee += trading_fee
        else:
            stock, cash, action, done = trading(stock, prices, cash, portpolio_proportion)
        
        if not done:
            print("Bankrupt !!!!, remain cash:", cash)
            break

        print("Date:", date, "Cash:", cash, "Stock:", stock, "Action:", action)
        print("-----------------------------------------------------")
        actions.append((date, cash, stock, return_value[0,0], is_profit, portfolio_value))

    print("Final return value:", actions[-1][-3])
    print("Trading fee:", total_trading_fee)
    visualize_action(data_raw, actions, config)
    return actions[-1][-3] #return return_value

def tuning(config):
    tune_config = {
        "DATA_PATH": config["DATA_PATH"],
        "START_DAY": config["START_DAY"],
        "END_DAY": config["END_DAY"],
        "INIT_CASH": config["INIT_CASH"],
        "SAVE_DIR": "Results",
        "SELL_ALL": True,
        "LAMBDA": None,
        "OBSERVATION_STEPS": None,
        "WAIT_STEPS": None
    }

    results = {
        "lambda": [],
        "observation_steps": [],
        "wait_steps": [],
        "return_value": []
    }

    for lamda in np.arange(0,1,0.05):
        for observation_step in np.arange(30, 360, 30):
            for wait_step in np.arange(10, 90, 10):
                tune_config["LAMBDA"] = lamda
                tune_config["OBSERVATION_STEPS"] = observation_step
                tune_config["WAIT_STEPS"] = wait_step
                return_value = main(tune_config)
                results["lambda"].append(lamda)
                results["observation_steps"].append(observation_step)
                results["wait_steps"].append(wait_step)
                results["return_value"].append(return_value)
                print("Lambda:", lamda, "Observation_step:", observation_step, "Wait_step:", wait_step, "Return value:", result)
                print("-----------------------------------------------------")
    df = pd.DataFrame(results)
    df.to_csv(config["SAVE_DIR"] + "/tuning_result.csv")

if __name__ == '__main__':
    config_path = "config_LP.yaml"
    config = read_yaml(config_path)
    if config['MODE'] == "test":
        main(config)
    elif config['MODE'] == "tuning":
        print("Tuning...")
        tuning(config)
    else:
        raise ValueError("Invalid mode")