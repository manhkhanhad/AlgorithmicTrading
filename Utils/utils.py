from xml.dom.minidom import Document
import yaml
import numpy as np
import os
import pandas as pd

def read_yaml(config_weight):
    try:
        with open(config_weight) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        config['SCENARIOS'] = list(config['PERIODS'].keys())
        os.makedirs(config['TRAINED_MODEL_FOLDER'],exist_ok=True)
        with open(os.path.join(config['TRAINED_MODEL_FOLDER'],'config.yaml'), 'w') as file:
            yaml.dump(config, file)

    except ValueError:
        print("No config file!")
    return config

def df_to_array(df, tech_indicator_list, if_vix):
    df = df.copy()
    unique_ticker = df.tic.unique()
    if_first_time = True
    for tic in unique_ticker:
        if if_first_time:
            price_array = df[df.tic == tic][["close"]].values
            tech_array = df[df.tic == tic][tech_indicator_list].values
            if if_vix:
                turbulence_array = df[df.tic == tic]["vix"].values
            else:
                turbulence_array = df[df.tic == tic]["turbulence"].values
            if_first_time = False
        else:
            price_array = np.hstack(
                [price_array, df[df.tic == tic][["close"]].values]
            )
            tech_array = np.hstack(
                [tech_array, df[df.tic == tic][tech_indicator_list].values]
            )
    print("Successfully transformed into array")
    return price_array, tech_array, turbulence_array

def update_new_data():
    pass


import requests
from bs4 import BeautifulSoup
import time
def crawl_stock_price(code):
    url = f'http://www.cophieu68.vn/historyprice.php?id={code}'

    while(1):
        res = requests.get(url, timeout=10)
        if res.ok == True:
            break
        else:
            time.sleep(sleep_time)
                
    soup = BeautifulSoup(res.content, features="lxml")
    
    table = soup.html.find('table', attrs={'class':'stock'})
    rows = table.find_all('tr')
    today_row = [ele.text.strip() for ele in rows[1].find_all('td')]
    
    return [code.upper(), today_row[7], today_row[8], today_row[9], today_row[5], today_row[6], today_row[1]]

def get_realtime_data(stock_codes):
    crawled_data = []
    for code in stock_codes:
        crawled_data.append(crawl_stock_price(code))

    data_dict = {"tic" : [],
            "open" : [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
            "date": []}

    for tic_data in crawled_data:
        data_dict['tic'].append(tic_data[0])
        data_dict['open'].append(float(tic_data[1]))
        data_dict['high'].append(float(tic_data[2]))
        data_dict['low'].append(float(tic_data[3]))
        data_dict['close'].append(float(tic_data[4]))
        data_dict['volume'].append(float(tic_data[5].replace(',','')))
        data_dict['date'].append(tic_data[6])

    new_data = pd.DataFrame(data_dict)
    new_data['date'] = pd.to_datetime(new_data['date'])
    new_data['date'] = new_data['date'].apply(lambda x: x.strftime("%Y-%m-%d"))

    return new_data