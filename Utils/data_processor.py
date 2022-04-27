from ast import parse
import vnquant.DataLoader as web
from datetime import datetime
import requests
import pandas as pd
import argparse
import pandas as pd
import os
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer

class DataDowloader:
    """
    Download data from internet

    Attibutes:
    _________
        start_day: str
            start date of the data (yyyy-mm-dd)
        end_day: str
            end date of the data (yyyy-mm-dd)
        stock_list: list
            list of stock to download
        data_source: str
            data source to download from ("vnd": vndirect  or 'cafe": CafeF)
        save_dir: str
            directory to save the data
        minimal: bool
            default is True, we only clone high, low, open, close, adjust price, volume of stocks. 
            In contrast, more information is added, for example volumn_reconcile, volumn_match,...
    _________
    """
    def __init__(self, config):
        self.start_day = config.start_day
        self.end_date = config.end_day
        self.stock_list = config.stock_list
        self.data_source = config.data_source
        self.save_path = config.save_path
        self.minimal = config.minimal

    def download(self):
        """
        Download data from internet
        """
        self.data = None
        for stock in self.stock_list:
            loader = web.DataLoader(symbols=[stock], start=self.start_day, end=self.end_date, minimal=self.minimal, data_source=self.data_source)
            tmp = loader.download()
            tmp.columns = ['high','low','open','close','adjust','volume']
            tmp = tmp.assign(tic=stock)
            tmp['date'] = tmp.index
            #tmp = tmp.drop('adjust',axis = 1)
            self.data = pd.concat([self.data,tmp], ignore_index=True)
        
        self.data.to_csv(self.save_path)

    def process(self):
        """
        Process data
        """
        print("Processing data")

        #Process missing data

        #get the list of day when the price data of all stocks are available
        list_tics = list(pd.unique(self.data['tic']))
        day_list = self.data[self.data.tic == list_tics[0]]['date']
        for tic_name in list_tics[1:]:
            day_list = pd.Series(list(set(day_list).intersection(set(self.data[self.data["tic"] == tic_name]['date']))))
        
        #fillter the data to only have the day when all stocks have data
        self.data = pd.merge(self.data, day_list.to_frame('date'), on = 'date')

        self.data.to_csv(self.save_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_day', type=str, default='2009-12-22')
    parser.add_argument('--end_day', type=str, default='2022-03-01')
    parser.add_argument('--stock_list', type=str, default=['AGR', 'BVH', 'CTG', 'DIG', 'DPM', 'DRC', 'DXG', 'EIB', 'FPT',
       'HAG', 'HAI', 'HBC', 'HSG', 'HT1', 'ITA', 'KBC', 'LGC', 'MSN',
       'PNJ', 'PVI', 'PVS', 'PVT', 'RIC', 'SAM', 'SBT', 'SSI', 'STB',
       'TCR', 'TSC', 'TTF', 'VCB', 'VIC', 'VNE', 'VNM'])
    parser.add_argument('--data_source', type=str, default='vnd')
    parser.add_argument('--save_path', type=str, default='Data/VN_stock_raw.csv')
    parser.add_argument('--minimal', type=bool, default=True)
    args = parser.parse_args()


    dowloader = DataDowloader(args)
    dowloader.download()
    dowloader.process()


class CoPhieu68_DataProcessor:
    def __init__(self, data_dir, save_path, indicator):
        """
        Args:
            data_dir: str 
                directory that contain data downloaded from CoPhieu68
            save_path: str
                path to save processed data
            indicator: list
                list of indicator to caculate
        """
        self.data_dir = data_dir
        self.save_path = save_path
        self.indicator = indicator
    def download(self):
        """
        Download data from internet
        """
        pass
    def download_VNINDEX(self):
        """
        Download VNINDEX data and processing
        """
        VNIndex = pd.read_csv("/content/VNIndex.csv")
        VNIndex.columns = ['date','close','open','high','low','volumn','change']
        VNIndex['date'] = pd.DatetimeIndex(VNIndex['date'])
        VNIndex['date'] = pd.to_datetime(VNIndex['date'], format='%Y%m%d')
        VNIndex['date'] = VNIndex['date'].apply(lambda x: x.strftime('%Y-%m-%d'))

        for col in ['close','open','high','low']:
            VNIndex[col] = VNIndex[col].str.replace(',','').astype(np.float64)
            VNIndex[col] = VNIndex[col].apply(pd.to_numeric)

        VNIndex = VNIndex.sort_values(['date'])
        VNIndex.to_csv("VNIndex_test.csv")

    def process(self):
        """
        Process data and save to csv
        """
        # Read and merge data to a Dataframe
        for f in os.listdir(self.data_dir):
            tmp = pd.read_csv(os.path.join(self.data_dir, f))
            tmp.columns = ['tic','date','open','high','low','close','volume']
            tmp = tmp.assign(tic=f.split('_')[-1][:-4].upper())
            data = pd.concat([data,tmp], ignore_index=True)

        # Convert datetime format
        data['date'] = pd.to_datetime(data['date'].astype(str), format='%Y%m%d')
        data['date'] = data['date'].apply(lambda x: x.strftime('%Y-%m-%d'))

        # Remove missing data
        list_tics = list(pd.unique(data['tic']))
        day_list = data[data.tic == list_tics[0]]['date']
        for tic_name in list_tics[1:]:
            day_list = pd.Series(list(set(day_list).intersection(set(data[data["tic"] == tic_name]['date']))))

        data = pd.merge(data, day_list.to_frame('date'), on = 'date')

        #Calculate Indicator

        fe = FeatureEngineer(
                            use_technical_indicator=True,
                            tech_indicator_list = self.indicator,
                            use_vix=True,
                            use_turbulence=True,
                            user_defined_feature = False)

        processed = fe.preprocess_data(data)

        processed.to_csv(self.save_path)