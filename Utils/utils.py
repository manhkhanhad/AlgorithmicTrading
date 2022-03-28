from xml.dom.minidom import Document
import yaml
import numpy as np
import os
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