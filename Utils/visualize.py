from multiprocessing.sharedctypes import Value
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
import plotly.express as px

def visualization_single(result_folder):
    df = pd.read_csv(os.path.join(result_folder,"account_value.csv"))

    fig = px.line(df, x="date", y="account_value", title='Portfolio Value during trading')
    fig.write_html(os.path.join(result_folder,"result.html"))
    fig.write_image(os.path.join(result_folder,"result.png"))

    #fig.show()

def visualize(config,result_folder,with_Baseline = False):
    #result_folder = config["RESULT_FOLDER"] + '/' + scenario
    df = {"date": None}
    for agent_name in config['AGENTS']:
        temp = pd.read_csv(os.path.join(result_folder, agent_name + "/account_value.csv"))
        if df['date'] == None:
            df['date'] = temp['date'].tolist()
        df[agent_name] = temp['account_value'].tolist()

    if with_Baseline == True:
        Baseline_name = os.path.split(config['BASELINE_PATH'])[-1][:-4]
        Baseline = pd.read_csv(config['BASELINE_PATH']).sort_values(['date'])
        #VNI = VNI[(VNI['Date'] <= max(df['date'])) & (VNI['Date'] >= min(df['date']))]
        Baseline = Baseline[Baseline['date'].isin(df['date'])]
        df[Baseline_name] = Baseline['close'].tolist()

    for col in df.keys():
        print("len(df[{}]) = {}".format(col,len(df[col])))

    df = pd.DataFrame.from_dict(df)
    
    # fig = px.line(df, x="date", y=df.keys()[1:], title='Portfolio Value during trading')
    # fig = px.line(df, x="date", y=df['VNIndex'])
    # fig.show()


    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for agent_name in config['AGENTS']:

        fig.add_trace(
        go.Scatter(x=df["date"], y=df[agent_name], name=agent_name),
        secondary_y=False,
        )
    if with_Baseline == True:
        fig.add_trace(
            go.Scatter(x=df["date"], y=df[Baseline_name], name=Baseline_name),
            secondary_y=True,
        )

    # Add figure title
    fig.update_layout(
        title_text="Porforlio Value during trading      Experiment name: " + ' '.join(result_folder.split('/')[-3:])
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Date")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Portfolio Value</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>VNIndex Score</b>", secondary_y=True)

    #fig.show()

    fig.write_html(os.path.join(result_folder,"result.html"))
    fig.write_image(os.path.join(result_folder,"result.png"))

    return df


def _create_trade_annotations(sell_buy_log):
    annotations = []
    for index, row in sell_buy_log.iterrows():
        date = row['date']
        price = row['price']
        action = row['action']

        if action != 0:
            if action < 0:   # Sell action
                color = 'FireBrick'
                ay = -15
                text_info = dict(
                    datetime=date,
                    action_name='SELL',
                    qty = -1 * action,
                    price = price
                )
            elif action > 0:
                color = 'DarkGreen'
                ay = 15
                text_info = dict(
                    datetime=date,
                    action_name='BUY',
                    qty = action,
                    price = price
                )
            
            hovertext = 'Date: {datetime}<br>' \
                        'Action: {action_name} <br>' \
                        'Amount: {qty} <br>' \
                        'Price: {price} <br>'.format(**text_info)

            annotations += [go.layout.Annotation(
                x=date, y=price,
                ax=0, ay= ay, xref='x1', yref='y1', showarrow=True,
                arrowhead=2, arrowcolor=color, arrowwidth=4,
                arrowsize=0.8, hovertext=hovertext, opacity=0.6,
                hoverlabel=dict(bgcolor=color)
            )]

    return tuple(annotations)

def visualize_trading_action(begin_trade, end_trade,scenario,config):
    data = pd.read_csv(config['DATA_PATH'])
    data = data[(data.date >= begin_trade) & (data.date < end_trade)][['high', 'low', 'open', 'close', 'volume', 'tic', 'date']]
    for agent_name in config['AGENTS']:
        
        actions = pd.read_csv(config["RESULT_FOLDER"] + scenario + '/' + agent_name + "/sell_buy.csv")
        os.makedirs(config["RESULT_FOLDER"] + scenario + '/' + agent_name + "/trading_action/html", exist_ok=True)
        os.makedirs(config["RESULT_FOLDER"] + scenario + '/' + agent_name + "/trading_action/png", exist_ok=True)

        for tic_name in data.tic.unique():
            tic_prices = data[data.tic == tic_name]
            fig = go.Figure(data=[go.Candlestick(x=tic_prices['date'],
                            open=tic_prices['open'],
                            high=tic_prices['high'],
                            low=tic_prices['low'],
                            close=tic_prices['close'])])

            sell_buy_log = pd.DataFrame({'date': list(tic_prices['date']), 'price': list(tic_prices['close']), 'action': list(actions[tic_name])})
            fig.layout.annotations += _create_trade_annotations(sell_buy_log)
            #fig.show()
            fig.write_html(os.path.join(config["RESULT_FOLDER"] + scenario + '/' + agent_name + "/trading_action","html/{}.html".format(tic_name)))
            fig.write_image(os.path.join(config["RESULT_FOLDER"] + scenario + '/' + agent_name+ "/trading_action","png/{}.png".format(tic_name)))

