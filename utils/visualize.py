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

def visualize(result_folder, list_agent, with_VNI = False):
    df = {"date": None}
    for agent_name in list_agent:
        temp = pd.read_csv(os.path.join(result_folder, agent_name + "/account_value.csv"))
        if df['date'] == None:
            df['date'] = temp['date'].tolist()
        df[agent_name] = temp['account_value'].tolist()

    if with_VNI == True:
        VNI = pd.read_csv("Data/VNIndex.csv").sort_values(['Date'])
        #VNI = VNI[(VNI['Date'] <= max(df['date'])) & (VNI['Date'] >= min(df['date']))]
        VNI = VNI[VNI['Date'].isin(df['date'])]
        df['VNIndex'] = VNI['Price'].tolist()
        print(len(VNI))

    df = pd.DataFrame.from_dict(df)
    # fig = px.line(df, x="date", y=df.keys()[1:], title='Portfolio Value during trading')
    # fig = px.line(df, x="date", y=df['VNIndex'])
    # fig.show()


    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for agent_name in list_agent:

        fig.add_trace(
        go.Scatter(x=df["date"], y=df[agent_name], name=agent_name),
        secondary_y=False,
        )

    fig.add_trace(
        go.Scatter(x=df["date"], y=df['VNIndex'], name="VNIndex"),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(
        title_text="Porforlio Value during trading      Experiment name: " + ''.join(result_folder.split('/')[-3:])
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