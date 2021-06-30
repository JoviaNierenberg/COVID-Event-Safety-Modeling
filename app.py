import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import covasim as cv
cv.options.set(dpi=300, show=False, close=True, verbose=0)
import pandas as pd
import numpy as np

data = pd.read_csv("avocado.csv")
data = data.query("type == 'conventional' and region == 'Albany'")
data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
data.sort_values("Date", inplace=True)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "COVID-19 Event Simulator"

# loan_min, loan_max = 10000, 20000
# Make some calculations based on value range retrieved
# loan_marks = loan_max // 4
# loan_min //= loan_marks
# inc_min, inc_max = 10000, 20000
# app_types = ["Individual", "Joints"]
# purposes =  ["Individual", "Jointss"]
# ownerships =  ["Individual", "Josints", "amazing", "testing", "many topions", "wow  "]
state_names = ["Alaska", "Alabama", "Arkansas", "American Samoa", "Arizona", "California", "Colorado", "Connecticut", "District ", "of Columbia", "Delaware", "Florida", "Georgia", "Guam", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Virgin Islands", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
event_settings = ["Indoor", "Outdoor", "Mixed"]
# extra functions
def OptionMenu(values, label, **kwargs):
    options = [{"label": s.replace("_", " ").capitalize(), "value": s} for s in values]
    kwargs["value"] = kwargs.get("value", values[0])

    if len(options) <= 4:
        component = dbc.RadioItems
        kwargs["inline"] = True
    else:
        component = dbc.Select
    if label == "Select U.S. State":
        pass
        return dbc.FormGroup([dbc.Label(label),component(placeholder="Select a U.S. State...", options=options, **kwargs)])
    return dbc.FormGroup([dbc.Label(label), component(options=options, **kwargs)])


def NumberInput(id_name, minval=1, maxval=7, label_text = None, instructions = None):
    # number_input = html.Div(
    # [
    #     html.P("Event Duration (Days)"),
    #     dbc.Input(type="number", min=0, max=10, step=1),
    # ],
    # id="styled-numeric-input",
    # )
    text_input = dbc.FormGroup(
        [
            dbc.Label(label_text),
            # dbc.Input(placeholder="Input goes here...", type="text"),
            dbc.Input(id = id_name, type="number", min=minval, max=maxval, step=1),
            dbc.FormText(instructions),
        ]
    )

    return text_input



def SwitchInput():
    switches = dbc.FormGroup(
        [
            dbc.Label("Non-Pharmaceutical Interventions"),
            dbc.Checklist(
                options=[
                    {"label": "Mask wearing", "value": 1},
                    {"label": "Event capacity reduction", "value": 2},
                    {"label": "Improved ventilation indoors", "value": 3},
                    # {"label": "Disabled Option", "value": 3, "disabled": True},
                ],
                value=[1],
                id="switches-input",
                switch=True,
            ),
        ]
    )
    return switches

def CustomRangeSlider(values, label, **kwargs):
    values = sorted(values)
    marks = {i: f"{i//1000}k" for i in values}

    return dbc.FormGroup(
        [
            dbc.Label(label),
            dcc.RangeSlider(
                min=values[0],
                max=values[-1],
                step=1000,
                value=[values[1], values[-2]],
                marks=marks,
                **kwargs,
            ),
        ]
    )


def get_unique(connection, db, col):
    query = f"""
    SELECT DISTINCT {col}
    FROM {db}.PUBLIC.LOAN_CLEAN;
    """
    return [x[0] for x in connection.execute(query).fetchall()]


def get_range(connection, db, col):
    query = f"""
    SELECT MIN({col}), MAX({col})
    FROM {db}.PUBLIC.LOAN_CLEAN;
    """
    return connection.execute(query).fetchall()[0]  





# --------------

# Build component parts
avp_graph = dcc.Graph(id="avp-graph", style={"height": "500px"})
loc_graph = dcc.Graph(id="loc_graph", style={"height": "500px"})
div_alert = dbc.Spinner(html.Div(id="alert-msg"))
# query_card = dbc.Card(
#     [
#         html.H4("Auto-generated SnowSQL Query", className="card-title"),
#         dcc.Markdown(id="sql-query"),
#     ],
#     body=True,
# )

controls = [
    OptionMenu(id="us-state-location", label="Select U.S. State", values=state_names),
    NumberInput(id_name= "event_duration", minval=1, maxval=7, label_text="Event Duration", instructions = "Choose between 1-7 days"),
    NumberInput(id_name="num_people", minval=1, maxval=100000, label_text="Number of Participants" ),
    # CustomRangeSlider(
    #     id="loan-amount",
    #     label="Loan Amount($)",
    #     values=range(loan_min, loan_max + 1, loan_marks),
    # ),
    # CustomRangeSlider(
    #     id="annual-income",
    #     label="Annual Income ($)",
    #     values=[0, 20000, 50000, 100000, 200000],
    # ),
    OptionMenu(id="event-settings", label="Event Setting", values=event_settings),
    # OptionMenu(id="purpose", label="Purpose", values=purposes),
    SwitchInput(),
    dbc.Button("Run Simulation", color="primary", id="button-train"),
]

# interventions = [
#     OptionMenu(id="purpose", label="Purpose", values=purposes),
#     dbc.Button("Run Simulation", color="primary", id="button-train"),
# ]


# Define Layout
app.layout = dbc.Container(
    fluid=True,
    children=[
        html.H1("COVID-19 Event Safety Modeling"),
        html.Hr(),
        # dbc.Row(dbc.Col(html.Div("Intro info can go here!"))),
        dbc.Row(dbc.Col(
            html.Div([
                # html.H2('Hello Dash'),
                html.Div([
                    html.P("This calculator lets you estimate COVID risk and find effective safety measures for customizable situations. Examples: how risky is a trip to my grocery store? What's the safest way to see a friend? How much would it help to wear a better mask at my workplace?")
                    # html.P("This conversion happens behind the scenes by Dash's JavaScript front-end")
                ])
            ])
        )),
        dbc.Row(
            [
                dbc.Col([dbc.Card(controls, body=True), div_alert], md=3),
                dbc.Col([avp_graph], md=5),
                # dbc.Col([avp_graph, query_card], md=4),
                # dbc.Col(dcc.Graph(id="coef-graph", style={"height": "800px"}), md=5),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(html.Div(""), md=3),
                dbc.Col([loc_graph], md=5),
                # dbc.Col([avp_graph, query_card], md=4),
                # dbc.Col(dcc.Graph(id="coef-graph", style={"height": "800px"}), md=5),
            ] #,
            # justify="center",
        ),
    ],
    style={"margin": "auto"},
)

@app.callback(
    # [
        # Output("alert-msg", "children"),
        Output("avp-graph", "figure"),
        # Output("coef-graph", "figure"),
        # Output("sql-query", "children"),
    # ],
    [Input("button-train", "n_clicks")],
    [
        State("event_duration", "value"),
        State("num_people", "value")
        # State("annual-income", "value"),
        # State("app-type", "value"),
        # State("home-ownership", "value"),
        # State("purpose", "value"),
    ],
)
def run_sim(n_clicks, event_duration, num_people):
   
    event_duration = event_duration if event_duration != None else 1
    num_people = num_people if num_people != None else 1
    print(num_people)

    # currently hard coded, need to change
    start_day = '2021-07-01'
    prevalence = 0.014 
    variant_transmissibility = 1.67 
    susceptibility_proportion = 0.747
    
    
    # define parameters in covasim format
    pars = dict(
        pop_type = 'hybrid', # Use a more realistic population model
        pop_infected = num_people*prevalence,
        start_day = start_day,
        n_days = event_duration
    )

    # sample output
    # x = np.arange(10)
    # avp_fig = go.Figure(data=go.Scatter(x=x, y=x**2))

    # model changes in beta on day 0
    beta_changes = cv.change_beta(0, variant_transmissibility*susceptibility_proportion)

    # run simulation
    sim = cv.Sim(pars, interventions = beta_changes)
    msim = cv.MultiSim(sim)
    msim.run(n_runs=10)
    msim.mean()
    # msim.plot(to_plot=['new_infections', 'cum_infections'])

    sim_json = msim.to_json()

    # testjson['results']['new_infections']
    # testjson['results']['new_infections_low']
    # testjson['results']['new_infections_high']
    df = pd.DataFrame(list(zip(sim_json['results']['date'], 
                            sim_json['results']['new_infections'],
                            sim_json['results']['new_infections_low'],
                            sim_json['results']['new_infections_high'])),
                    columns =['dates','new_infections', 'new_infections_low', 'new_infections_high'])

    print(df)
    # avp_fig = go.Figure(data=go.Scatter(x=df['dates'], y=df['new_infections']))

    # simple plot 
    avp_fig = go.Figure([
        go.Scatter(
            x=df['dates'].tolist(),
            y=df['new_infections'].tolist(),
            line=dict(color='rgb(0,100,80)'),
            mode='lines',
            name="New Daily Infections"
        ),
        go.Scatter(
            x=df['dates'].tolist()+df['dates'].tolist()[::-1], # x, then x reversed
            y=df['new_infections_high'].tolist()+df['new_infections_low'].tolist()[::-1], # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        )
    ])
    """
    avp_fig = go.Figure([
        go.Scatter(
            x=df['dates'].tolist(),
            y=df['new_infections'].tolist(),
            line=dict(color='rgb(0,100,80)'),
            mode='lines'
        ),
        go.Scatter(
            x=df['dates'].tolist()+df['dates'].tolist()[::-1], # x, then x reversed
            y=df['new_infections_high'].tolist()+df['new_infections_low'].tolist()[::-1], # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
       go.Scatter(
            x=df['dates'].tolist(),
            y=sim_json['results']['cum_infections'],
            line=dict(color='rgb(0,100,80)'),
            mode='lines'
        ),
        go.Scatter(
            x=df['dates'].tolist()+df['dates'].tolist()[::-1], # x, then x reversed
            y=sim_json['results']['cum_infections_high']+sim_json['results']['cum_infections_low'][::-1], # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        )
    ])

    """

    return avp_fig

if __name__ == "__main__":
    app.run_server(debug=True)