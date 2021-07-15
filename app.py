import dash #combo of flask, react.js, plotly.js
import dash_core_components as dcc #  create interactive components
import dash_bootstrap_components as dbc
import dash_html_components as html #access HTML tags
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import covasim as cv
cv.options.set(dpi=300, show=False, close=True, verbose=0)
import pandas as pd
import numpy as np



# ------------- intializing app -------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "COVID-19 Event Simulator"


# ------------- app input functions and values -------------
state_names = ["USA","Alaska", "Alabama", "Arkansas", "American Samoa", "Arizona", "California", "Colorado", "Connecticut", "District of Columbia", "Delaware", "Florida", "Georgia", "Guam", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Virgin Islands", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
event_settings = ["Indoor", "Outdoor", "Mixed"]
vax_options = [ "Default (assumes regional prevalence)", "Mandatory Vaccination", "No Vaccination"]
testing_options = ["No Testing", "Entry antigen", "Daily antigen", "PCR 2-day", "PCR 4-day"]

def OptionMenu(values, label, **kwargs):
    # options = [{"label": s.replace("_", " ").capitalize(), "value": s} for s in values]
    options = [{"label": s, "value": s} for s in values]
    kwargs["value"] = kwargs.get("value", values[0])

    if len(options) <= 4:
        component = dbc.RadioItems
        kwargs["inline"] = True
    else:
        component = dbc.Select
    if label == "Select U.S. State":
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

# ------------- vax and testing simulation functions -------------
"""
Function: test_intervention - creates intervention for a testing strategy
Note: this only works for cases generated at event 

To-Do
---------
For testing, esp daily antigen, shouldnt the days_testing argument = 1, instead of 0. We can specify an end date to ensure that testing stops on day 1? 
See documentation: # https://docs.idmod.org/projects/covasim/en/latest/covasim.interventions.html?highlight=test_num#covasim.interventions.test_num

Parameters
----------
test_del: Delay in test result
days_testing: Number of days in event testing susceptibles
subtarg: Indices of population to subtarget
sens: Sensitivity of test used

""" 

def test_intervention(test_del, days_testing, rapid, window, pars, sens=.984, subtarg=None):
    if (rapid):
        pars['pop_infected'] -= sens * pars['pop_infected']
    else:
       # Newly infected are those infected after test (assume even distribution of test date from 1 to 'window' days ago)
       # Assume 12 days infectious, all tests within 4.6 day period from exposed->infectious (max window 4)
       # These people will test negative even with a perfect test, because they were exposed after testing negative
        prevalence = 0.014 # TO BE UPDATED W/ STATE LEVEL DATA
        num_preinfectious = prevalence * pars['pop_size'] * sum([x*(1/(12*window)) for x in range(1,window+1)])

        # Finally, remove population that is detected by the reported tests
        pars['pop_infected'] += num_preinfectious - sens * pars['pop_infected']
        
    
    return cv.test_num(daily_tests=pars['pop_size']*days_testing, 
                       start_day=pars['start_day'], 
                       subtarget=subtarg,
                       symp_test=0,
                       sensitivity=sens,
                       test_delay=test_del), pars

"""
Function: vaccine_intervention: Creates intervention for a vaccine-based entry strategy. returns intervention

To-Do
---------
Check cummulative infections for mandatory vaxing - values seem way too high?

Parameters
----------
percent_vax: Percentage of overall population vaccinated
passport: If true, 100% of attendees must be vaccinated
efficacy_inf: Efficacy against infection
efficacy_symp: Efficacy against symptoms 
"""


def vaccine_intervention(percent_vax,efficacy_inf, efficacy_symp, pars, passport = False):
    pars['pop_infected'] -= (1-efficacy_inf) * percent_vax * pars['pop_infected']
    if passport:
        percent_vax = 1
    return cv.simple_vaccine(days=0, prob=percent_vax, rel_sus=efficacy_inf, rel_symp=efficacy_symp), pars


# ------------- Build component parts -------------
# avp_graph = dcc.Graph(id="avp_graph", style={"height": "500px"})
# loc_graph = dcc.Graph(id="loc_graph", style={"height": "500px"})
div_alert = dbc.Spinner(html.Div(id="alert-msg"))
# query_card = dbc.Card(
#     [
#         html.H4("Auto-generated SnowSQL Query", className="card-title"),
#         dcc.Markdown(id="sql-query"),
#     ],
#     body=True,
# )

controls = [
    OptionMenu(id="location", label="Select U.S. State", values=state_names),
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
    OptionMenu(id="event_setting", label="Event Setting", values=event_settings),
    OptionMenu(id="test_setting", label="Testing Options", values=testing_options),
    OptionMenu(id="vax_setting", label="Vaccination Requirements", values=vax_options),
    # SwitchInput(),
    dbc.Button("Run Simulation", color="primary", id="button-train", n_clicks=0),
    # dbc.Spinner(children=[avp_graph]),
    # dbc.Spinner(html.Div(id="loading-output")),
]

# interventions = [
#     OptionMenu(id="purpose", label="Purpose", values=purposes),
#     dbc.Button("Run Simulation", color="primary", id="button-train"),
# ]


# ------------- define app layout -------------
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
                    html.P("This tool helps you estimate the COVID risk of holding in-person events and find effective safety measures for minimizing transmission. You can compare multiple custom strategies on the same plot. For example, you can compare hosting your event in North Carolina requiring masks to in North Dakota requiring vaccines.")
                    # html.P("This conversion happens behind the scenes by Dash's JavaScript front-end")
                ])
            ])
        )),
        dbc.Row(
            [
                dbc.Col([dbc.Card(controls, body=True), div_alert], md=3),
                dbc.Col(dbc.Spinner(children=[dcc.Graph(id="avp_graph", style={"height": "500px"})], size="sm", color="primary")),
                dbc.Col(dbc.Spinner(children=[dcc.Graph(id="loc_graph", style={"height": "500px"})], size="sm", color="primary")),
                # dbc.Col([avp_graph]),
                # dbc.Col([loc_graph]),
                # dbc.Col(dcc.Graph(id="coef-graph", style={"height": "800px"}), md=5),
            ]
        ),
        # dbc.Row(
        #     [
        #         dbc.Col(html.Div(""), md=3),
        #         dbc.Col([loc_graph], md=5),
        #         # dbc.Col([avp_graph, query_card], md=4),
        #         # dbc.Col(dcc.Graph(id="coef-graph", style={"height": "800px"}), md=5),
        #     ] #,
        #     # justify="center",
        # ),
    ],
    style={"margin": "auto"},
)

# ------------- Define callbacks -------------
@app.callback(
    # [
        # Output("alert-msg", "children"),
        # Output("loading-output", "children"),
        Output("avp_graph", "figure"),
        Output("loc_graph", "figure"),
        # Output("coef-graph", "figure"),
        # Output("sql-query", "children"),
    # ],
    [Input("button-train", "n_clicks")],
    [
        State("location", "value"),
        State("event_duration", "value"),
        State("num_people", "value"),
        State("event_setting", "value"),
        State("test_setting", "value"),
        State("vax_setting", "value")
    ],
)
def run_sim(n_clicks, location, event_duration, num_people, event_setting, test_setting, vax_setting):
# def run_sim(event_duration, num_people, location, event_environment=None, mask_wearing=False, test_type=None, use_vaccines=True, mandatory_vax=False):
    if n_clicks < 1: 
        avp_fig = go.Figure()
        return avp_fig, avp_fig
    # ----- basic event characteristics ----- 
    event_duration = event_duration if event_duration != None else 1 #temp fix for misfiring nclicks
    num_people = num_people if num_people != None else 1
    location = "USA-" + location if location != "USA" else "USA"
    event_environment= None if event_setting == "Mixed" else event_setting
    mask_wearing=False
    test_type= None if test_setting == "No Testing" else test_setting
    use_vaccines=False if vax_setting == "No Vaccination" else True
    mandatory_vax=True if vax_setting == "Mandatory Vaccination" else False
    # print("test_setting: ", test_setting)
    # print("vax_setting: ", vax_setting)
    # print("event_environment:", event_environment, 
    # "test_type:", test_type, "use_vaccines:", use_vaccines, "mandatory_vax:", mandatory_vax)

    # ----- Point estimates ----- 
    indoor_factor = 9.35
    outdoor_factor = .11 
    mask_factor = .56
    ventilation_factor = .69
    capacity_factor = .5
    start_day = '2021-07-01' # default start date
    prevalence = 0.014 # TO BE UPDATED W/ STATE LEVEL DATA
    variant_transmissibility = 1.67 
    susceptibility_proportion = 0.747 # TO BE UPDATED W/ STATE LEVEL DATA
    
    # ----- define parameters in covasim format ----- 
    pars = dict(
        pop_type = 'hybrid', # Use a more realistic population model
        pop_size = num_people,
        pop_infected = num_people*prevalence,
        start_day = start_day,
        n_days = event_duration,
        location = location # Case insensitive
    )
    
    # -----  model changes (variants, environment, NPIs, testing, vaccines) to beta on day 0 ----- 
    all_interventions = []
    
    # variant transmissibility
    variant_trans = cv.change_beta(0, variant_transmissibility*susceptibility_proportion)
    all_interventions.append(variant_trans)
    
    # environment
    if event_environment == "Indoor":
        environment = cv.change_beta(0, indoor_factor)
        all_interventions.append(environment)
    elif event_environment == "Outdoor":
        environment = cv.change_beta(0, outdoor_factor)
        all_interventions.append(environment)
    
    # mask wearing
    if mask_wearing:
        masks = cv.change_beta(0, mask_factor)
        all_interventions.append(masks)   
    
    # testing 
    """
    -   test_del: Delay in test result
    -   days_testing: Number of days in event testing susceptibles
    -   subtarg: Indices of population to subtarget
    -   sens: Sensitivity of test used
    """
    testing_scenarios = {
        # format = [test_del, days_testing, rapid, window]
        "Entry antigen":[0,0,True,0],
        "Daily antigen": [0,event_duration,True,0], #[0,7,True,0]
        "PCR 2-day":[0,0,False,2],
        "PCR 4-day":[0,0,False,4],        
    } 
    subtarg = None # Subtargetting of tests
    sens = .984 # Sensitivity of test used
    if test_type != None:
        test_int, pars = test_intervention(test_del=testing_scenarios[test_type][0], 
                                           days_testing=testing_scenarios[test_type][1], 
                                           rapid=testing_scenarios[test_type][2], 
                                           window=testing_scenarios[test_type][3], 
                                           pars=pars, 
                                           sens=sens)
        all_interventions.append(test_int)
        
    
    # vaccinations - assumes state level vax numbers (default), unless otherwise specified (either mandatory vax, or no vax)
    perc_vax = .481 # TO BE UPDATED W/ STATE LEVEL DATA
    passport = mandatory_vax # True if 100% of attendees must be vaccinated
    v_efficacy_inf = .1 # Vax efficacy against infection (e.g .2 == 80% efficacy)
    v_efficacy_symp = .06 # Vax efficacy against symptoms
    if use_vaccines:
        vc, pars = vaccine_intervention(perc_vax, v_efficacy_inf, v_efficacy_symp, pars, passport)
        all_interventions.append(vc)

    
    # ----- run simulation -----
    # print("Running SIMULATION")
    sim = cv.Sim(pars, interventions = all_interventions)
    msim = cv.MultiSim(sim)
    msim.run(n_runs=10)
    msim.mean()
    # msim.plot(to_plot=['new_infections', 'cum_infections'])
    sim_json = msim.to_json()

    # ----- plotting -----
    df = pd.DataFrame(list(zip(sim_json['results']['t'], 
                            sim_json['results']['new_infections'],
                            sim_json['results']['new_infections_low'],
                            sim_json['results']['new_infections_high'])),
                    columns =['dates','new_infections', 'new_infections_low', 'new_infections_high'])
    
    df_cum_infections = pd.DataFrame(list(zip(sim_json['results']['t'], 
                            sim_json['results']['cum_infections'],
                            sim_json['results']['cum_infections_low'],
                            sim_json['results']['cum_infections_high'])),
                    columns =['dates','cum_infections', 'cum_infections_low', 'cum_infections_high'])

    avp_fig = go.Figure([
        go.Scatter(
            x=df['dates'].tolist(),
            y=df['new_infections'].tolist(),
            line=dict(color='rgb(0,100,80)'),
            mode='lines',
            name="New Cases"
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
    #  https://stackoverflow.com/questions/55704058/plotly-how-to-set-the-range-of-the-y-axis
    avp_fig.update_layout(yaxis_range=[-0.5,max(max(df['new_infections_high'].tolist()), 8)+2],
    yaxis_title='Cases',
    title_text='New Daily Infections', title_x=0.5, title_y=0.875,
    hovermode="x",
    xaxis = dict(dtick = 1),
    showlegend=False                     
    )

    
    loc_fig = go.Figure([
        go.Scatter(
            x=df_cum_infections['dates'].tolist(),
            y=df_cum_infections['cum_infections'].tolist(),
            line=dict(color='rgb(0,100,80)'),
            mode='lines',
            name="Total Cases"
        ),
        go.Scatter(
            x=df_cum_infections['dates'].tolist()+df_cum_infections['dates'].tolist()[::-1], # x, then x reversed
            y=df_cum_infections['cum_infections_high'].tolist()+df_cum_infections['cum_infections_low'].tolist()[::-1], # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        )
    ])
    loc_fig.update_layout(
    # yaxis_range=[-0.5,max(max(df_cum_infections['cum_infections_high'].tolist()), 8)+2],
    yaxis_title='Cases',
    title_text='Cummulative Infections', title_x=0.5, title_y=0.875,
    hovermode="x",
    xaxis = dict(dtick = 1),
    showlegend=False                     
    )
    # avp_fig.show()
    # loc_fig.show()
    return avp_fig, loc_fig

if __name__ == "__main__":
    app.run_server(debug=True)