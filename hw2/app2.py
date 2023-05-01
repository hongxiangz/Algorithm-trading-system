from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime, date, timedelta
import dash
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
import eikon as ek
# import refinitiv.dataplatform.eikon as ek
import refinitiv.data as rd
import blotter as blt
from dash import html
import base64
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

percentage = dash_table.FormatTemplate.percentage(3)

image_filename = '150041677385731_.pic.jpg'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())



controls = dbc.Card(
    [
        dbc.Row(html.Button('QUERY Refinitiv', id='run-query', n_clicks=0)),
        dbc.Row([
            html.H5('Asset:',
                    style={'display': 'inline-block', 'margin-right': 20}),
            dcc.Input(id='asset', type='text', value="IVV",
                      style={'display': 'inline-block',
                             'border': '1px solid black'}),
            dbc.Table(
                [
                    html.Thead(html.Tr([html.Th("Î±1"), html.Th("n1")])),
                    html.Tbody([
                        html.Tr([
                            html.Td(
                                dbc.Input(
                                    id='alpha1',
                                    type='number',
                                    value=-0.01,
                                    max=1,
                                    min=-1,
                                    step=0.01,
                                    style={'width':'auto'}
                                )
                            ),
                            html.Td(
                                dcc.Input(
                                    id='n1',
                                    type='number',
                                    value=3,
                                    min=1,
                                    step=1,
                                    style={'width':'auto'}
                                )
                            )
                        ])
                    ])
                ],
                bordered=True
            ),
            dbc.Table(
                [
                    html.Thead(html.Tr([html.Th("Î±2"), html.Th("n2")])),
                    html.Tbody([
                        html.Tr([
                            html.Td(
                                dbc.Input(
                                    id='alpha2',
                                    type='number',
                                    value=0.01,
                                    max=1,
                                    min=-1,
                                    step=0.01,
                                    style={'width':'auto'}
                                )
                            ),
                            html.Td(
                                dcc.Input(
                                    id='n2',
                                    type='number',
                                    value=5,
                                    min=1,
                                    step=1,
                                    style={'width':'auto'}
                                )
                            )
                        ])
                    ])
                ],
                bordered=True
            )
        ]),
        dbc.Row(html.Button('UPDATE BLOTTER', id='update-blotter', n_clicks=0)),
        dbc.Row([
            dcc.DatePickerRange(
                id='refinitiv-date-range',
                min_date_allowed = date(2015, 1, 1),
                max_date_allowed = datetime.now(),
                start_date = datetime.date(
                    datetime.now() - timedelta(days=3*365)
                ),
                end_date = datetime.now().date()
            )
        ])
    ],
    body=True,
)

app.layout = dbc.Container(
    [   html.H2('Created/Modified by: Xiaokuan Zhao, Rebecca Xiao, Xinyang Ding',
                style={'font-size': '36px', 'color': '#150',
                       'font-family': 'Times New Roman, sans-serif', 'text-align': 'center'}),
        dbc.Row(
            [
                dbc.Col(controls, md=4),
                dbc.Col(
                    # Put your reactive graph here as an image!
                    html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), alt="image",
                             style = {'width': '50%', 'height': '50%'}),
                    md = 10
                )
            ],
            align="center",
        ),
        html.H2('Trade Blotter:', style={'fontSize': '28px', 'color': '#150',
                       'fontFamily': 'Times New Roman, sans-serif'}),
        dash_table.DataTable(id = "blotter"),
    ],
    fluid=True
)
'''
@app.callback(
    Output("blotter","data"),
    Input("run-query", "n_clicks"),
    [State('refinitiv-date-range','start_date'),
     State('refinitiv-date-range', 'end_date'),
     State('alpha1', 'value'),
     State('n1', 'value'),
     State('alpha2', 'value'),
     State('n2', 'value'),
     State('asset', 'value'),
     ],
    prevent_initial_call=True
)
def query_and_make_blotter(n_clicks, start_date_str, end_date_str, alpha1, n1, alpha2, n2, asset_id):
    blt.query_data(start_date_str, end_date_str, asset_id)
    blotter = blt.make_blotter(alpha1, n1, alpha2, n2)
    return blotter.to_dict('records')
'''

@app.callback(
    Output("blotter","data"),
    [Input("update-blotter", "n_clicks"),
     Input("run-query", "n_clicks")],
    [State('refinitiv-date-range','start_date'),
     State('refinitiv-date-range', 'end_date'),
     State('alpha1', 'value'),
     State('n1', 'value'),
     State('alpha2', 'value'),
     State('n2', 'value'),
     State('asset', 'value'),
     ],
    prevent_initial_call = True
)

def combined_callback(btn1_clicks, btn2_clicks, start_date, end_date, alpha1, n1, alpha2, n2, asset_id):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if "run-query" in changed_id:
        blt.query_data(start_date, end_date, asset_id)
        blotter = blt.make_blotter(alpha1, n1, alpha2, n2)
        return blotter.to_dict('records')

    elif "update-blotter" in changed_id:
        blotter = blt.make_blotter(alpha1, n1, alpha2, n2)
        return blotter.to_dict('records')


if __name__ == '__main__':
    app.run_server(debug=True, port = 8050)