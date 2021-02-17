#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from textwrap import dedent as d

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objects as go
from decimal import *

# import the css template, and pass the css template into dash
external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']
app=dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title="Journal Topic Modelling Visualisation"


def Visualise_3D_Network(
        nodes,
        node_title,
        data_x,
        data_y,
        z_column,
        embeddings_x=None,
        embeddings_y=None,
        labels = None,
        edges = '',
        edge_influential = '',
        edge_source = '',
        edge_target = '',
        scale_by=2,
        main_article_value=0
):
    if isinstance(nodes, str) and nodes.index('csv'):
        nodes=pd.read_csv(nodes)

    main_article = nodes.loc[main_article_value]

    print('Preparing Data for Visualisation...')
    data=nodes[nodes['Abstract'] != 'Empty']  # Drop empty rows from input

    # Separate References (articles dated before main article date) and Citations (articles dated after)

    references=data[data[z_column]<=data.loc[0, z_column]]
    references=references.iloc[1:]  # Drop main article (this will be added separately)

    # To be used for plotting without colour change relating to clusters/outliers
    # ref_node_x = references[data_x]
    # ref_node_y = references[data_y]
    # ref_node_z = references[z_column]
    # ref_node_label = references[labels]
    # ref_node_title = references['Title']
    # ref_node_doi = references['DOI']

    citations=data[data[z_column]>=data.loc[0, z_column]]
    citations=citations.iloc[1:]  # Drop main article (this will be added separately)

    # To be used for plotting without colour change relating to clusters/outliers
    # cit_node_x = citations[data_x]
    # cit_node_y = citations[data_y]
    # cit_node_z = citations[z_column]
    # cit_node_label = citations[labels]
    # cit_node_title = citations['Title']
    # cit_node_doi = citations['DOI']

    # |||| Clusters & Outliers |||| #

    # References #
    ref_cluster=references[references[labels] != -1][labels]
    ref_cluster_unique = ref_cluster.unique()
    
    # |||| Cluster data gen done in cluster loop, to enable customer colours/values |||| #
    # ref_cluster_x = ref_cluster[data_x].tolist()
    # ref_cluster_y = ref_cluster[data_y].tolist()
    # ref_cluster_z = ref_cluster[z_column].tolist()
    # ref_cluster_labels = ref_cluster[labels].tolist()
    # ref_cluster_title = ref_cluster['Title'].tolist()
    # ref_cluster_doi=ref_cluster['DOI'].tolist()

    ref_outlier=references[references[labels] == -1]

    ref_outlier_x=ref_outlier[data_x].tolist()
    ref_outlier_y=ref_outlier[data_y].tolist()
    ref_outlier_z=ref_outlier[z_column].tolist()
    ref_outlier_labels=ref_outlier[labels].tolist()
    ref_outlier_title=ref_outlier['Title'].tolist()
    ref_outlier_abstract = ref_outlier['Abstract'].tolist()
    ref_outlier_doi=ref_outlier['DOI'].tolist()

    # Citations #
    cit_cluster=citations[citations[labels] != -1][labels]
    cit_cluster_unique=cit_cluster.unique()

    # |||| Cluster data gen done in cluster loop, to enable customer colours/values |||| #
    # cit_cluster_x = cit_cluster[data_x].tolist()
    # cit_cluster_y = cit_cluster[data_y].tolist()
    # cit_cluster_z = cit_cluster[z_column].tolist()
    # cit_cluster_labels = cit_cluster[labels].tolist()
    # cit_cluster_title = cit_cluster['Title'].tolist()
    # cit_cluster_doi = cit_cluster['DOI'].tolist()

    cit_outlier=citations[citations[labels] == -1]

    cit_outlier_x=cit_outlier[data_x].tolist()
    cit_outlier_y=cit_outlier[data_y].tolist()
    cit_outlier_z=cit_outlier[z_column].tolist()
    cit_outlier_labels=cit_outlier[labels].tolist()
    cit_outlier_title=cit_outlier['Title'].tolist()
    cit_outlier_abstract=cit_outlier['Abstract'].tolist()
    cit_outlier_doi=cit_outlier['DOI'].tolist()


    items_to_trace = []
    #########################################################################################################################
    # # Creation of Lists of Influential Papers  (Separate lists to enable customisation)
    # list_of_influential_reference_indexes=[]
    # list_of_influential_citation_indexes=[]
    #
    #
    # # Filter for above list population
    # for i, node_value in enumerate(nodes):
    #     if nodes[i, 'Year']<=nodes[0, 'Year']:  # filter to identify if reference
    #         paper_id=nodes[i, 'Paper_Id']
    #         if edges[edge_source] == paper_id and edges[edge_influential] == True:
    #             list_of_influential_reference_indexes.append(nodes.index[(nodes['Source'] == paper_id)])
    #
    #     if nodes[i, 'Year']<=nodes[0, 'Year']:
    #         paper_id=nodes[i, 'Paper_Id']
    #         if edges[edge_target] == paper_id and edges[edge_influential] == True:  # filter to identity id in edges
    #             list_of_influential_citation_indexes.append(nodes.index[(nodes['Target'] == paper_id)])

    x_coord = main_article[data_x]
    y_coord = main_article[data_y]
    z_coord = main_article[z_column]
    title = main_article[node_title]


    items_to_trace.append(go.Scatter3d(x=[Decimal(x_coord) ** Decimal(scale_by)],
                                       y=[Decimal(y_coord) ** Decimal(scale_by)],
                                       z=[z_coord],
                                       hovertext=title,
                                       # hoverinfo=abstract,
                                       mode='markers',
                                       # text=[],
                                       # textposition="bottom center",
                                       marker=dict(
                                           size=5,
                                           # color= nodes[outlier_z][i],
                                           # set color to an array/list of desired values
                                           color="black",  # choose a colorscale
                                           opacity=1)))

   # Reference Trace
    
    # |||| Outlier Reference Trace |||| #
    for i in range(len(ref_outlier_x)):

        x_coord=ref_outlier_x[i]
        y_coord=ref_outlier_y[i]
        z_coord=ref_outlier_z[i]

        title = ref_outlier_title[i]
        abstract = ref_outlier_abstract[i]
        cluster_label = ref_outlier_labels[i]
        doi = ref_outlier_doi[i]

        # Setting variables, to indicate influence
        opacity=1
        size=3

        items_to_trace.append(go.Scatter3d(x=[x_coord ** scale_by],
                                           y=[y_coord ** scale_by],
                                           z=[z_coord],
                                           hovertext=title,
                                           hoverinfo=abstract,
                                           mode='markers',
                                           # text=[],
                                           # textposition="bottom center",
                                           marker=dict(
                                               size=size,
                                               # color= nodes[outlier_z][i],
                                               cmax=main_article[z_column]+20,
                                               cmin=1950,
                                               color=[z_coord],
                                               # set color to an array/list of desired values
                                               colorscale="inferno",  # choose a colorscale
                                               opacity=opacity)))

        ref_outlier_edge_x = []
        ref_outlier_edge_y = []
        ref_outlier_edge_z = []

        x0 = ref_outlier_x[i]
        x1 = main_article[data_x]

        y0 = ref_outlier_y[i]
        y1 = main_article[data_y]

        z0 = ref_outlier_z[i]
        z1 = main_article[z_column]

        ref_outlier_edge_x.append(x0 ** scale_by)
        ref_outlier_edge_x.append(x1 ** scale_by)
        ref_outlier_edge_x.append(None)

        ref_outlier_edge_y.append(y0 ** scale_by)
        ref_outlier_edge_y.append(y1 ** scale_by)
        ref_outlier_edge_y.append(None)

        ref_outlier_edge_z.append(z0)
        ref_outlier_edge_z.append(z1)
        ref_outlier_edge_z.append(None)

        ref_outlier_edge_trace=go.Scatter3d(x=ref_outlier_edge_x,
                                          y=ref_outlier_edge_y,
                                          z=ref_outlier_edge_z,
                                          line=dict(width=1,
                                                    # cmax=main_article_year,
                                                    # cmin=1950,
                                                    color='black'),
                                          # colorscale='blugrn'),
                                          hoverinfo='none',
                                          mode='lines')

        items_to_trace.append(ref_outlier_edge_trace)

        
    # |||| Cluster Reference Trace |||| #
    for i, item in enumerate(ref_cluster_unique):
        
        ref_cluster=references[references[labels] == item]

        ref_cluster_x = ref_cluster[data_x].tolist()
        ref_cluster_y = ref_cluster[data_y].tolist()
        ref_cluster_z = ref_cluster[z_column].tolist()
        ref_cluster_labels = ref_cluster[labels].tolist()
        ref_cluster_title = ref_cluster['Title'].tolist()
        ref_cluster_abstract=ref_cluster['Abstract'].tolist()
        ref_cluster_doi=ref_cluster['DOI'].tolist()

        for i in range(len(ref_cluster_x)):
            x_coord=Decimal(ref_cluster_x[i]) ** Decimal(scale_by)
            y_coord=Decimal(ref_cluster_y[i]) ** Decimal(scale_by)
            z_coord=Decimal(ref_cluster_z[i])
            title = ref_cluster_title[i]
            abstract = ref_cluster_abstract[i]
            cluster_label = ref_cluster_labels[i]
            doi = ref_cluster_doi[i]

    
            # Setting variables, to indicate influence
            opacity=1
            size=3
    
            items_to_trace.append(go.Scatter3d(x=[x_coord],
                                               y=[y_coord],
                                               z=[z_coord],
                                               hovertext=title,
                                               # hoverinfo= str(abstract),
                                               mode='markers',
                                               # text=[],
                                               # textposition="bottom center",
                                               marker=dict(
                                                   size=size,
                                                   # color= nodes[outlier_z][i],
                                                   cmax=main_article[z_column]+20,
                                                   cmin=1950,
                                                   color=[z_coord],
                                                   # set color to an array/list of desired values
                                                   colorscale="inferno",  # choose a colorscale
                                                   opacity=opacity)))

            ref_cluster_edge_x=[]
            ref_cluster_edge_y=[]
            ref_cluster_edge_z=[]

            x0=ref_cluster_x[i]
            x1=main_article[data_x]

            y0=ref_cluster_y[i]
            y1=main_article[data_y]

            z0=ref_cluster_z[i]
            z1=main_article[z_column]

            ref_cluster_edge_x.append(Decimal(x0) ** Decimal(scale_by))
            ref_cluster_edge_x.append(Decimal(x1) ** Decimal(scale_by))
            ref_cluster_edge_x.append(None)

            ref_cluster_edge_y.append(Decimal(y0) ** Decimal(scale_by))
            ref_cluster_edge_y.append(Decimal(y1) ** Decimal(scale_by))
            ref_cluster_edge_y.append(None)

            ref_cluster_edge_z.append(z0)
            ref_cluster_edge_z.append(z1)
            ref_cluster_edge_z.append(None)

            ref_cluster_edge_trace=go.Scatter3d(x=ref_cluster_edge_x,
                                                y=ref_cluster_edge_y,
                                                z=ref_cluster_edge_z,
                                                line=dict(width=1,
                                                          # cmax=main_article_year,
                                                          # cmin=1950,
                                                          color='orange'),
                                                # colorscale='blugrn'),
                                                hoverinfo='none',
                                                mode='lines')

            items_to_trace.append(ref_cluster_edge_trace)

        ref_cluster_x=[Decimal(x) ** Decimal(scale_by) for x in ref_cluster_x]
        ref_cluster_y=[Decimal(x) ** Decimal(scale_by) for x in ref_cluster_y]

        mesh_colour=['red', 'blue', 'green', 'orange', 'purple', 'yellow','red','pink','lightblue']

        mesh_trace=go.Mesh3d(
            alphahull=1,
            name=i,
            opacity=0.3,
            color='orange',
            x=ref_cluster_x,
            y=ref_cluster_y,
            z=ref_cluster_z
        )
        items_to_trace.append(mesh_trace)



    # |||| Outlier Citation Trace |||| #
    for i in range(len(cit_outlier_x)):
        x_coord=cit_outlier_x[i]
        y_coord=cit_outlier_y[i]
        z_coord=cit_outlier_z[i]

        title=cit_outlier_title[i]
        abstract=cit_outlier_abstract[i]
        cluster_label=cit_outlier_labels[i]
        doi=cit_outlier_doi[i]

        # Setting variables, to indicate influence
        opacity=1
        size=3

        items_to_trace.append(go.Scatter3d(x=[x_coord ** scale_by],
                                           y=[y_coord ** scale_by],
                                           z=[z_coord],
                                           hovertext=title,
                                           hoverinfo=abstract,
                                           mode='markers',
                                           # text=[],
                                           # textposition="bottom center",
                                           marker=dict(
                                               size=size,
                                               # color= nodes[outlier_z][i],
                                               cmax=main_article[z_column]+20,
                                               cmin=main_article[z_column],
                                               color=[z_coord],
                                               # set color to an array/list of desired values
                                               colorscale='orange',  # choose a colorscale
                                               opacity=opacity)))
        cit_outlier_edge_x=[]
        cit_outlier_edge_y=[]
        cit_outlier_edge_z=[]

        x0=cit_outlier_x[i]
        x1=main_article[data_x]

        y0=cit_outlier_y[i]
        y1=main_article[data_y]

        z0=cit_outlier_z[i]
        z1=main_article[z_column]

        cit_outlier_edge_x.append(Decimal(x0) ** Decimal(scale_by))
        cit_outlier_edge_x.append(Decimal(x1) ** Decimal(scale_by))
        cit_outlier_edge_x.append(None)

        cit_outlier_edge_y.append(Decimal(y0) ** Decimal(scale_by))
        cit_outlier_edge_y.append(Decimal(y1) ** Decimal(scale_by))
        cit_outlier_edge_y.append(None)

        cit_outlier_edge_z.append(z0)
        cit_outlier_edge_z.append(z1)
        cit_outlier_edge_z.append(None)

        cit_outlier_edge_trace=go.Scatter3d(x=cit_outlier_edge_x,
                                          y=cit_outlier_edge_y,
                                          z=cit_outlier_edge_z,
                                          line=dict(width=1,
                                                    # cmax=main_article_year,
                                                    # cmin=1950,
                                                    color='orange'),
                                          # colorscale='blugrn'),
                                          hoverinfo='none',
                                          mode='lines')

        items_to_trace.append(cit_outlier_edge_trace)

    # |||| Cluster Citation Trace |||| #
    for i, item in enumerate(cit_cluster_unique):

        cit_cluster=citations[citations[labels] == item]

        cit_cluster_x=cit_cluster[data_x].tolist()
        cit_cluster_y=cit_cluster[data_y].tolist()
        cit_cluster_z=cit_cluster[z_column].tolist()
        cit_cluster_labels=cit_cluster[labels].tolist()
        cit_cluster_title=cit_cluster['Title'].tolist()
        cit_cluster_abstract=cit_cluster['Abstract'].tolist()
        cit_cluster_doi=cit_cluster['DOI'].tolist()

        for i in range(len(cit_cluster_x)):
            x_coord=Decimal(cit_cluster_x[i]) ** Decimal(scale_by)
            y_coord=Decimal(cit_cluster_y[i]) ** Decimal(scale_by)
            z_coord=cit_cluster_z[i]
            title=cit_cluster_title[i]
            abstract=cit_cluster_abstract[i]
            cluster_label=cit_cluster_labels[i]
            doi=cit_cluster_doi[i]

            # Setting variables, to indicate influence
            opacity=1
            size=3

            items_to_trace.append(go.Scatter3d(x=[x_coord],
                                               y=[y_coord],
                                               z=[z_coord],
                                               hovertext=title,
                                               # hoverinfo=abstract,
                                               mode='markers',
                                               # text=[],
                                               # textposition="bottom center",
                                               marker=dict(
                                                   size=size,
                                                   # color= nodes[outlier_z][i],
                                                   cmax=main_article[z_column]+20,
                                                   cmin=main_article[z_column],
                                                   color=[z_coord],
                                                   # set color to an array/list of desired values
                                                   colorscale='blugrn',  # choose a colorscale
                                                   opacity=opacity)))

            cit_cluster_edge_x=[]
            cit_cluster_edge_y=[]
            cit_cluster_edge_z=[]

            x0=cit_cluster_x[i]
            x1=main_article[data_x]

            y0=cit_cluster_y[i]
            y1=main_article[data_y]

            z0=cit_cluster_z[i]
            z1=main_article[z_column]

            cit_cluster_edge_x.append(Decimal(x0) ** Decimal(scale_by))
            cit_cluster_edge_x.append(Decimal(x1) ** Decimal(scale_by))
            cit_cluster_edge_x.append(None)

            cit_cluster_edge_y.append(Decimal(y0) ** Decimal(scale_by))
            cit_cluster_edge_y.append(Decimal(y1) ** Decimal(scale_by))
            cit_cluster_edge_y.append(None)

            cit_cluster_edge_z.append(z0)
            cit_cluster_edge_z.append(z1)
            cit_cluster_edge_z.append(None)

            cit_cluster_edge_trace=go.Scatter3d(x=cit_cluster_edge_x,
                                                y=cit_cluster_edge_y,
                                                z=cit_cluster_edge_z,
                                                line=dict(width=1,
                                                          # cmax=main_article_year,
                                                          # cmin=1950,
                                                          color='lightblue'),
                                                # colorscale='blugrn'),
                                                hoverinfo='none',
                                                mode='lines')

            items_to_trace.append(cit_cluster_edge_trace)

        cit_cluster_x_list=[Decimal(x) ** Decimal(scale_by) for x in cit_cluster_x]
        cit_cluster_y_list=[Decimal(x) ** Decimal(scale_by) for x in cit_cluster_y]

        mesh_colour=['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'orange', 'pink', 'lightblue', 'blue', 'green', 'orange', 'purple', 'yellow', 'orange', 'pink', 'lightblue']

        mesh_trace=go.Mesh3d(
            alphahull=1,
            name=i,
            opacity=0.3,
            color='lightblue',
            x=cit_cluster_x_list,
            y=cit_cluster_y_list,
            z=cit_cluster_z
        )
        items_to_trace.append(mesh_trace)

    #################################################################################################################################################################
    figure={
        "data": items_to_trace,
        "layout": go.Layout(title='Journal Network Visualization', showlegend=False, hovermode='closest',
                            margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                            xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False, 'type': 'log'},
                            yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            height=750,
                            clickmode='event+select',
                            scene=dict(zaxis = dict(title_text='Publish Date',
                                                    type='log',
                                                    # autorange= 'reversed',
                                                    # mirror = 'ticks'
                                                    # tickmode = 'array',
                                                    # tickvals = [2020, 1950],
                                                    # ticktext = ['pizza', 'pie'],
                                                    ),
                                       aspectmode='manual',
                                       aspectratio=dict(x=1.5, y=1.5, z=1.5))

                            # annotations=[
                            #     dict(
                            #         ax=(G.nodes[edge[0]]['pos'][0] + G.nodes[edge[1]]['pos'][0]) / 2,
                            #         ay=(G.nodes[edge[0]]['pos'][1] + G.nodes[edge[1]]['pos'][1]) / 2, axref='x', ayref='y',
                            #         x=(G.nodes[edge[1]]['pos'][0] * 3 + G.nodes[edge[0]]['pos'][0]) / 4,
                            #         y=(G.nodes[edge[1]]['pos'][1] * 3 + G.nodes[edge[0]]['pos'][1]) / 4, xref='x', yref='y',
                            #         showarrow=True,
                            #         arrowhead=3,
                            #         arrowsize=4,
                            #         arrowwidth=1,
                            #         opacity=1
                            #     ) for edge in G.edges]
                            )}

    return figure


#### Plotly Colour Gradients ####
# aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
# 'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
# 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
# 'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
# 'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
# 'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
# 'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
# 'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
# 'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
# 'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
# 'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
# 'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
# 'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
# 'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
# 'ylorrd'].


def Visualise_on_Local_Host(
        nodes,
        node_title,
        data_x,
        data_y,
        z_column,
        embeddings_x=None,
        embeddings_y=None,
        labels='',
        edges='',
        edge_influential='',
        edge_source='',
        edge_target='',
        scale_by=2,
        main_article_value=0
):
    ######################################################################################################################################################################
    # styles: for right side hover/click component
    styles={
        'pre': {
            'border': 'thin lightgrey solid',
            'overflowX': 'scroll'
        }
    }

    app.layout=html.Div([
        #########################Title
        html.Div([html.H1("Journal Vis")],
                 className="row",
                 style={'textAlign': "center"}),
        #############################################################################################define the row
        html.Div(
            className="row",
            children=[

                ############################################middle graph component
                html.Div(
                    className="eight columns",
                    children=[dcc.Graph(id="my-graph",
                                        figure=Visualise_3D_Network(nodes,
                                                                    node_title,
                                                                    data_x,
                                                                    data_y,
                                                                    z_column,
                                                                    labels=labels
                                                                    ))],
                ),

                #########################################right side two output component
                html.Div(
                    className="two columns",
                    children=[
                        html.Div(
                            className='twelve columns',
                            children=[
                                dcc.Markdown(d("""
                                **Hover Data**
    
                                Hover over the nodes to see key information.
                                """)),
                                html.Pre(id='hover-data', style=styles['pre'])
                            ],
                            style={'height': '400px'}),

                        html.Div(
                            className='twelve columns',
                            children=[
                                dcc.Markdown(d("""
                                **Click Data**
    
                                Click on journals in the graph for more information.
                                """)),
                                html.Pre(id='click-data', style=styles['pre'])
                            ],
                            style={'height': '400px'})
                    ]
                )
            ]
        )
    ])


    ################################### callback for additional components
    # @app.callback(
    #     dash.dependencies.Output('my-graph', 'figure'))
    # def update_output():
    #
    #     return Visualise_Network_3D("test.csv",
    #                                 'Title',
    #                                 'Outliers_x',
    #                                 'Outliers_y',
    #                                 'Year',
    #                                 'Clustered_x',
    #                                 'Clustered_y',
    #                                 'Year',
    #                                 'Clustered_labels',
    #                                 "Data/node.csv",
    #                                 'Influential',
    #                                 'Source',
    #                                 'Target'
    #                                 )
    #     # to update the global variable of YEAR and ACCOUNT


    ################################ callback for right side components
    @app.callback(
        dash.dependencies.Output('hover-data', 'children'),
        [dash.dependencies.Input('my-graph', 'hoverData')])
    def display_hover_data(hoverData):
        return json.dumps(hoverData, indent=2)


    @app.callback(
        dash.dependencies.Output('click-data', 'children'),
        [dash.dependencies.Input('my-graph', 'clickData')])
    def display_click_data(clickData):
        return json.dumps(clickData, indent=2)


    if __name__ == '__main__':
        app.run_server(debug=True)

# Visualise_on_Local_Host('node.csv',
#                         'Title',
#                         'x data',
#                         'y data',
#                         'Year',
#                         labels='labels',
#                         )
