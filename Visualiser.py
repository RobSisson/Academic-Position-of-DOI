#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import networkx as nx
import plotly.graph_objs as go

import pandas as pd
from colour import Color
from datetime import datetime
from textwrap import dedent as d
import json

# import the css template, and pass the css template into dash
external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']
app=dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title="Journal Topic Modelling Visualisation"

import numpy as np

import plotly.graph_objects as go


def Column_Filter(
        df,
        column,
        value_to_remove='',
):
    values=df.loc[df[column] != value_to_remove, column].tolist()

    return values


# Visualize clusters
def Visualise_3D_Network(
        nodes,
        node_title,
        data_x,
        data_y,
        z_column,
        # embeddings_x,
        # embeddings_y,
        # outlier_x,
        # outlier_y,
        # outlier_z,
        # cluster_x,
        # cluster_y,
        # cluster_z,
        labels,
        edges = '',
        edge_influential = '',
        edge_source = '',
        edge_target = '',
        scale_by=2,
        main_article_value=0
):
    if isinstance(nodes, str) and nodes.index('csv'):
        print('Loading CSV as DataFrame...')
        nodes=pd.read_csv(nodes)

        print('Confirming if loaded correctly...')
        print(nodes[data_x][0:3])

    items_to_trace=[]  # contains edge_trace, node_trace, middle_node_trace

    x_coords = nodes[nodes[data_x] != 'Empty', data_x].tolist()

    x=nodes.index[nodes[data_x] != 'Empty'].tolist()


def Column_Filter(
        df,
        column,
        value_to_remove = '',
):
    values =  df[df[column] != value_to_remove].tolist()

    for i in range(0, len(x)):
        coords=nodes.loc[i, data_x]

        if i in y_outlier: y_coords.append(y_outlier_coords)
        if i in y_clustered: y_coords.append(y_clustered_coords)


    y_coords=nodes[nodes[data_y] != 'Empty', data_y].tolist()

    x_data_index = nodes.index[nodes[data_x] != 'Empty'].tolist()
    y_data_index = nodes.index[nodes[data_y] != 'Empty'].tolist()
    # x_embeddings_index = nodes.index[nodes[embeddings_x] != 'Empty'].tolist()
    # y_embeddings_index = nodes.index[nodes[embeddings_y] != 'Empty'].tolist()

    # z_coords=[] #  Not currently active as both outliers and clusters use the same z coord column ('Year')

    # if outliers and clusters are in seperate columns, use below script to merge
    # y_outlier=nodes.index[nodes[outlier_y].notna()].tolist()
    # y_clustered=nodes.index[nodes[cluster_y].notna()].tolist()
    #
    # length_y=len(y_outlier)+len(y_clustered)
    #
    # for i in range(0, length_y):
    #     y_outlier_coords=nodes.loc[i, outlier_y]
    #     y_clustered_coords=nodes.loc[i, cluster_y]
    #     if i in y_outlier: y_coords.append(y_outlier_coords)
    #     if i in y_clustered: y_coords.append(y_clustered_coords)

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

    ###############################################################################################################################################################
    # Reference Trace

    main_article_year=nodes.loc[main_article_value, 'Year']

    empty_references=nodes.index[nodes[data_y]=='Empty' and nodes[data_y]=='Empty'].to_list()

    print(type(nodes[z_column]))
    print('on one')
    print(type(main_article_year))

    dirty_reference_index=nodes.index[int(nodes[z_column])<=int(main_article_year)].to_list()
    dirty_reference_index.pop(main_article_value)

    # Removal of values with missing coordinates
    reference_index=[ref_index for ref_index in dirty_reference_index if ref_index not in empty_references]

    # print(reference_index)

    for i, item in enumerate(reference_index):
        if item != main_article_value:
            x_coord=x_coords[item]
            y_coord=y_coords[item]
            z_coord=nodes.loc[item, z_column]
            # print(z_coord)

            title=nodes.loc[item, node_title]

            # Setting variables, to indicate influence
            opacity=1
            size=3

            # Separate variable adjustments to enable future use
            # if item in list_of_influential_reference_indexes:
            #     opacity = 1
            #     size = 12
            #
            # if item in list_of_influential_citation_indexes:
            #     opacity=1
            #     size = 12

            items_to_trace.append(go.Scatter3d(x=[x_coord ** scale_by],
                                               y=[y_coord ** scale_by],
                                               z=[z_coord],
                                               hovertext=title,
                                               hoverinfo="text",
                                               mode='markers',
                                               # text=[],
                                               # textposition="bottom center",
                                               marker=dict(
                                                   size=size,
                                                   # color= nodes[outlier_z][i],
                                                   cmax=main_article_year+20,
                                                   cmin=1950,
                                                   color=[z_coord],
                                                   # set color to an array/list of desired values
                                                   colorscale="inferno",  # choose a colorscale
                                                   opacity=opacity)))

    # Outlier Trace

    # node_index = nodes.index[nodes[outlier_x].notna() &
    #                    nodes[outlier_y].notna() &
    #                    nodes[outlier_z].notna()].to_list()
    #
    # for i, item in enumerate(node_index):
    #
    #     x_coord =nodes.loc[item, outlier_x]
    #     y_coord =nodes.loc[item, outlier_y]
    #     z_coord =nodes.loc[item, outlier_z]
    #
    #     title = nodes.loc[item, node_title]
    #
    #     # Setting variables, to indicate influence
    #     opacity = 1
    #     size = 3
    #
    #     # Separate variable adjustments to enable future use
    #     # if item in list_of_influential_reference_indexes:
    #     #     opacity = 1
    #     #     size = 12
    #     #
    #     # if item in list_of_influential_citation_indexes:
    #     #     opacity=1
    #     #     size = 12
    #
    #     items_to_trace.append(go.Scatter3d(x=[x_coord],
    #                                              y=[y_coord],
    #                                              z=[z_coord],
    #                                              hovertext= title,
    #                                              hoverinfo="text",
    #                                              mode='markers',
    #                                              # text=[],
    #                                              # textposition="bottom center",
    #                                              marker=dict(
    #                                                  size=size,
    #                                                  # color= nodes[outlier_z][i],
    #                                                  cmax=2020,
    #                                                  cmin=1950,
    #                                                  color= [z_coord],
    #                                                  # set color to an array/list of desired values
    #                                                  colorscale="Viridis",  # choose a colorscale
    #                                                  opacity=opacity) ) )

    ################################################################################################################################################################
    # Colour if Clustered

    # Tracing of Clustered Nodes

    empty_citations=nodes.index[nodes[data_y]=='Empty' and nodes[data_y]=='Empty'].to_list()

    dirty_citation_index=nodes.index[int(nodes[z_column])>int(main_article_year)].to_list()
    dirty_citation_index.pop(main_article_value)

    # Removal of values with missing coordinates
    citation_index=[ref_index for ref_index in dirty_citation_index if ref_index not in empty_citations]

    # print(citation_index)
    # #
    # print(len(citation_index))
    # print(len(x_coords))

    for i, item in enumerate(citation_index):
        if item != main_article_value:
            x_coord=x_coords[item]
            y_coord=y_coords[item]
            z_coord=nodes.loc[item, z_column]

            # Setting variables, to indicate influence
            opacity=1
            size=3

            # Separate variable adjustments to enable future use
            # if item in list_of_influential_reference_indexes:
            #     opacity = 1
            #     size = 12
            #
            # if item in list_of_influential_citation_indexes:
            #     opacity=1
            #     size = 12
            title=nodes.loc[item, node_title]

            items_to_trace.append(go.Scatter3d(x=[x_coord ** scale_by],
                                               y=[y_coord ** scale_by],
                                               z=[z_coord],
                                               hovertext=title,
                                               hoverinfo="text",
                                               mode='markers',
                                               marker=dict(
                                                   size=size,
                                                   cmax=2022,
                                                   cmin=main_article_year,
                                                   color=[z_coord],
                                                   # set color to an array/list of desired values
                                                   colorscale='blugrn',  # choose a colorscale
                                                   opacity=opacity
                                               )))

        # aggrnyl
        # ', '
        # agsunset
        # ', '
        # algae
        # ', '
        # amp
        # ', '
        # armyrose
        # ', '
        # balance
        # ',
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

    # Colour Code Based on Clustering Combine with below meshing?
    # for i, cluster_label in enumerate(nodes[cluster_labels].unique().tolist()):
    #     cluster_type_index = nodes.index[(nodes[cluster_labels] == cluster_label) &
    #                                      nodes[cluster_x].notna() &
    #                                      nodes[cluster_y].notna() &
    #                                      nodes[cluster_z].notna() &
    #                                      nodes[]
    #     ].tolist()
    #     for i, item in enumerate(cluster_type_index):
    #         x_coord=nodes.loc[item, cluster_x]
    #         y_coord=nodes.loc[item, cluster_y]
    #         z_coord=nodes.loc[item, cluster_z]
    #
    #         # Setting variables, to indicate influence
    #         opacity = 1
    #         size = 4
    #
    #         # Separate variable adjustments to enable future use
    #         # if item in list_of_influential_reference_indexes:
    #         #     opacity = 1
    #         #     size = 12
    #         #
    #         # if item in list_of_influential_citation_indexes:
    #         #     opacity=1
    #         #     size = 12
    #         title=nodes.loc[item, node_title]
    #
    #         items_to_trace.append(go.Scatter3d(x=[x_coord],
    #                                              y=[y_coord],
    #                                              z=[z_coord],
    #                                              hovertext=title,
    #                                              hoverinfo="text",
    #                                              mode='markers',
    #                                              marker=dict(
    #                                                  size=size,
    #                                                  cmax=2020,
    #                                                  cmin=1950,
    #                                                  color=[z_coord],
    #                                                  # set color to an array/list of desired values
    #                                                  colorscale='sunset',  # choose a colorscale
    #                                                  opacity=opacity
    #                                              )))

    # ################################################################################################################################################################
    # # Cluster Mesh
    #
    # cluster_labels_list=[x for x in nodes[labels].unique().tolist() if str(x) != 'nan']
    #
    #
    # for i, cluster_label in enumerate(cluster_labels_list):
    #     cluster_index_list=nodes.index[nodes[labels] == cluster_label].to_list()
    #
    #     cluster_x_coord_list=nodes.loc[cluster_index_list, cluster_x].to_list()
    #     cluster_y_coord_list=nodes.loc[cluster_index_list, cluster_y].to_list()
    #     cluster_z_coord_list=nodes.loc[cluster_index_list, cluster_z].to_list()
    #
    #     cluster_x_coord_list=[x ** scale_by for x in cluster_x_coord_list]
    #     cluster_y_coord_list=[x ** scale_by for x in cluster_y_coord_list]
    #
    #     mesh_colour=['red', 'blue', 'green']
    #
    #     mesh_trace=dict(
    #         alphahull=10,
    #         name=i,
    #         opacity=0.1,
    #         type="mesh3d",
    #         color=mesh_colour[i],
    #         x=cluster_x_coord_list,
    #         y=cluster_y_coord_list,
    #         z=cluster_z_coord_list
    #     )
    #     items_to_trace.append(mesh_trace)
    #
    # ############################################################################################################################################################
    # # Edge Trace
    #
    # edge_x=[]
    # edge_y=[]
    # edge_z=[]
    #
    # for index, i in enumerate(reference_index):
    #     if i != main_article_value:
    #         x0 = x_coords[i]
    #         x1 = x_coords[main_article_value]
    #
    #         y0 = y_coords[i]
    #         y1 = y_coords[main_article_value]
    #
    #         z0 = nodes.loc[i, outlier_z]
    #         z1 = nodes.loc[main_article_value, outlier_z]
    #
    #         edge_x.append(x0 ** scale_by)
    #         edge_x.append(x1 ** scale_by)
    #         edge_x.append(None)
    #
    #         edge_y.append(y0 ** scale_by)
    #         edge_y.append(y1 ** scale_by)
    #         edge_y.append(None)
    #
    #         edge_z.append(z0)
    #         edge_z.append(z1)
    #         edge_z.append(None)
    #
    #         reference_edge_trace=go.Scatter3d(
    #             x=edge_x,
    #             y=edge_y,
    #             z=edge_z,
    #             line=dict(width=1,
    #                       # cmax=main_article_year,
    #                       # cmin=1950,
    #                       color='orange'),
    #                       # colorscale='blugrn'),
    #             hoverinfo='none',
    #             mode='lines')
    #
    #         items_to_trace.append(reference_edge_trace)
    #
    # edge_x=[]
    # edge_y=[]
    # edge_z=[]
    #
    # for index, i in enumerate(citation_index):
    #     if i != main_article_value:
    #
    #         x0 = x_coords[i]
    #         x1 = x_coords[main_article_value]
    #
    #         y0 = y_coords[i]
    #         y1 = y_coords[main_article_value]
    #
    #         z0 = nodes.loc[i, outlier_z]
    #         z1 = nodes.loc[main_article_value, outlier_z]
    #
    #         edge_x.append(x0 ** scale_by)
    #         edge_x.append(x1 ** scale_by)
    #         edge_x.append(None)
    #
    #         edge_y.append(y0 ** scale_by)
    #         edge_y.append(y1 ** scale_by)
    #         edge_y.append(None)
    #
    #         edge_z.append(z0)
    #         edge_z.append(z1)
    #         edge_z.append(None)
    #
    #         citation_edge_trace=go.Scatter3d(
    #             x=edge_x,
    #             y=edge_y,
    #             z=edge_z,
    #             line=dict(width=1,
    #                       # cmax=2022,
    #                       # cmin=main_article_year,
    #                       color='green'),
    #                       # colorscale="inferno"),
    #             hoverinfo='none',
    #             mode='lines')
    #
    #         items_to_trace.append(citation_edge_trace)

    # #||||||  Hover of Middle of Connection for Details  |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
    #
    #  middle_hover_trace=go.Scatter(x=[], y=[], hovertext=[], mode='markers', hoverinfo="text",
    #                                marker={'size': 20, 'color': 'LightSkyBlue'},
    #                                opacity=0)
    #  #
    #  # index=0
    #  # for edge in G.edges:
    #  #     x0, y0=G.nodes[edge[0]]['pos']
    #  #     x1, y1=G.nodes[edge[1]]['pos']
    #  #     hovertext="Citation of: "+str(G.edges[edge]['Source'])+"<br>"+ \
    #  #               "In Paper: "+str(G.edges[edge]['Target'])+"<br>"+ \
    #  #               "Influential: "+str(G.edges[edge]['TransactionAmt'])+"<br>"+ \
    #  #               "Publish Date: "+str(G.edges[edge]['Date'])
    #  #     middle_hover_trace['x']+=tuple([(x0+x1) / 2])
    #  #     middle_hover_trace['y']+=tuple([(y0+y1) / 2])
    #  #     middle_hover_trace['hovertext']+=tuple([hovertext])
    #  #     index=index+1

    # nodes = go.Scatter3d(x=outliers.x,
    #                                  y=outliers.y,
    #                                  z=data['Year'][abstract_df.index.to_list()],
    #                                  mode='markers',
    #                                  marker=dict(
    #                                      size=12,
    #                                      color=data['Year'][abstract_df.index.to_list()],
    #                                      # set color to an array/list of desired values
    #                                      colorscale='Viridis',  # choose a colorscale
    #                                      opacity=1
    #                                  ))
    #
    # edges = go.Scatter3d(x=outliers.x,
    #                                  y=outliers.y,
    #                                  z=data['Year'][abstract_df.index.to_list()],
    #                                  mode='markers',
    #                                  marker=dict(
    #                                      size=12,
    #                                      color=data['Year'][abstract_df.index.to_list()],
    #                                      # set color to an array/list of desired values
    #                                      colorscale='Viridis',  # choose a colorscale
    #                                      opacity=1
    #                                  ))


    #################################################################################################################################################################
    figure={
        "data": items_to_trace,
        "layout": go.Layout(title='Interactive Journal Network Visualization', showlegend=False, hovermode='closest',
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



def Visualise_on_Local_Host(
        nodes,
        node_title,
        outlier_x,
        outlier_y,
        outlier_z,
        cluster_x,
        cluster_y,
        cluster_z,
        cluster_labels,
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
                                                                    outlier_x,
                                                                    outlier_y,
                                                                    outlier_z,
                                                                    cluster_x,
                                                                    cluster_y,
                                                                    cluster_z,
                                                                    cluster_labels,
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

Visualise_on_Local_Host('node.csv',
                        'Title',
                        'x data',
                        'y data',
                        'Year',
                        'Clustered_x',
                        'Clustered_y',
                        'Year',
                        'labels',
                        )