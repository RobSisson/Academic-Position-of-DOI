### Data Collection & Management Packages ###
import requests
import pandas as pd
import numpy as np

### Time Management & Monitoring Packages ###
from time import sleep
import datetime as dt
from datetime import timedelta
from tqdm import tqdm

### Internal Packages ###
import Search_Variables as M
import CrossRef_Functions as Cr
import Datacite_Functions as Dc
import DOAJ_Functions as Doaj


import requests
from bs4 import BeautifulSoup
import json

# def build_selection():
# if Main.recommend_pointed_build == 1:
#     build =
# else:
#     build =
#
# return build

def api_call(url):
    # print(url)
    return requests.request("GET", url)


def collect_data():
    result_df_created=0

    if M.databases["cr"] == 1:
        result=Cr.cross_ref_search(M.date)
        result_df_created+=1

    if M.databases["dc"] == 1:
        if result_df_created == 0:
            result=Dc.datacite_search(M.date)
            result_df_created+=1
        else:
            result=pd.concat([result, Dc.datacite_search(M.date)])

    if M.databases["doaj"] == 1:
        if result_df_created == 0:
            result=Doaj.doaj_search(M.date)
            result_df_created+=1
        else:
            result=pd.concat([result, Doaj.doaj_search(M.date)])

    if M.results_use["export_to_csv"] == 1:
        result.to_csv(convert_to_csv(M.date), sep='\t', encoding='utf-8', index=False)
        print('CSV Created')
    if M.results_use["print_results"] == 1:
        print(result)


def convert_to_csv(date):
    name="Journal_Search"
    if M.databases["cr"] == 1:
        Cr="Cr"
        name=".".join((name, Cr))

    if M.databases["dc"] == 1:
        Dc="Dc"
        name=".".join((name, Dc))

    if M.databases["doaj"] == 1:
        Doaj="Doaj"
        name=".".join((name, Doaj))

    name=" - ".join((name, date))

    import os
    # path=r"C:\Users\rob_s\PycharmProjects\systemo\Storage"
    output_file=os.path.join(M.directory, name+".csv")
    print('CSV Path Created')
    return output_file

def doi_to_abstract_semantic(doi):
    sem_url='https://api.semanticscholar.org/v1/paper/'+str(doi)
    response=api_call(sem_url)
    parsed=response.json()["references"]
    print(json.dumps(parsed, indent=4, sort_keys=True))
    result=pd.json_normalize(parsed)

    return result.to_csv('reftest.csv')


def json_extract(obj, key):
    """Recursively fetch values from nested JSON."""
    arr=[]

    def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    values=extract(obj, arr, key)
    return values


class ReturnValue(object):
    __slots__=["Abstract", "Fields", "Topics"]

    def __init__(self, Abstract, Fields, Topics):
        self.Abstract=Abstract
        self.Fields=Fields
        self.Topics=Topics


def doi_to_key_info(doi):
    sem_url='https://api.semanticscholar.org/v1/paper/'+str(doi)
    response=api_call(sem_url)
    parsed=response.json()

    return ReturnValue(parsed["abstract"], parsed["fieldsOfStudy"], parsed["topics"])


import time


def doi_semantic_search(doi):
    doi_semantic_search_time=time.time()

    sem_url='https://api.semanticscholar.org/v1/paper/'+str(doi)
    response=api_call(sem_url)
    parsed=response.json()



    total_paper_number=len(json_extract(parsed, 'paperId'))

    edge=pd.DataFrame(columns=['Influential',  # Overflow to prevent CSV-induced data corruption
                               'Source',
                               'Target',
                               'Year'],
                      index=range(0, total_paper_number))

    node=pd.DataFrame(columns=['Title',
                               'Abstract',
                               'Fields',  # Numpy Array with ['Field',...]
                               'Topics',  # Numpy Array with [['Topic', 'TopicID'],...]
                               'Authors',  # Numpy Array with [['Name', 'AuthorID'], CitationVelocity],...
                               'Venue',
                               'Paper_Id',
                               'DOI',
                               'Year',
                               'Ref_Number',
                               'Cite_Number',
                               'Influential_Count'],
                      index=range(0, total_paper_number))

    ######## Main Article

    node['Title'][0]=parsed["title"]
    node['Abstract'][0]=parsed["abstract"]
    node['Fields'][0]=parsed["fieldsOfStudy"]
    node['Topics'][0]=parsed["topics"]
    node['Authors'][0]=parsed["authors"]
    node['Venue'][0]=parsed["venue"]
    node['Paper_Id'][0]=parsed["paperId"]
    node['DOI'][0]=parsed["doi"]
    node['Year'][0]=parsed["year"]
    node['Influential_Count'][0]=parsed["influentialCitationCount"]

    ######## References

    references_json=parsed["references"]
    number_references=len(references_json)
    node['Ref_Number'][0]=str(number_references)

    for i, reference in tqdm(enumerate(references_json), desc='References'):

        # Populate node
        node['Title'][i+1]=json_extract(reference, 'title')[0]
        node['Paper_Id'][i+1]=json_extract(reference, 'paperId')[0]
        node['DOI'][i+1]=json_extract(reference, 'doi')[0]
        node['Year'][i+1]=''.join(str(e) for e in json_extract(reference, 'year'))

        # Populate edge
        edge['Target'][i+1]=parsed["paperId"]
        edge['Source'][i+1]=json_extract(reference, 'paperId')[0]
        edge['Year'][i+1]=''.join(str(e) for e in json_extract(reference, 'year'))
        edge['Influential'][i+1]=json_extract(reference, 'isInfluential')[0]

    ######## Citations

    citations_json=parsed["citations"]
    number_citations=len(citations_json)
    node['Cite_Number'][0]=str(number_citations)

    for i, citation in tqdm(enumerate(citations_json), desc='Citations'):
        # Populate node
        node['Title'][i+number_references+1]=json_extract(citation, 'title')[0]
        node['Paper_Id'][i+number_references+1]=json_extract(citation, 'paperId')[0]
        node['DOI'][i+number_references+1]=json_extract(citation, 'doi')[0]
        node['Year'][i+number_references+1]= ''.join(str(e) for e in json_extract(citation, 'year'))

        # Populate edge
        edge['Target'][i+number_references+1]=json_extract(citation, 'paperId')[0]
        edge['Source'][i+number_references+1]=parsed["paperId"]
        edge['Year'][i+number_references+1]=''.join(str(e) for e in json_extract(citation, 'year'))
        edge['Influential'][i+number_references+1]=json_extract(citation, 'isInfluential')[0]

    print(edge)

    #### Clean Extra Values (Overestimated in creation of df) ####
    # Mark empty node rows
    fill_empty=node.Paper_Id.fillna('')

    fill_empty_edge=edge.Source.fillna('')

    # Create list of empty node Rows
    where_empty=np.where(fill_empty.to_numpy() == '')
    list_empty_values=where_empty[0]

    where_empty_edge=np.where(fill_empty_edge.to_numpy() == '')
    list_empty_values_edge=where_empty_edge[0]
    # Drop empty node Rows
    node=node.drop(list_empty_values)

    edge=edge.drop(list_empty_values_edge)

    edge.to_csv('edge.csv')

    #### Identify Node Rows for Populating ####
    # Mark Empty Abstracts
    fillna=node.Abstract.fillna('')
    cleaning=fillna.to_numpy()

    np_where=np.where(cleaning == '')

    list_empty_values=np_where[0]

    time_remaining=(len(list_empty_values) / 100) * 5.15

    print('Due to API limitations (100 calls/5 minutes), populating '+
          str(len(list_empty_values))+
          ' values will take approximately '+
          str(round(time_remaining))+
          ' minutes')
    print('Every 100 calls, the script will export progress, '
          'and then sleep for the remainder for the 5 minute window, '
          'then resume')

    sleep(1) # to prevent TQDM visual error

    input_fails = 0
    collection_fails = 0

    call_limit=100
    time_limit=400

    node.to_csv('temp_node.csv')

    for i in tqdm(list_empty_values, desc='Populating Documents'):

        if i == call_limit:
            time_difference=int((time.time()-doi_semantic_search_time))
            if time_difference<time_limit:
                node.to_csv('temp_node.csv')
                for s in range(0, time_difference):
                    sleep(1)
                    if s == time_difference:
                        print('Slept for '+str(s)+' seconds')

            call_limit+=100
            time_limit+=400

        try:
            info=doi_to_key_info(node.Paper_Id.iat[list_empty_values[i]])

            try:
                node.Abstract.iat[list_empty_values[i]]=info.Abstract
                node.Fields.iat[list_empty_values[i]]=info.Fields
                node.Topics.iat[list_empty_values[i]]=info.Topics

            except:
                input_fails+=1
                print(str(node.Paper_Id.iat[list_empty_values[i]])+' Input Failed')

        except:
            collection_fails+=1
            print(str(node.Paper_Id.iat[list_empty_values[i]])+' Collection Failed')

    print(str(input_fails)+' Number of Failed Inputs')
    print(str(collection_fails)+' Number of Failed Inputs')

    return node, edge

response = doi_semantic_search('10.1038/nrn3241')

response[0].to_csv('node.csv')
