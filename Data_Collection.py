### Data Collection & Management Packages ###
import requests
import pandas as pd
import numpy as np

### Time Management & Monitoring Packages ###
import time
from time import sleep
import datetime as dt
from datetime import timedelta
from tqdm import tqdm


import requests
import json
from bs4 import BeautifulSoup


def api_call(url):
    # print(url)
    return requests.request("GET", url)


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

from concurrent.futures import as_completed
from requests_futures.sessions import FuturesSession
from bs4 import BeautifulSoup

def Concurrent_Abstract_Request(
        suffix,
):
    suffix_2 = []

    session=FuturesSession()

    futures=[]

    result ={}

    for index, i in enumerate(suffix):
        print(i)
        future=session.get('https://api.crossref.org/v1/works/' + str(i))
        future.i=i
        futures.append(future)

    for future in as_completed(futures):
        print(future)
        resp=future.result()
        parsed =  resp.json()
        if json_extract(parsed, 'abstract') in [[None], [], None]:
            print('test')
            suffix_2.append(future.i)
        else:
            print('else')
            for i in json_extract(parsed, 'abstract'):
                result[future.i] = (BeautifulSoup(i, features="html.parser").get_text())

    futures_2 = []
    for index, i in enumerate(suffix_2):
        print('yoyoyo')
        print(i)
        future=session.get('https://doaj.org/api/v2/search/articles/doi%3A' + str(i))
        future.i=i
        futures.append(futures_2)

    for future in as_completed(futures_2):
        resp=future.result()
        try:
            parsed =  resp.json()
        except:
            print('break')
            break
        if json_extract(parsed, 'abstract') in [[None], [], None]:
            print('Not found')
        else:
            print('found')
            for i in json_extract(parsed, 'abstract'):
                result[future.i]=(BeautifulSoup(i, features="html.parser").get_text())
    print(suffix_2)
    return result



import asyncio
import aiohttp
from aiohttp import ClientSession, ClientConnectorError

async def fetch_html(url: str, session: ClientSession, **kwargs) -> tuple:
    try:
        resp = await session.request(method="GET", url=url, **kwargs)
    except ClientConnectorError:
        return (url, 404)
    return (url, resp.text)

async def make_requests(urls: set, **kwargs) -> None:
    async with ClientSession() as session:
        tasks = []
        for url in urls:
            tasks.append(
                fetch_html(url=url, session=session, **kwargs)
            )
        results = await asyncio.gather(*tasks)

    for result in results:
        print(f'{result[1]} - {str(result[0])}')

if __name__ == "__main__":
    import pathlib
    import sys

    if sys.version_info[0] == 3 and sys.version_info[1]>=8 and sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    assert sys.version_info >= (3, 7), "Script requires Python 3.7+."
    here = pathlib.Path(__file__).parent







def doi_to_key_info(doi):
    sem_url='https://api.semanticscholar.org/v1/paper/'+str(doi)
    response=api_call(sem_url)
    parsed=response.json()

    return ReturnValue(parsed["abstract"], parsed["fieldsOfStudy"], parsed["topics"])



def Doi_to_Data(doi):
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
    fillna = node.Abstract.fillna('')
    cleaning = fillna.to_numpy()

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

    node.to_csv('node.csv') # pre abstract collection backup

    for i in tqdm(list_empty_values, desc='Populating Documents'):

        if i == call_limit:
            time_difference=int((time.time()-doi_semantic_search_time))
            if time_difference<time_limit:
                node.to_csv('node.csv') # backup
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

    ################

    # Mark Empty Abstracts
    fillna=node.Abstract.fillna('')
    cleaning=fillna.to_numpy()

    np_where=np.where(cleaning != '')
    print(np_where)

    list_empty_2_values=np_where[0]

    print(list_empty_2_values)


    doi_list= []
    for i, index in enumerate(list_empty_2_values):
        doi_list.append(node.DOI.iat[index])
    print(doi_list)

    node.to_csv('node.csv')  # pre abstract round 2 collection backup

    abstracts = Concurrent_Abstract_Request(doi_list)

    for i in list_empty_values:
        node.Abstract.iat[list_empty_values[i]]=abstracts[i]

    node.to_csv('node.csv') # final node

    return node, edge



