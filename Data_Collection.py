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


# def json_extract(obj, key):
#     """Recursively fetch values from nested JSON."""
#     arr=[]
#
#     def extract(obj, arr, key):
#         """Recursively search for values of key in JSON tree."""
#         if isinstance(obj, dict):
#             for k, v in obj.items():
#                 if isinstance(v, (dict, list)):
#                     extract(v, arr, key)
#                 elif k == key:
#                     arr.append(v)
#         elif isinstance(obj, list):
#             for item in obj:
#                 extract(item, arr, key)
#         return arr
#
#     values=extract(obj, arr, key)
#     return values


class ReturnValue(object):
    __slots__=["Abstract", "Fields", "Topics"]

    def __init__(self, Abstract, Fields, Topics):
        self.Abstract=Abstract
        self.Fields=Fields
        self.Topics=Topics


import asyncio
from aiohttp import ClientSession, ClientConnectorError
import time


def Compile_Endpoint_List(
        endpoints,
        apis_to_call,

):
    endpoint=[]

    for i, e in enumerate(endpoints):

        apis_to_call_temp=[]
        for i, api in enumerate(apis_to_call[i]):
            apis_to_call_temp.append([0, api])

        array=[e, apis_to_call_temp, None]
        endpoint.append(array)

    return endpoint


def Compile_Kwargs_list(
        keys,
        values,
        positions
):
    kwargs=[]

    if type(keys) != list:
        keys=[keys]

    if type(values) != list:
        values=[values]

    if type(positions) != list:
        positions=[positions]

    for i, item in enumerate(keys):
        array=[[keys[i], values[i]], positions[i]]
        kwargs.append(array)

    return kwargs


def Kwargs_list_dict(
        kwargs
):
    result={}
    for i, item in enumerate(kwargs):
        result_id=item[1]

        for i, kwarg in enumerate(item):
            result[result_id]=(str(item[0][i-1])+str(item[0][i]))

    return result


def Kwargs_to_dict(
        keys,
        values,
        positions
):
    return Kwargs_list_dict(Compile_Kwargs_list(keys, values, positions))


def Compile_Api_List(  # Create usable API list out of input info
        apis,  # base api (always first in url generation)
        filters,  # the api response will be filtered for these keys, add in format  [[apikey1, apikey2],api2key1,...]
        limits,  # add in format [api 1 limit, api 2 limit,... api n limit]
        periods,  # add in format [api 1 period, api 2 period,... api n period]
        keys=None,
        # add in format [[api 1 key 1, ... api 1 key n], ... [api n key 1, ... api 1 key n]] add None for no keys
        values=None,  # same format as keys
        positions=None  # same format as keys
):
    api=[]
    kwargs=[]

    if None not in [keys, values, positions] and (len(keys) == len(values) == len(
            positions)):  # convert raw lists to the format [{position: key+value, ...}, {position: key+value, ...}, ... ]
        for i, item in enumerate(keys):
            # print(item)
            if item != None:
                kwargs.append(Kwargs_to_dict(keys[i], values[i], positions[i]))
            else:
                kwargs.append(None)

    for i, end in enumerate(apis):
        array=[apis[i], filters[i], limits[i], periods[i], kwargs[i], 0, 0]
        api.append(array)

    return api


def Create_URL(
        api,
        endpoint,
        # **kwargs,
):
    url=[api[0]]

    if api[4] != None:  # api[4] is the column containing additional {positions:key+value,} including auth keys
        kwargs=api[4]

        for i in range(len(
                kwargs)+1):  # for number of additional values + 1 (since the total string will include the key+values and the endpoint
            try:
                url.append(kwargs[i])  # append key+value that holds position i

            except:
                url.append(endpoint)  # if not present, append endpoint

        url=''.join(url)  # list -> string
    else:
        url=str(api[0])+str(endpoint)

    return url


async def check_api(api_list, api_number) -> bool:
    api_to_check=api_list[api_number]

    limit=api_to_check[2]
    period=api_to_check[3]

    # count = api_to_check[5]
    # timer = api_to_check[6]

    if api_to_check[6] == 0:  # check is timer has been started, if not, start it
        api_to_check[6]=time.time()  # start timer
        api_to_check[5]+=1  # add one to count
        return True

    else:
        elapsed=time.time()-api_to_check[6]  # calculate elapsed time
        if (elapsed<period) & (api_to_check[5]<limit-1):  # if timer is below period and count is below limit-1
            api_to_check[5]+=1  # add one to count
            return True

        elif (elapsed<period) & (api_to_check[5]>=limit-1):  # if timer is below period and count is above limit-1
            return False

        elif elapsed>=period:  # if timer is beyond period for that api, reset the timer and the count, then return true
            api_to_check[6]=time.time()  # start timer
            api_to_check[5]=0  # reset count
            return True


async def Create_Next_URLs(
        api_list,
        endpoint_list,
):
    url_list=[]
    print(endpoint_list)
    for i, endpoint in enumerate(endpoint_list):
        for i, api_to_call in enumerate(endpoint[1]):

            check=await check_api(api_list, api_to_call[1])

            if (api_to_call[0]) == 0 and check == True:
                api_number=api_to_call[1]  # this would be the number of the api to call, not the api itself

                url_list.append(Create_URL(api_list[api_number], endpoint[0]))  # append created url to list

                # endpoint[1].remove(api_to_call) # once url is appended, remove the relevant api_to_call from the endpoint

                # api_to_call[1] = 1 # alt version of checking if an endpoint has been called
                break  #

            # if len(endpoint[1]) == 0:
            #     print("All Api's for "+str(endpoint[0]+' searched, and no result found'))
            #     endpoint_list.remove(endpoint)

    return url_list


async def json_extract(obj, key):
    """Recursively fetch values from nested JSON."""
    arr=[]

    async def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    await extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                await extract(item, arr, key)
        return arr

    values=await extract(obj, arr, key)
    return values


async def fetch_html(url: str, session: ClientSession, id, **kwargs) -> list:
    try:
        resp=await session.request(method="GET", url=url, **kwargs)
        resp_json=await resp.json()

        result=await json_extract(resp_json, 'abstract')

        test_abstract_presence=result[
            0]  # will cause an exception if no search key is found and fail out to the 2nd except statement

    except ClientConnectorError:
        return [id, url, 404]
    except:
        return [id, url, None]

    return [id, url, result[0]]


async def make_requests(urls: list, **kwargs):
    async with ClientSession() as session:
        tasks=[]
        for i, url in enumerate(urls):
            tasks.append(
                fetch_html(url=url, session=session, id=i, **kwargs)
            )
        responses=await asyncio.gather(*tasks)

    return responses


if __name__ == "__main__":
    import pathlib
    import sys

    if sys.version_info[0] == 3 and sys.version_info[1]>=8 and sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    assert sys.version_info>=(3, 7), "Script requires Python 3.7+."
    here=pathlib.Path(__file__).parent


async def DOI_List_to_Result(
        endpoint_list,
        apis_to_call=None,
        # add info in format: [[0, 1], [2], [0, 1, 2]...] Using the value of the API, leave blank for all
):
    apis=['https://api.semanticscholar.org/v1/paper/',
          'https://api.crossref.org/v1/works/',
          # 'https://doaj.org/api/v2/search/articles/doi%3A'
          ]
    filters=[['abstract'],
             ['abstract'],
             # ['abstract']
             ]
    limits=[100, 50]
    period=[300, 100]
    keys=[None, None]
    values=[None, None]
    positions=[None, None]

    api_list=Compile_Api_List(apis=apis,
                              filters=filters,
                              limits=limits,
                              periods=period,
                              keys=keys,
                              values=values,
                              positions=positions)

    if apis_to_call == None:
        apis_to_call=[]
        api_values=[]

        for i, api in enumerate(apis):
            api_values.append(i)

        for i, item in enumerate(endpoint_list):
            apis_to_call.append(api_values)

    endpoint_list=Compile_Endpoint_List(endpoints=endpoint_list,
                                        apis_to_call=apis_to_call)

    result=[]

    loop=0

    while len(endpoint_list)>len(result):
        loop+=1
        print(len(endpoint_list)-len(result))
        url_list=await Create_Next_URLs(api_list, endpoint_list)

        response=await asyncio.gather(make_requests(urls=url_list))

        for i, item in enumerate(response[0]):
            print('below')
            print(item)

            print(item[0])

            if item[2] not in [[], None, '']:
                print('gogogo')
                result.append(item)
                for i, api_to_call in enumerate(endpoint_list[item[0]][1]):
                    api_to_call[0]=1
            else:
                for i, api_to_call in enumerate(endpoint_list[item[0]][1]):
                    if i<loop:
                        api_to_call[0]=1
                    else:
                        break

    return result







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

    abstracts = asyncio.run(DOI_List_to_Result(doi_list))

    for i in list_empty_values:
        node.Abstract.iat[list_empty_values[i]]=abstracts[i]

    node.to_csv('node.csv') # final node

    return node, edge



