### Data Collection ###
import requests
# import json
import asyncio
from aiohttp import ClientSession, ClientConnectorError

### Data Management Packages ###
import pandas as pd

### Time Management & Monitoring Packages ###
import time

# from tqdm import tqdm # functionality still to add

### Used in tests ###
# import datetime as dt
# from datetime import timedelta
# from bs4 import BeautifulSoup


def api_call(url):
    # print(url)
    return requests.request("GET", url)


def Compile_Endpoint_List(
        endpoints,
        apis_to_call,

):
    endpoint=[]

    for i, e in enumerate(endpoints):

        apis_to_call_temp=[]
        for index, api in enumerate(apis_to_call[i]):
            apis_to_call_temp.append([0, api])

        array=[e, apis_to_call_temp, None, i]
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

                url_list.append(
                    [Create_URL(api_list[api_number], endpoint[0]), endpoint[3]])  # append created url to list

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

        if result[0] == None:
            result[0] = 'No Result Found'

    except ClientConnectorError:
        return [id, url, 404]
    except:
        return [id, url, 'No Result Found']

    return [id, url, result[0]]


async def make_requests(urls: list, **kwargs):
    async with ClientSession() as session:
        tasks=[]
        for i, url in enumerate(urls):
            tasks.append(
                fetch_html(url=url[0], session=session, id=url[1], **kwargs)
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

    while_loop=0

    while len(endpoint_list)>(len(result)):
        while_loop+=1

        url_list=await Create_Next_URLs(api_list, endpoint_list)

        response=await asyncio.gather(make_requests(urls=url_list))

        for i, item in enumerate(response[0]):

            response_id=item[0]

            filtered_result=item[2]

            if filtered_result != 'No Result Found':
                # print('Result found, appending...')
                result.append(item)
                # print('Preventing future api calls for this endpoint...')
                for i, api_to_call in enumerate(endpoint_list[response_id][1]):
                    if api_to_call[0] == 0:
                        api_to_call[0]=2  # Changing all remaining api's to call to value 2
            else:
                # print('No result found for this endpoint, when calling this API')

                for i, api_to_call in enumerate(endpoint_list[response_id][1]):
                    if api_to_call[0] == 0:
                        # print('Preventing this api from being called again')
                        api_to_call[0]=1  # Changing previously called api value to 1
                        break

                if while_loop == len(endpoint_list[response_id][1]):
                    # print('Appending to empty result...')
                    result.append(item)

    reordered=[[y, z] for x, y, z in sorted(result)]

    return reordered


def Populate_Journal_Info(
        documents
    ):

    if isinstance(documents, str) and documents.index('csv'):
        print('Loading CSV as DataFrame...')
        node=pd.read_csv(documents)
    else:
        node = documents

    #### Identify Node Rows for Populating ####
    # Mark Empty Abstracts
    values={'Abstract': 'Empty', 'DOI': 'Empty'}
    node = node.fillna(value=values)

    index_list = node.loc[node['Abstract'] == 'Empty'].index.tolist()
    doi_list = node.loc[node['Abstract'] == 'Empty', 'DOI'].tolist()
    id_list = node.loc[node['Abstract'] == 'Empty', 'Paper_Id'].tolist()

    endpoint_list = []
    for i, doi in enumerate(doi_list):
        if doi != 'Empty':
            endpoint_list.append(doi_list[i])
        else: # uses Paper_Id instead, though this can only be used in semantic scholar, not other DOI searches
            endpoint_list.append(id_list[i])

    abstracts = asyncio.run(DOI_List_to_Result(endpoint_list))


    from bs4 import BeautifulSoup

    for i, abstract in enumerate(abstracts):
        print(abstract)
        if '<' in abstract[1]:
            soup = BeautifulSoup(abstract[1], 'html.parser').get_text()
            node.Abstract.iat[index_list[i]] = soup
        elif abstract[1] == 'No Result Found':
            node.Abstract.iat[index_list[i]]='Empty'
        else:
            node.Abstract.iat[index_list[i]]=abstract[1]

    node.to_csv('node.csv')

    return node

# print(Populate_Journal_Info('node.csv'))
