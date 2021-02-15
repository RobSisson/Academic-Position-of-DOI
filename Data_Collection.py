### Data Collection & Management Packages ###
import requests
import pandas as pd
import numpy as np

### Time Management & Monitoring Packages ###
import time
from tqdm import tqdm

def api_call(url):
    # print(url)
    return requests.request("GET", url)

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

    # Export tidy pandas to CSV
    node.to_csv('node.csv')
    edge.to_csv('edge.csv')

    return node, edge

# Doi_to_Data('10.1093/CJE/BEY045')




