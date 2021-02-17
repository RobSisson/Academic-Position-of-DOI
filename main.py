doi = '10.1093/cje/bey045'

import Data_Collection

data = Data_Collection.Doi_to_Data(doi)

import Data_Population

populated = Data_Population.Populate_Journal_Info(data[0])

import Data_Processing

prepared = Data_Processing.Prepare_Data(populated, 'Abstract', 'Title')

embeddings = Data_Processing.Model_Topics(prepared[0], transformer= 'allenai/scibert_scivocab_uncased')

coords = Data_Processing.Plot_and_Cluster(embeddings)

result = Data_Processing.Merge_Mapping_Results(populated, coords, prepared[1])

result.to_csv('node.csv')


# Once above run below, can be done by running Visualiser.py
# import Visualiser
#
# Visualiser.Visualise_on_Local_Host('node.csv',
#                                    'Title',
#                                    'x data',
#                                    'y data',
#                                    'Year',
#                                    labels='labels',
#                                    )


