import Data_Collection, Data_Population, Data_Processing, Visualiser

import pandas as pd

doi = '10.1093/cje/bey045'

data = Data_Collection.Doi_to_Data(doi)

populated = Data_Population.Populate_Journal_Info(data[0])

prepared = Data_Processing.Prepare_Data(populated, 'Abstract', 'Title')

embeddings = Data_Processing.Model_Topics(prepared[0], transformer= 'allenai/scibert_scivocab_uncased')

coords = Data_Processing.Plot_and_Cluster(embeddings)

result = Data_Processing.Merge_Mapping_Results(populated, coords, prepared[1])

result.to_csv('node.csv')

# Visualiser.Visualise_on_Local_Host(result,
#                                    'Title',
#                                    'x data',
#                                    'y data',
#                                    'Year',
#                                    'Clustered_x',
#                                    'Clustered_y',
#                                    'Year',
#                                    'labels',
#                                    )
