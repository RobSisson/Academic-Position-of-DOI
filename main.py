import Data_Collection, Data_Processing, Visualiser

import pandas as pd

doi = '10.1093/cje/bey045'

data = Data_Collection.Doi_to_Data(doi)

processed_data = Data_Processing.Model_2D_Topic(data[0],
                                                'Abstract',
                                                transformer= 'allenai/scibert_scivocab_uncased'
                                                ) #.to_csv('Processed Node Data - '+ str(doi)+'.csv')

Visualiser.Visualise_on_Local_Host(processed_data,
                                   'Title',
                                   'Outliers_x',
                                   'Outliers_y',
                                   'Year',
                                   'Clustered_x',
                                   'Clustered_y',
                                   'Year',
                                   'Clustered_labels',
                                   )
