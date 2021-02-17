# The Academic Position of a DOI
A data pipeline which collects the references and citations of a specified DOI, undertakes topic modelling using SciBERT, then plots the results on a 3D graph distributed by publish date.

# Breakdown

1. Call Semantic Scholar API using user selected DOI, returning response in a dataframe.
2. Asynchronously call journal article APIs to populate the abstracts of the references and citations identified in the first call. 
   This uses 2 or 3 APIs, called in sequence if the previous API fails to populate the field, while managing unique API limits/rates.
   
3. Process the populated abstracts using Top2Vec and Sci-BERT.
4. Transform embeddings using UMap.
5. Create clusters based spatial distance algorithms using HDBScan.
6. Plot the clusters and outliers for both references and citations in 3D, with the x and y values as UMap data/HDBScan clusters, and the publication date as the z value using Plotly.
7. Present the 3D graph using dash.

# Still To do
 - Improve colour scheme of visualiser, making outliers/clusters and refs/cites more clear
 - Add additional information to hover/click
 - Add additional cluster details
 - Add additional metrics for clustering, and in vis adjustment
 - Make UMap factors more easily adjustable (ideally within the visualiser)

# Future Developments
 - Implement wider article acquisition, including the references and citations of other articles to build up a more holistic view of the immediate topic; especially aiming to highlight cross-referencing
 - Integrate metaknowledge.py if possible