import pandas as pd
from top2vec import Top2Vec
import seaborn as sns
import matplotlib.pyplot as plt
import umap
import timeit
import numpy as np
from tqdm import tqdm
import time

import sklearn

from transformers import *

from sentence_transformers import SentenceTransformer

import tensorflow

def DataFrame_to_Documents(df, content_column):

    abstract_df = df[content_column].dropna()

    documents = abstract_df.tolist()

    return documents, abstract_df


def Topic_Model_2D(documents,
                   content_column,
                   embedding_progress_bar = True,
                   transformer = None,
                   umap_n_neighbors = None,
                   umap_n_components = None,
                   umap_min_dist = None,
                   umap_metric = None,
                   hbdscan_min_cluster_size = None,
                   hbdscan_metric = None,
                   hbdscane_cluster_selection_method = None,
                   ):
    csv_converted=0

    if isinstance(documents, str) and documents.index('csv'):
        print('Loading CSV as DataFrame...')
        documents_df=pd.read_csv(documents)

        print('Confirming if loaded correctly...')
        print(documents_df[content_column][0:3])

        csv_converted=1

    if isinstance(documents, pd.DataFrame) or isinstance(documents_df, pd.DataFrame):
        print('Converting Dataframe to list...')
        if csv_converted == 1:
            return_doc=DataFrame_to_Documents(documents_df, content_column)
        else:
            return_doc=DataFrame_to_Documents(documents, content_column)

        documents=return_doc[0]

        print('Confirming if loaded correctly...')
        print(documents[0:3])
    print(type(documents))

    print('Loading Model...')

    model = SentenceTransformer(transformer)

    print('|||||| Setup Complete ||||||')
    print(' |||||  \  \    /  /  ||||| ')
    print('  ||||   \  \  /  /   ||||  ')
    print('   |||    \  \/  /    |||   ')
    print('    ||     \    /     ||    ')
    print('     |      \  /      |     ')
    print('             \/             ')

    number_of_documents = len(documents)

    print('Encoding ' + str(number_of_documents) + ' documents using ' + str(transformer) + ':')

    embeddings = model.encode(documents, show_progress_bar=embedding_progress_bar)

    print('||||  Encoding Complete ||||')
    print(' ||||   \  \    /  /   |||| ')
    print('  ||||   \  \  /  /   ||||  ')
    print('   |||    \  \/  /    |||   ')
    print('    ||     \    /     ||    ')
    print('     |      \  /      |     ')
    print('             \/             ')

    print('Umap Embeddings --> Hdbscan Clustering --> Data Preparation')

    import umap

    print(embeddings)

    umap_embeddings=umap.UMAP(n_neighbors=5,
                              min_dist=0.3,
                              metric='correlation').fit_transform(embeddings.data)
    print(umap_embeddings)
    print('Umap Embeddings Successful')
    import hdbscan
    cluster=hdbscan.HDBSCAN(min_cluster_size=15,
                            metric='euclidean',
                            cluster_selection_method='eom').fit(umap_embeddings)

    print(cluster)
    print('Hdbscan Clustering Successful')

    import matplotlib.pyplot as plt

    # Prepare data
    umap_data=umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    result=pd.DataFrame(umap_data, columns=['x', 'y'])
    result['labels']=cluster.labels_
    print('Data Preparation Successful')

    print('Exporting Data')



    print('||| Data Export Complete |||')
    print(' |||    \  \    /  /    ||| ')
    print('  ||||   \  \  /  /   ||||  ')
    print('   |||    \  \/  /    |||   ')
    print('    ||     \    /     ||    ')
    print('     |      \  /      |     ')
    print('             \/             ')

    outliers=result.loc[result.labels == -1, :]
    clustered=result.loc[result.labels != -1, :]

    outliers_x = outliers.x.to_frame(name='Outliers_x')
    outliers_y = outliers.y.to_frame(name='Outliers_y')
    clustered_x = clustered.x.to_frame(name='Clustered_x')
    clustered_y = clustered.y.to_frame(name='Clustered_y')
    clustered_labels = clustered.labels.to_frame(name='Clustered_labels')

    result = pd.concat([documents_df, outliers_x, outliers_y, clustered_x, clustered_y, clustered_labels], axis=1)

    return result




# Topic_Model_2D("Data/node.csv", 'Abstract', transformer= 'allenai/scibert_scivocab_uncased').to_csv('test.csv')

def Rescale_Column(
        documents,
        column_to_rescale,
        scale,
):
    if isinstance(documents, str) and documents.index('csv'):
        print('Loading CSV as DataFrame...')
        data=pd.read_csv(documents)

        print('Confirming if loaded correctly...')
        print(data[column_to_rescale][0:3])

        csv_converted=1

    max = data[column_to_rescale].max()
    min = data[column_to_rescale].min()

    difference =  max - min

    print(difference)

print(Rescale_Column())

#
# data = pd.read_csv("Data/node.csv")
# abstract_df = data["Abstract"].dropna()
# documents = abstract_df.tolist()
# print(len(documents))


# bert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
# tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')


# print('Test')







# https://www.kdnuggets.com/2020/11/topic-modeling-bert.html
#
# docs_df = pd.DataFrame(documents, columns=["Doc"])
# docs_df['Topic'] = cluster.labels_
# docs_df['Doc_ID'] = range(len(docs_df))
# docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})
#
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
#
#
# def c_tf_idf(documents, m, ngram_range=(1, 1)):
#     count=CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
#     t=count.transform(documents).toarray()
#     w=t.sum(axis=1)
#     tf=np.divide(t.T, w)
#     sum_t=t.sum(axis=0)
#     idf=np.log(np.divide(m, sum_t)).reshape(-1, 1)
#     tf_idf=np.multiply(tf, idf)
#
#     return tf_idf, count
#
#
# tf_idf, count=c_tf_idf(docs_per_topic.Doc.values, m=len(data))
#
# def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
#     words = count.get_feature_names()
#     labels = list(docs_per_topic.Topic)
#     tf_idf_transposed = tf_idf.T
#     indices = tf_idf_transposed.argsort()[:, -n:]
#     top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
#     return top_n_words
#
# def extract_topic_sizes(df):
#     topic_sizes = (df.groupby(['Topic'])
#                      .Doc
#                      .count()
#                      .reset_index()
#                      .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
#                      .sort_values("Size", ascending=False))
#     return topic_sizes
#
# top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
# topic_sizes = extract_topic_sizes(docs_df); topic_sizes.head(10)
#
# for i in range(20):
#     # Calculate cosine similarity
#     similarities = sklearn.metrics.pairwise.cosine_similarity(tf_idf.T)
#     np.fill_diagonal(similarities, 0)
#
#     # Extract label to merge into and from where
#     topic_sizes = docs_df.groupby(['Topic']).count().sort_values("Doc", ascending=False).reset_index()
#     topic_to_merge = topic_sizes.iloc[-1].Topic
#     topic_to_merge_into = np.argmax(similarities[topic_to_merge + 1]) - 1
#
#     # Adjust topics
#     docs_df.loc[docs_df.Topic == topic_to_merge, "Topic"] = topic_to_merge_into
#     old_topics = docs_df.sort_values("Topic").Topic.unique()
#     map_topics = {old_topic: index - 1 for index, old_topic in enumerate(old_topics)}
#     docs_df.Topic = docs_df.Topic.map(map_topics)
#     docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})
#
#     # Calculate new topic words
#     m = len(documents)
#     tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m)
#     top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
#
# topic_sizes = extract_topic_sizes(docs_df); topic_sizes.head(10)

#
# type(embeddings)
# type(model)
#
# # start = time.perf_counter()
# #
# # model = Top2Vec(documents,speed="deep-learn", workers=4)
# #
# # stop = time.perf_counter()
# #
# # print(f"Runtime {start - stop:0.4f} seconds")
#
# try:
#     embeddings.save("Models/top2vec_d2v")
# except:
#     a = 1
#
#
# # model = Top2Vec.load("Models/top2vec_d2v")
#
# try:
#     number_of_topics = embeddings.get_num_topics()
#
#     print("Number of Topics: " + str(model.get_num_topics()))
#
#
# except:
#     a = 1
#
# try:
#     topic_sizes, topic_nums = model.get_topic_sizes()
#
#     print('Topic Sizes')
#     print(topic_sizes)
#
#     print('Topic Nums')
#     print(topic_nums)
#
#
# except:
#     a = 1
# try:
#     topic_words, word_scores, topic_nums = model.get_topics(number_of_topics)
#
#     print('Topic Words')
#     print(topic_words)
#
#     print('Word scores')
#     print(word_scores)
#
#     print('Topic Nums')
#     print(topic_nums)
#
#
# except:
#     a = 1
#
# try:
#     topic_words, word_scores, topic_scores, topic_nums = model.search_topics(keywords=["neural"],
#                                                                              num_topics=number_of_topics)
#
#     print('Topic Words')
#     print(topic_words)
#
#     print('Word scores')
#     print(word_scores)
#
#     print('Topic scores')
#     print(topic_scores)
#
#     print('Topic Nums')
#     print(topic_nums)
#
# except:
#     a = 1
#
# try:
#     documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=1, num_docs=5)
#
#     # print('Documents')
#     # print(documents)
#     #
#     # print('Document scores')
#     # print(document_scores)
#     #
#     # print('Document IDs')
#     # print(document_ids)
#
#     for doc, score, doc_id in zip(documents, document_scores, document_ids):
#         print(f"Document: {doc_id}, Score: {score}")
#         print("-----------")
#         print(doc)
#         print("-----------")
#         print()
# except:
#     a = 1
#
# try:
#     documents, document_scores, document_ids = model.search_documents_by_keywords(keywords=["neural", "analysis"], num_docs=5)
#     for doc, score, doc_id in zip(documents, document_scores, document_ids):
#         print(f"Document: {doc_id}, Score: {score}")
#         print("-----------")
#         print(doc)
#         print("-----------")
#         print()
# except:
#     a = 1
#
# try:
#     words, word_scores = model.similar_words(keywords=["cortical"], keywords_neg=[], num_words=20)
#     for word, score in zip(words, word_scores):
#         print(f"{word} {score}")
# except:
#     a+=1
# print(a)




# import umap
#
# print(embeddings)
#
# umap_embeddings = umap.UMAP(n_neighbors=5,
#                             min_dist=0.3,
#                             metric='correlation').fit_transform(embeddings.data)
# print(umap_embeddings)
# print('Test')
# import hdbscan
# cluster = hdbscan.HDBSCAN(min_cluster_size=15,
#                           metric='euclidean',
#                           cluster_selection_method='eom').fit(umap_embeddings)
#
# print(cluster)
# print('Test')
# import matplotlib.pyplot as plt
#
# # Prepare data
# umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
# result = pd.DataFrame(umap_data, columns=['x', 'y'])
# result['labels'] = cluster.labels_
#
# # Visualize clusters
# fig, ax = plt.subplots(figsize=(20, 10))
# outliers = result.loc[result.labels == -1, :]
# clustered = result.loc[result.labels != -1, :]
# plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
# plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
# plt.colorbar()
#
# # https://www.kdnuggets.com/2020/11/topic-modeling-bert.html
#
# docs_df = pd.DataFrame(documents, columns=["Doc"])
# docs_df['Topic'] = cluster.labels_
# docs_df['Doc_ID'] = range(len(docs_df))
# docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})
#
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
#
#
# def c_tf_idf(documents, m, ngram_range=(1, 1)):
#     count=CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
#     t=count.transform(documents).toarray()
#     w=t.sum(axis=1)
#     tf=np.divide(t.T, w)
#     sum_t=t.sum(axis=0)
#     idf=np.log(np.divide(m, sum_t)).reshape(-1, 1)
#     tf_idf=np.multiply(tf, idf)
#
#     return tf_idf, count
#
#
# tf_idf, count=c_tf_idf(docs_per_topic.Doc.values, m=len(data))
#
# def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
#     words = count.get_feature_names()
#     labels = list(docs_per_topic.Topic)
#     tf_idf_transposed = tf_idf.T
#     indices = tf_idf_transposed.argsort()[:, -n:]
#     top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
#     return top_n_words
#
# def extract_topic_sizes(df):
#     topic_sizes = (df.groupby(['Topic'])
#                      .Doc
#                      .count()
#                      .reset_index()
#                      .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
#                      .sort_values("Size", ascending=False))
#     return topic_sizes
#
# top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
# topic_sizes = extract_topic_sizes(docs_df); topic_sizes.head(10)
#
# for i in range(20):
#     # Calculate cosine similarity
#     similarities = sklearn.metrics.pairwise.cosine_similarity(tf_idf.T)
#     np.fill_diagonal(similarities, 0)
#
#     # Extract label to merge into and from where
#     topic_sizes = docs_df.groupby(['Topic']).count().sort_values("Doc", ascending=False).reset_index()
#     topic_to_merge = topic_sizes.iloc[-1].Topic
#     topic_to_merge_into = np.argmax(similarities[topic_to_merge + 1]) - 1
#
#     # Adjust topics
#     docs_df.loc[docs_df.Topic == topic_to_merge, "Topic"] = topic_to_merge_into
#     old_topics = docs_df.sort_values("Topic").Topic.unique()
#     map_topics = {old_topic: index - 1 for index, old_topic in enumerate(old_topics)}
#     docs_df.Topic = docs_df.Topic.map(map_topics)
#     docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})
#
#     # Calculate new topic words
#     m = len(documents)
#     tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m)
#     top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
#
# topic_sizes = extract_topic_sizes(docs_df); topic_sizes.head(10)
#
#
#
#
#
# def scatter_plot(data, num_topics, clusterer, name, noise, model):
#     cluster_labels = clusterer.labels_
#     doc_labels = model.doc_top
#
#     if noise == True:
#         color_palette = sns.color_palette('bright', num_topics)
#         # Colour points for each label, if the label is -1 (deemed noise) then colour it grey.
#         cluster_colors = [color_palette[x] if x >= 0
#                           else (0.5, 0.5, 0.5)
#                           # else (1, 1, 1)
#                           for x in cluster_labels]
#         cluster_member_colors = [sns.desaturate(x, p) for x, p in
#                                  zip(cluster_colors, clusterer.probabilities_)]
#
#         # Create Scatter plot of 2d vectors
#         fig, ax = plt.subplots()
#
#         scatter = ax.scatter(*data.T, linewidth=0, c=cluster_member_colors, alpha=1, s=4.5)
#         plt.savefig("graphs/" + str(name) + "_noise.svg", format="svg")
#
#     elif noise == False:
#         color_palette = sns.color_palette('bright', num_topics)
#         # Colour points for each label, if the label is -1 (deemed noise) then colour it grey.
#         cluster_colors = [color_palette[x] if x >= 0
#                           #else (0.5, 0.5, 0.5)
#                           else (1, 1, 1)
#                           for x in cluster_labels]
#         cluster_member_colors = [sns.desaturate(x, p) for x, p in
#                                  zip(cluster_colors, clusterer.probabilities_)]
#
#         # Create Scatter plot of 2d vectors
#         fig, ax = plt.subplots()
#
#         labels = np.float32(cluster_labels)
#
#         data_df = pd.DataFrame(data)
#         data_df[3] = labels
#         data_df[4] = cluster_member_colors
#         data_df[5] = doc_labels
#         data_df.columns = ["x", "y", "c_label", "colors", "d_label"]
#
#
#
#         # Drop Noise
#         index = data_df[data_df["c_label"] == -1].index
#         data_df.drop(index, inplace=True)
#
#         print(data_df["d_label"].value_counts())
#
#         __data = np.array(data_df[["x", "y"]])
#
#         scatter = ax.scatter(*__data.T, linewidth=0, c=data_df["colors"].values.tolist(), alpha=1, s=4.5)
#
#         # Plot Labels in center of each cluster
#         for i, label in tqdm(enumerate(data_df["d_label"].unique())):
#             ax.annotate(int(label),
#                         data_df.loc[data_df['d_label'] == label, ['x', 'y']].mean(),
#                         horizontalalignment='center',
#                         verticalalignment='center',
#                         size=10, #weight='bold',
#                         )#color=color_palette[int(label)]
#         plt.savefig("graphs/" + str(name) + "_noise_removed.svg", format="svg")
#
#
#
#     # Generate legend from all document labels.
#     # As -1 is used to denote noisy docs, increase everything by 1.
#     """legend_range = list(range(labels.min()+1, labels.max()+1))
#     legend1 = ax.legend(*scatter.legend_elements(),
#                         loc="lower left", title="Topic ID")
#     ax.add_artist(legend1)"""
#     #plt.show()
#
#
#
# scatter_plot(embeddings.data, model.num_topics, embeddings.cluster, "d2v_master", noise=False, model=model)
# topic_sizes, topic_nums = documents.get_topic_sizes()
#
# topic_words, word_scores, topic_nums = documents.get_topics()
# for i, words  in enumerate(topic_words):
#     print("Topic ID: " + str(topic_nums[i]))
#     print(words)
#     print("\n")
#
#
# # Perform topic modelling of the smaller sub-topic
# documents = pd.DataFrame(documents)
# documents["topic_id"] = model.doc_top
# for topic_id in topic_nums:
#     print("TopicID: " + str(topic_id))
#     sub_documents = documents.loc[documents["topic_id"]==topic_id]
#     print("Topic Length: "+str(len(sub_documents)))
#     #print(topic_words)
#
#
# vecs = model._get_document_vectors(norm=False)
# vecs_reduced = model.data
# labels = model.doc_top
# c_labels = model.cluster.labels_
#
# sil_data = pd.DataFrame(list(zip(vecs, vecs_reduced, labels, c_labels)))
# sil_data.columns = ["vecs", "vecs_reduced", "labels", "c_labels"]
# sil_data["topic_id"] = documents["topic_id"]
# # Drop Noise
# index = sil_data[sil_data["c_labels"] == -1].index
# sil_data.drop(index, inplace=True)
#
# """tlist = [0,5,7,8,12, 13,14,17,19,28]
# sil_data = sil_data[sil_data["topic_id"].isin(tlist)]"""
#
# from sklearn.metrics import silhouette_score
# score = silhouette_score(sil_data["vecs_reduced"].values.tolist(), sil_data["labels"].values.tolist(), metric='euclidean')
# print("SCORE: " + str(score))














#
#
#
#
#
#
#
# def scatter_plot(data, num_topics, clusterer, name, noise, model):
#     cluster_labels = clusterer.labels_
#     doc_labels = model.doc_top
#
#     if noise == True:
#         color_palette = sns.color_palette('bright', num_topics)
#         # Colour points for each label, if the label is -1 (deemed noise) then colour it grey.
#         cluster_colors = [color_palette[x] if x >= 0
#                           else (0.5, 0.5, 0.5)
#                           # else (1, 1, 1)
#                           for x in cluster_labels]
#         cluster_member_colors = [sns.desaturate(x, p) for x, p in
#                                  zip(cluster_colors, clusterer.probabilities_)]
#
#         # Create Scatter plot of 2d vectors
#         fig, ax = plt.subplots()
#
#         scatter = ax.scatter(*data.T, linewidth=0, c=cluster_member_colors, alpha=1, s=4.5)
#         plt.savefig("graphs/" + str(name) + "_noise.svg", format="svg")
#
#     elif noise == False:
#         color_palette = sns.color_palette('bright', num_topics)
#         # Colour points for each label, if the label is -1 (deemed noise) then colour it grey.
#         cluster_colors = [color_palette[x] if x >= 0
#                           #else (0.5, 0.5, 0.5)
#                           else (1, 1, 1)
#                           for x in cluster_labels]
#         cluster_member_colors = [sns.desaturate(x, p) for x, p in
#                                  zip(cluster_colors, clusterer.probabilities_)]
#
#         # Create Scatter plot of 2d vectors
#         fig, ax = plt.subplots()
#
#         labels = np.float32(cluster_labels)
#
#         data_df = pd.DataFrame(data)
#         data_df[3] = labels
#         data_df[4] = cluster_member_colors
#         data_df[5] = doc_labels
#         data_df.columns = ["x", "y", "c_label", "colors", "d_label"]
#
#
#
#         # Drop Noise
#         index = data_df[data_df["c_label"] == -1].index
#         data_df.drop(index, inplace=True)
#
#         print(data_df["d_label"].value_counts())
#
#         __data = np.array(data_df[["x", "y"]])
#
#         scatter = ax.scatter(*__data.T, linewidth=0, c=data_df["colors"].values.tolist(), alpha=1, s=4.5)
#
#         # Plot Labels in center of each cluster
#         for i, label in tqdm(enumerate(data_df["d_label"].unique())):
#             ax.annotate(int(label),
#                         data_df.loc[data_df['d_label'] == label, ['x', 'y']].mean(),
#                         horizontalalignment='center',
#                         verticalalignment='center',
#                         size=10, #weight='bold',
#                         )#color=color_palette[int(label)]
#         plt.savefig("graphs/" + str(name) + "_noise_removed.svg", format="svg")
#
#
#
#     # Generate legend from all document labels.
#     # As -1 is used to denote noisy docs, increase everything by 1.
#     """legend_range = list(range(labels.min()+1, labels.max()+1))
#     legend1 = ax.legend(*scatter.legend_elements(),
#                         loc="lower left", title="Topic ID")
#     ax.add_artist(legend1)"""
#     #plt.show()
#
#
# scatter_plot(model.data, model.num_topics, model.cluster, "d2v_master", noise=False, model=model)