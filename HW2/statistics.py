import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import statistics
from networkx.algorithms.community import girvan_newman
import seaborn as sn


PATH = "il.csv"
TOP = 10

def first(l):
    return l[1]

graph = pd.read_csv(PATH)

filltered = graph[graph["Weight"] >= 3]

GA = nx.from_pandas_edgelist(filltered, source="Source", target="Target", edge_attr="Weight", create_using=nx.DiGraph())
print(nx.info(GA))

degree_centrality = nx.degree_centrality(GA)
print("degree_centralityrality: " + str(sorted(degree_centrality.items(), key=lambda x:first(x), reverse=True)[:TOP]))

closeness_centrality = nx.closeness_centrality(GA)
print("closeness_centrality: " + str(sorted(closeness_centrality.items(), key=lambda x:first(1), reverse=True)[:TOP]))

betweenness_centrality = nx.betweenness_centrality(GA)
print("betweenness_centrality: " + str(sorted(betweenness_centrality.items(), key=lambda x:first(1), reverse=True)[:TOP]))

eighenvector_centrality = nx.eigenvector_centrality(GA, max_iter=1500)
print("Eighenvector_centrality: " + str(sorted(eighenvector_centrality.items(), key=lambda x:first(1), reverse=True)[:TOP]))

cc = nx.clustering(GA)
print("Mean Clustering: " + str(statistics.mean(cc.values())))
print("Cluster Coefficient: " + str(sorted(cc.items(), key=lambda x:first(1), reverse=True)[:TOP]))

pagerank = nx.pagerank(GA, max_iter=500)
print("Page rank: " + str(sorted(pagerank.items(), key=lambda x:first(1), reverse=True)[:TOP]))

adar = nx.adamic_adar_index(GA.to_undirected())
adar_dict = {}
for u, v, p in adar:
    adar_dict[(u, v)] = p
print("Adamic - Adar: " + str(sorted(adar_dict.items(), key=lambda x:first(1), reverse=True)[:TOP]))

pa = nx.preferential_attachment(GA.to_undirected())
pa_dict = {}
for u, v, p in pa:
    pa_dict[(u, v)] = p
print("Preferential Attachment: " + str(sorted(pa_dict.items(), key=lambda x:first(1), reverse=True)[:TOP]))
