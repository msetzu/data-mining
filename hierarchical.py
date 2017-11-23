import pickle
import itertools
import threading
import matplotlib.pyplot as pp
import numpy as np
import pandas as pd
from settings import *
from objects import HR
from scipy.spatial.distance import pdist
from sklearn.metrics import *
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import mode
from scipy.spatial.distance import pdist
import sys
import settings
from settings import *
from optparse import OptionParser
from scipy.cluster.hierarchy import linkage, dendrogram

def data_analysis(hr):
    parsed_arguments = parse_arguments()
    metrics = ["euclidean", "minkowski", "cityblock", "chebyshev", "cosine"]
    metrics2 = ["euclidean", "l1", "l2", "manhattan", "cosine"]


    if parsed_arguments["single"]:
        for metric in metrics:
            plot_dendrogram(hr, metric, "single", 3)
    if parsed_arguments["complete"]:
        #for metric in metrics:
            #plot_dendrogram(hr, metric, "complete", 3)
        for metric in metrics2:
            calculate_sil(hr, metric, "complete")    
    if parsed_arguments["average"]:
        for metric in metrics:
            plot_dendrogram(hr, metric, "average", 3)
    if parsed_arguments["weighted"]:
        for metric in metrics:
            plot_dendrogram(hr, metric, "weighted")
    if parsed_arguments["median"]:
        for metric in metrics:
            plot_dendrogram(hr, metric, "median")
    if parsed_arguments["ward"]:
        for metric in metrics:
            plot_dendrogram(hr, metric, "ward")


def parse_arguments():
    arguments = {}
    parser = OptionParser()
    parser.add_option("--single", action="store_true", dest="single", help="Show hier with linkage single")
    parser.add_option("--complete", action="store_true", dest="complete", help="Show hier with linkage complete")
    parser.add_option("--weighted", action="store_true", dest="weighted", help="Show hier with linkage weighted")
    parser.add_option("--median", action="store_true", dest="median", help="Show hier with linkage median")
    parser.add_option("--ward", action="store_true", dest="ward", help="Show hier with linkage ward")
    parser.add_option("--average", action="store_true", dest="average", help="Show hier with linkage average")
    
    (options, args) = parser.parse_args()
    arguments["single"] = options.single
    arguments["complete"] = options.complete
    arguments["weighted"] = options.weighted
    arguments["median"] = options.median
    arguments["ward"] = options.ward
    arguments["average"] = options.average

    return arguments

def clean_data(hr, variables):
    if variables == 1:
        hr = hr.drop(['Work_accident','left','sales'], axis=1)
        #clean for salary: int low - 10000 medium - 25000 high - 50000
        hr = hr.replace(["low", "medium", "high"] , [10000, 25000, 50000])
    elif variables == 2:
        #tengo solo num di progetti, left e salario
        hr = hr.drop(['satisfaction_level','average_montly_hours','last_evaluation','time_spend_company','Work_accident', 'left', 'promotion_last_5years','sales'], axis=1)
        hr = hr.replace(["low", "medium", "high"] , [10000, 25000, 50000])
    elif variables == 3:
        #tengo solo mum progetti avg mon hours
        hr = hr.drop(['satisfaction_level','last_evaluation','time_spend_company','Work_accident','left','promotion_last_5years','sales','salary'], axis=1)    
    return hr

def calculate_sil(hr, metric, method):

    #connectivity = kneighbors_graph(hr, n_neighbors=100, include_self=False)
    #connectivity = 0.5 * (connectivity + connectivity.T)
    nlinkage = AgglomerativeClustering(n_clusters=4, linkage=method, affinity=metric) #connectivity=connectivity)
    nlinkage.fit(hr)
    #provare con dump delle link labels
    hist, bins = np.histogram(nlinkage.labels_, bins=range(0, len(set(nlinkage.labels_)) + 1))
    print (dict(zip(bins, hist)))
    print (silhouette_score(hr, nlinkage.labels_))     
    
def plot_dendrogram(hr, metric, method, print):
    data_dist = pdist(hr.values, metric=metric)
    data_link = linkage(data_dist, method=method, metric=metric)
    pp.xlabel('matrix '+metric)
    pp.ylabel('distance '+method)
    dendrogram(
        data_link,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=20,  # show only the last p merged clusters
        show_leaf_counts=True,  # otherwise numbers in brackets are counts
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )

    if print == 1:
        pp.title('Hierarchical with all variables')
        #metti show e poi salvi dopo
        pp.savefig('H_all_variables_'+metric+'_'+method+'.png')
    elif print == 2:
        pp.title('Hierarchical with left, number of projects, salary')
        pp.savefig('H_projects_left_salary_'+metric+'_'+method+'.png')
    elif print == 3:
        pp.title('Hierarchical with number of projects and avg monthly hours')
        pp.savefig('H_projects_avg_hours_'+metric+'_'+method+'.png')   
   


if __name__ == '__main__':

    sys.setrecursionlimit(15000)
    #silo per un solo punto con ciclo for 
    #per db scan: assumendo un min point di 4, prendere la distanza maggiore dal il core, per ogni core, ordinare e plottare. Ad esempio, aggiungere la dist dal 4 punto, ordinare in ordine crescente di distanza dal core, plottare usare quel valore come parametri
    hr = clean_data(pd.read_csv("hr.csv"), 3)
    
    data_analysis(hr)


