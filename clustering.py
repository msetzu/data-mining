import pickle
import itertools
import threading

import matplotlib.pyplot as pp

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

import settings
from settings import *

kmeans_clustering = {
    "runs": 1000,
    "init": "random",
    "precompute_distances": True,
    "algorithm": "elkan",
    "ks": list(range(2, 15)) + [20, 30, 40, 50, 75, 100]
}

dbscan_clustering = {
    "neighbors": [4, 6, 8, 10, 15, 25, 50, 100, 200]
}

metrics = ["euclidean", "minkowski", "cityblock", "chebyshev", "cosine"]


def cluster(hr):
    datasets = [hr.normal, hr.discrete]
    titles = ["clustering.normal.pickle", "clustering.discrete.pickle"]
    clustering_vars = [correlated_labels[:-1], correlated_labels]
    threads = []
    starts = [-1, -1]

    # KMeans
    # for dataset, clustering_var, title, start in zip(datasets, clustering_vars, titles, starts):
    #     thread = threading.Thread(target=cluster_kmeans, args=(dataset, clustering_var, title, start))
    #     threads.append(thread)
    #     thread.start()

    # DBscan
    # for dataset, clustering_var, title, in zip(datasets, clustering_vars, titles):
    #     thread = threading.Thread(target=eps_growths, args=(dataset, clustering_var, "dbscan.eps." + title))
    #     threads.append(thread)
    #     thread.start()
    # eps_growths(hr.normal, correlated_labels[:-1], "dbscan.eps.graphs")

    # for dataset, clustering_var, title, start in zip(datasets, clustering_vars, titles, starts):
    #     thread = threading.Thread(target=cluster_dbscan, args=(dataset, clustering_var, title, start))
    #     threads.append(thread)
    #     thread.start()


# noinspection PyTypeChecker
def cluster_kmeans(dataset, clustering_vars, title, start, draw_validation="False"):
    """
    Cluster the given dataset according to the provided clustering vars. Dump in a pickle file named tittle.
    :param dataset:				The dataset to cluster.
    :param clustering_vars:		The dataset variables to cluster.
    :param title:				The dump filename.
    :param start:				The variables combinations to skip.
    :return:					Nothing.
    """
    combinations = list(itertools.combinations(clustering_vars, 2))
    combinations = combinations[start:]

    if start != 0:
        with open(title, "rb") as log:
            kmeans = pickle.load(log)
    elif start == -1:
        return
    else:
        kmeans = {}

    for i, (var_x, var_y) in enumerate(combinations):
        entry_title = str(labels_pretty_print[var_x]) + " - " + str(labels_pretty_print[var_y])

        entry = {clusters: k_means(df=dataset[[var_x, var_y]], k=clusters)
                 for clusters in kmeans_clustering["ks"]}
        print("Done with kmeans")
        cohesions = {metric: {k: sum(list(map(lambda x: x[metric], list(entry[k][1]["cohesion"].values()))))
                              for k in kmeans_clustering["ks"]} for metric in metrics}

        print("Done with cohesions")
        average_separations = {metric: [sum(sum([entry[k][1]["separation"][j][metric] for j in range(k)])) / k
                                        for k in kmeans_clustering["ks"]] for metric in metrics}

        print("Done with separations")
        graph_cohesions = {metric: [sum([sum(sum(cdist(df[df["cluster"] == i], df[df["cluster"] == i])))
                                         for i in set(df["cluster"])]) for df in
                                    list(map(lambda x: x[0], entry.values()))]
                           for metric in metrics}

        print("Done with graph cohesions")
        graph_separations = {metric: [sum([sum(sum(cdist(df[df["cluster"] == i], df[df["cluster"] != i])))
                                           for i in set(df["cluster"])]) for df in
                                      list(map(lambda x: x[0], entry.values()))]
                             for metric in metrics}

        print("Done with graph separations")
        silhouettes = {metric: [entry[k][1]["silohuette score"][metric] for k in kmeans_clustering["ks"]]
                       for metric in metrics}

        entry["cohesions"] = cohesions
        entry["separation"] = average_separations
        entry["graph cohesions"] = graph_cohesions
        entry["graph separation"] = graph_separations
        entry["silhouettes"] = silhouettes
        kmeans[entry_title] = entry
        print("[" + title + "]" + " Done for " + entry_title)

        # Save to file
        with open(title, "wb") as log:
            pickle.dump(obj=kmeans, file=log, protocol=pickle.HIGHEST_PROTOCOL)

    if draw_validation:
        draw_clusters_validation(kmeans)
        draw_cluster_homogeneity(kmeans)


def k_means(df, k):
    """
    Compute k-means on the given dataframe df.
    :param df:					The dataframe to pick the values from.
    :param k:					The number of clusters.
    :return:					A tuple (df_prime, d), where
                                df_prime := enriched df with a "cluster" column.
                                d := {
                                    "cluste2rs":{cluster_idx: dataframe.filter("cluster" == cluster_idx)},
                                    "centroids": centroids,
                                    "inertia": SSE,
                                    "cohesion": {cluster_idx: {measure: SSE(cluster)}},
                                    "separation": {cluster_idx: {measure: separation(cluster)}},
                                    "silohuette": silohuette_score
                                }
    """
    results = {}

    kmeans = KMeans(n_clusters=k, n_init=kmeans_clustering["runs"], init=kmeans_clustering["init"],
                    precompute_distances=kmeans_clustering["precompute_distances"],
                    algorithm=kmeans_clustering["algorithm"])
    columns = list(filter("idx".__ne__, list(df.columns)))
    kmeans.fit(df[columns])

    # Centroid computation
    centroids = kmeans.cluster_centers_
    results["centroids"] = centroids

    # Cluster formation
    predicted_clusters = kmeans.predict(list(df.values))
    df_prime = df.assign(cluster=pd.Series(predicted_clusters))
    results["clusters"] = {cluster_idx: df[df_prime["cluster"] == cluster_idx] for cluster_idx in range(k)}

    # Inertia
    results["inertia"] = kmeans.inertia_

    # Prototype cohesion
    results["cohesion"] = {cluster_idx: {metric: float(sum(cdist(results["clusters"][cluster_idx],
                                         np.reshape(centroids[cluster_idx], (-1, 2)),
                                         metric=metric))) for metric in metrics} for cluster_idx in range(k)}

    # Prototype separation
    results["separation"] = {cluster_idx: {metric: abs(sum(cdist(results["clusters"][cluster_idx],
                                           [el for i, el in enumerate(centroids) if i != cluster_idx], metric=metric)))
                                           for metric in metrics} for cluster_idx in range(k)}

    # Silohuette
    results["silohuette score"] = {metric: silhouette_score(df, kmeans.labels_, metric=metric) for metric in metrics}

    return (df_prime, results)


def draw_clusters_validation(kmeans):
    """
    Draw the cluster validation graphs: show silhouette score (with asymptote), (graph)cohesion and (graph)separation.
    :param 	kmeans:		The kmeans dictionary.
    :return				Nothing.
    """
    for combination in kmeans.keys():
        entry = kmeans[combination]
        cohesions = entry["cohesions"]
        average_separations = entry["separation"]
        graph_cohesions = entry["graph cohesions"]
        graph_separations = entry["graph separation"]
        silohuettes = entry["silhouettes"]

        # Plotting
        for metric in metrics:
            cohesion = list(cohesions[metric].values())
            separation = list(average_separations[metric])
            silohuettes_asymptote_height = max(max(cohesion), max(separation)) / 2
            scaled_silohuettes = list(map(lambda x: x * silohuettes_asymptote_height, list(silohuettes[metric])))

            figure, axes = pp.subplots()
            colors = [large_palette_full["navy"], large_palette_full["red"], large_palette_full["green"],
                      large_palette_full["yellow"]]

            axes.plot(kmeans_clustering["ks"], cohesion, label="Cohesion", color=colors[0])
            axes.plot(kmeans_clustering["ks"], separation, label="Separation", color=colors[1])
            axes.plot(kmeans_clustering["ks"], scaled_silohuettes, label="Silhouette", color=colors[2])
            axes.axhline(y=silohuettes_asymptote_height, linestyle="dashed", color=colors[3])
            title = combination + "\n[" + metric + "]"
            axes.set_xlabel("Clusters")
            legend = axes.legend(loc="best")
            pp.title(title)
            pp.savefig(title.replace("\n", "") + ".png", bbox_extra_artists=[legend])
            pp.savefig(title.replace("\n", "") + ".svg", bbox_extra_artists=[legend])
            pp.clf()
            pp.cla()
            pp.close(figure)

            # Graph measures
            graph_cohesion = graph_cohesions[metric]
            graph_separation = graph_separations[metric]
            silohuettes_asymptote_height = max(max(graph_cohesion), max(graph_separation)) / 2
            scaled_silohuettes = list(map(lambda x: x * silohuettes_asymptote_height, list(silohuettes[metric])))

            figure, axes = pp.subplots()
            axes.plot(kmeans_clustering["ks"], graph_cohesion, label="Graph cohesion", color=colors[0])
            axes.plot(kmeans_clustering["ks"], graph_separation, label="Graph separation", color=colors[1])
            axes.plot(kmeans_clustering["ks"], scaled_silohuettes, label="Silhouette", color=colors[2])
            axes.axhline(y=silohuettes_asymptote_height, linestyle="dashed", color=colors[3])
            axes.set_xlabel("Clusters")
            axes.grid()
            legend = axes.legend(loc="best")
            title = "Graph " + combination + "\n[" + metric + "]"
            pp.title(title)
            pp.savefig(title.replace("\n", "") + ".png", bbox_extra_artists=[legend])
            pp.savefig(title.replace("\n", "") + ".svg", bbox_extra_artists=[legend])
            pp.close(figure)


def draw_cluster_homogeneity(kmeans, title_prefix=""):
    """
    Draw cluster homogeneity for the provided kmeans object.
    :param kmeans: 			The kmeans object.
    :param title_prefix: 	Perfix for the plotted graph filename.
    :return: 				Nothing.
    """
    for combination, k in itertools.product(kmeans.keys(), kmeans_clustering["ks"]):
        df = kmeans[combination][k][0]
        columns = df.columns

        # Plotting
        for cluster_variable in columns:
            frequencies_matrix = pd.crosstab(df[cluster_variable], df["cluster"])
            frequencies_percentages = frequencies_matrix.div(frequencies_matrix.sum(1).astype(float), axis=0)
            frequencies_percentages.plot(kind="bar", stacked=True, color=large_palette_full)
            pp.gca().legend_.remove()
            pp.savefig(title_prefix + "Heterogeneity for " + cluster_variable)
            pp.cla()
            pp.clf()


def draw_clustered_scatter_plot(kmeans, colors=large_palette_full):
    """
    Draw the scatter plot of the two variables.
    :param kmeans:  The kmeans object.
    :param colors:	The color palette to use.
    :return: 		Nothing.
    """
    for combination, k in itertools.product(kmeans.keys(), kmeans_clustering["ks"]):
        df = kmeans[combination][k][0]
        title = combination
        columns = list(df.columns)
        columns.remove("cluster")

        figure, axes = pp.subplots()
        for cluster_idx, color_key in zip(range(k), colors.keys()):
            points = df[df["cluster"] == cluster_idx][columns]
            axes.scatter(points[columns[0]], points[columns[1]], color=colors[color_key])

        axes.set_xlabel(labels_pretty_print[columns[0]])
        axes.set_ylabel(labels_pretty_print[columns[1]])
        pp.savefig(title + ".svg")
        pp.cla()
        pp.clf()
        pp.close(figure)


def eps_growths(df, clustering_vars, dump_file):
    """
    Draw the distance graph for the given dataframe according to the kth closest neighbor and distance measures.
    Dump to file_name a dictionary
        d := { entry := { metric := { k:= [d1, ..., dn]}}}
    with the distances of the kth nearest neighbor according to metric per given entry.
    :param df: 				The DataFrame object.
    :param clustering_vars:	The dataset variables to cluster.
    :param dump_file:       The dump file name.
    :return: 				Nothing.
    """
    entries = list(itertools.combinations(clustering_vars, 2))

    for (var_x, var_y) in entries:
        entry = var_x + "," + var_y
        pretty_entry = labels_pretty_print[var_x] + ", " + labels_pretty_print[var_y]
        print("Going for " + pretty_entry)
        df_prime = df[[var_x, var_y]]

        for metric in metrics:
            distance_matrix = cdist(df_prime, df_prime, metric=metric)
            # sort rows, then sort columns
            distance_matrix.sort(axis=1)
            distance_matrix.sort(axis=0)

            for k in dbscan_clustering["neighbors"]:
                print("Going for " + str(k) + " :" + pretty_entry)
                figure, axes = pp.subplots()
                distances = distance_matrix[:, k]

                axes.plot(range(df_prime.shape[0]), distances)
                axes.set_title(pretty_entry + "\n" + str(k) + "th neighbor " + metric + " distance")
                axes.set_xlabel("Points at distance eps")
                axes.set_ylabel("Distance")
                axes.grid()

                pp.savefig(pretty_entry + " :" + str(k) + "th neighbor " + metric + "distance" + ".svg")
                pp.clf()
                pp.cla()
                pp.close(figure)

                with open(entry + "-" + metric + "-" + str(k) + dump_file, "wb") as log:
                    pickle.dump(distances, log, protocol=pickle.HIGHEST_PROTOCOL)
