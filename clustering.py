import pickle
import itertools
import matplotlib.pyplot as pp

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

from settings import *


clustering = {
	"runs": 1000,
	"init": "random",
	"precompute_distances": True,
	"algorithm": "elkan",
	"ks": list(range(2,15)) + [20, 30, 40, 50, 75, 100]
}

metrics = ["euclidean", "minkowski", "cityblock", "chebyshev"]


def cluster(hr):
	combinations = list(itertools.combinations(correlated_labels, 2))
	datasets = [hr.discrete, hr.normal, hr.data]
	titles = ["clusters.discrete.pickle", "clusters.normal.pickle", "clusters.raw.pickle"]

	for dataset, title in zip(datasets, titles):
		kmeans = {}
		for i, (var_x, var_y) in enumerate(combinations):
			entry_title = str(labels_pretty_print[var_x]) + " - " + str(labels_pretty_print[var_y])

			entry = {clusters: k_means(df=dataset[[var_x, var_y]], k=clusters)
					 for clusters in clustering["ks"]}

			entry["cohesions"] = {metric: {k: sum(list(map(lambda x: x[metric], list(entry[k][1]["cohesion"].values()))))
										   for k in clustering["ks"]} for metric in metrics}
			entry["average separations"] = {metric: [sum(sum([entry[k][1]["separation"][j][metric] for j in range(k)])) / k
													 for k in clustering["ks"]] for metric in metrics}
			entry["graph cohesions"] = {metric: [sum([sum(sum(cdist(df[df["cluster"] == i], df[df["cluster"] == i])))
													  for i in set(df["cluster"])]) for df in list(map(lambda x: x[0], entry.values()))]
										for metric in metrics}
			entry["graph separations"] = {metric: [sum([sum(sum(cdist(df[df["cluster"] == i], df[df["cluster"] != i])))
														for i in set(df["cluster"])]) for df in list(map(lambda x: x[0], entry.values()))]
										  for metric in metrics}
			entry["silohuettes"] = {metric: [entry[k][1]["silohuette score"][metric] for k in clustering["ks"]]
									for metric in metrics}

			kmeans[entry_title] = entry
			print("Done for " + entry_title)

		#Save to file
		with open(title, "wb") as log:
			pickle.dump(obj=kmeans, file=log, protocol=pickle.HIGHEST_PROTOCOL)

		#draw_clusters_validation(kmeans)


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

	kmeans = KMeans(n_clusters=k, n_init=clustering["runs"], init=clustering["init"], precompute_distances=clustering["precompute_distances"], algorithm=clustering["algorithm"])
	columns = df.columns
	columns.remove("idx")
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
	results["cohesion"] = {cluster_idx: {metric:
							float(sum(cdist(results["clusters"][cluster_idx], np.reshape(centroids[cluster_idx], (-1, 2)),
							metric=metric))) for metric in metrics} for cluster_idx in range(k)}

	# Prototype separation
	results["separation"] = {cluster_idx: {metric:
							   abs(sum(cdist(results["clusters"][cluster_idx],
								[el for i, el in enumerate(centroids) if i != cluster_idx], metric=metric)))
								for metric in metrics} for cluster_idx in range(k)}

	# Silohuette
	results["silohuette score"] = {metric: silhouette_score(df, kmeans.labels_, metric=metric) for metric in metrics}

	return (df_prime, results)


def draw_clusters_validation(kmeans):
	for combination in kmeans.keys():
		entry = kmeans[combination]
		cohesions = entry["cohesions"]
		average_separations = entry["separation"]
		graph_cohesions = entry["graph cohesions"]
		graph_separations = entry["graph separation"]
		silohuettes = entry["silhouettes"]

		## Plotting
		for metric in metrics:
			cohesion = list(cohesions[metric].values())
			separation = list(average_separations[metric])
			silohuettes_asymptote_height = max(max(cohesion), max(separation))/2
			scaled_silohuettes = list(map(lambda x: x * silohuettes_asymptote_height, list(silohuettes[metric])))

			figure, axes = pp.subplots()
			colors = [large_palette_full["navy"], large_palette_full["red"], large_palette_full["green"], large_palette_full["yellow"]]

			axes.plot(clustering["ks"], cohesion, label="Cohesion", color=colors[0])
			axes.plot(clustering["ks"], separation, label="Separation", color=colors[1])
			axes.plot(clustering["ks"], scaled_silohuettes, label="Silhouette", color=colors[2])
			axes.axhline(y=silohuettes_asymptote_height, linestyle="dashed", color=colors[3])
			title = combination + "\n[" + metric + "]"
			axes.set_xlabel("Clusters")
			legend = axes.legend(loc="best")
			pp.title(title)
			pp.savefig(title + ".png", bbox_extra_artists = [legend])
			pp.savefig(title + ".svg", bbox_extra_artists = [legend])

			# Graph measures
			graph_cohesion = graph_cohesions[metric]
			graph_separation = graph_separations[metric]
			silohuettes_asymptote_height = max(max(graph_cohesion), max(graph_separation))/2
			scaled_silohuettes = list(map(lambda x: x * silohuettes_asymptote_height, list(silohuettes[metric])))

			figure, axes = pp.subplots()
			axes.plot(clustering["ks"], graph_cohesion, label="Graph cohesion", color=colors[0])
			axes.plot(clustering["ks"], graph_separation, label="Graph separation", color=colors[1])
			axes.plot(clustering["ks"], scaled_silohuettes, label="Silhouette", color=colors[2])
			axes.axhline(y=silohuettes_asymptote_height, linestyle="dashed", color=colors[3])
			axes.set_xlabel("Clusters")
			legend = axes.legend(loc="best")
			title = "Graph " + combination + "\n[" + metric + "]"
			pp.title(title)
			pp.savefig(title + ".png", bbox_extra_artists = [legend])
			pp.savefig(title + ".svg", bbox_extra_artists = [legend])


def draw_clustered_scatter_plot(df, centroids, colors=large_palette_full, title=""):
	"""
	Draw the scatter plot of the two variables.
	:param df:					The
	:param colors:				The color palette to use.
	:param title:				The plot title.
	:return: 					Nothing.
	"""
	columns = df.columns
	columns.remove("idx")
	columns.remove("cluster")
	df_prime = df[columns]
	clusters_idxs = len(set(df["cluster"]))

	figure, axes = pp.subplots()

	for cluster_idx, color_key in zip(range(clusters_idxs), colors.keys()):
		points = df[df["cluster"] == cluster_idx][columns]
		axes.scatter(points[columns[0]], points[columns[1]], color=colors[color_key])
		axes.scatter(centroids[0], centroids[1], color=colors[color_key])

	axes.set_xlabel(labels_pretty_print[columns[0]])
	axes.set_ylabel(labels_pretty_print[columns[1]])
	pp.savefig(title + ".png")
	pp.savefig(title + ".svg")