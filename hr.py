from optparse import OptionParser

import pandas as pd, numpy as np
import matplotlib.pyplot as pp
from matplotlib import mlab

from objects import HR

bins = 10
data_balance_treshold = 0.05


def data_balance(hr, var, var_pretty_print):
	"""
	Define the data distribution and balance: group by bins, discretize into bins and check the distance from the mean.
	A little distance w.r.t. the variance indicates great distance from the mean w.r.t. all the other bins, hence
	we might classify this bin as an 'outlier' bin.
	:param hr:				The data object.
	:param var:				The variable whose balance to compute.
	:param var_pretty_print:The variable's pretty print name.
	:return:				Nothing.
	"""
	values = hr.data[var]
	var_quantiles = pd.qcut(values, 10, labels=list(map(lambda x: str(x), list(range(0, 10)))), duplicates="drop")
	quantiles_count = var_quantiles.value_counts()
	bins_mean = quantiles_count.mean()
	bins_std = quantiles_count.std()
	bins_distances = list(map(lambda x: abs(x - bins_mean), quantiles_count))
	bins_weighted_distances = list(map(lambda x: x / bins_std, bins_distances))

	if max(bins_weighted_distances) / bins >= data_balance_treshold:
		print("Quantiles for " + str(var_pretty_print) + ". \t[BALANCED]")
	else:
		print("Quantiles for " + str(var_pretty_print) + ". \t[UNBALANCED]")


def draw_distributions(hr, vars, vars_pretty_print):
	for var, var_pretty_print in zip(vars, vars_pretty_print):
		draw_distribution(hr, var, var_pretty_print)


def draw_discrete_distributions(hr, vars, vars_pretty_print, suffixes):
	for var, var_pretty_print, suffix in zip(vars, vars_pretty_print, suffixes):
		draw_discrete_distribution(hr, var, var_pretty_print, suffix)


def draw_categorical_distributions(hr, vars, vars_pretty_print, vars_pretty_print_categories):
	for var, var_pretty_print, categories in zip(vars, vars_pretty_print, vars_pretty_print_categories):
		draw_categorical_distribution(hr, var, var_pretty_print, categories)


def draw_distribution(hr, var, var_pretty_print):
	"""
	Draw the variables' distributions.
	:param hr: 					The data object.
	:param var: 				The variable whose distribution to draw.
	:param var_pretty_print: 	The variable's pretty print name.
	:return: 					Nothing.
	"""
	var_val = hr.data[var]

	# Fix for last_evaluation
	if var == "last_evaluation":
		var_val = list(map(lambda x: x * 365 / 7, var_val))

	var_val_std = hr.std[var]
	var_val_mean = hr.mean[var]
	num_bins = int(np.ceil(np.log2(len(var_val))) + 1)

	figure, axes = pp.subplots()

	# add a 'best fit' line
	n, bins, patches = axes.hist(var_val, num_bins, label='Data distribution', normed=1)
	y = mlab.normpdf(bins, var_val_mean, var_val_std)
	axes.plot(bins, y, '.-.')
	axes.set_xlabel(str(var_pretty_print))
	axes.set_ylabel('Employees')
	axes.set_title(r'Distribution: $\mu$ =' + format(var_val_std, '.4f') + ', $\sigma$ =' + format(var_val_mean, '.4f'))
	pp.title(str(var_pretty_print) + ': data distribution')


def draw_discrete_distribution(hr, var, var_pretty_print, suffix):
	"""
    Draw the distcrete variables' distributions.
    :param hr: 					The data object.
    :param var: 				The variable whose distribution to draw.
    :param var_pretty_print: 	The variable's pretty print name.
    :return: 					Nothing.
    """
	var_val = list(hr.data[var])

	figure, axes = pp.subplots()
	x_values = list(set(var_val))
	x_values.sort()

	frequency = list()
	for year in x_values:
		frequency.append(var_val.count(year))

	pp.bar(x_values, frequency)
	axes.set_xticks(x_values)
	axes.set_ylabel('Employees')
	axes.set_title(str(var_pretty_print) + " " + str(suffix))


def draw_categorical_distribution(hr, var, var_pretty_print, categories):
	"""
    Draw the distcrete variables' distributions.
    :param hr: 					The data object.
    :param var: 				The variable whose distribution to draw.
    :param var_pretty_print: 	The variable's pretty print name.
    :return: 					Nothing.
    """
	var_val = list(hr.data[var])

	figure, axes = pp.subplots()
	x_values = list(set(var_val))
	x_list = range(len(x_values))
	x_values.sort()
	x_values.reverse()

	frequency = list()
	for year in x_values:
		frequency.append(var_val.count(year))

	# Some categorical labels are '0', '1'
	if x_values == [1, 0]:
		x_values = categories

	pp.bar(x_list, frequency)
	pp.xticks(x_list, x_values)
	axes.set_ylabel('Employees')
	axes.set_title(str(var_pretty_print))
	pp.setp(axes.get_xticklabels(), rotation=45)


def draw_correlation_matrix(correlation_matrix):
	figure, axes = pp.subplots(1,1)
	figure.suptitle('Correlation matrix', fontsize=12, fontweight='bold')
	
	image = axes.imshow(correlation_matrix, cmap="Blues", vmin=0, vmax=0.5)
	pp.colorbar(image)

	# Add labels, slightly modify for graph purposes
	labels = ['satisfaction','last evaluation','projects','hours (monthly)','employment time','accidents','left','recent promotions','sales','salary']
	axes.set_xticks(np.arange(0,correlation_matrix.shape[0], correlation_matrix.shape[0]*1.0/len(labels)))
	axes.set_yticks(np.arange(0,correlation_matrix.shape[1], correlation_matrix.shape[1]*1.0/len(labels)))
	axes.set_xticklabels(labels)
	axes.set_yticklabels(labels)
	
	# Adjust axes rotation
	pp.setp(axes.get_xticklabels(), rotation=45)

	# Hide borders
	axes.spines['top'].set_visible(False)
	axes.spines['right'].set_visible(False)

	pp.draw()


# Main
if __name__ == '__main__':
	labels = ['satisfaction_level', 'last_evaluation', 'average_montly_hours', 'time_spend_company', 'number_project', 'Work_accident', 'left', 'promotion_last_5years', 'sales', 'salary']
	pretty_prints = ['Self-reported satisfaction', 'Time since last valuation, in weeks', 'AVG Monthly hours', 'Projects', 'Time in company in years', 'Accident', 'Left', 'Promoted (5 years)', 'Sales', 'Salary']
	labels_pretty_print = {k: v for k, v in zip(labels, pretty_prints)}
	continuous_labels = labels[0:3]
	discrete_labels = labels[3:5]
	discrete_suffixes = ['degree',  '']
	categorical_labels = labels[5:]
	categorical_labels_boolean = [('Injured', 'Safe'), ('Left', 'Stayed'), ('Promoted', 'Not promoted')]

	data = pd.read_csv("./hr.csv")
	hr = HR(data)

	parser = OptionParser()
	parser.add_option("-d", "--distribution", dest="distributions",
					  help="List distributions to plot, comma separated", metavar="DISTRIBUTIONS")
	parser.add_option("--correlation", action="store_true", dest="correlation")
	#parser.add_option("--balance", action="store_true", dest="balance")

	(options, args) = parser.parse_args()
	draw_correlation = options.correlation
	#draw_balance = options.balance
	if not(options.distributions is None):
		distributions = options.distributions.split(",")
	else:
		distributions = list()

	if distributions.count("all") > 0:
		draw_distributions(hr, continuous_labels, pretty_prints[:3])
		draw_discrete_distributions(hr, discrete_labels, pretty_prints[3:5], discrete_suffixes)
		draw_categorical_distributions(hr, labels[5:], pretty_prints[5:], categorical_labels_boolean)
	else:
		discrete_vars = set(distributions).intersection(discrete_labels)
		continuous_vars = set(distributions	).intersection(continuous_labels)
		for var in discrete_vars:
			var_pretty_print = labels_pretty_print[var]
			draw_discrete_distribution(hr, var, var_pretty_print)
		for var in continuous_vars:
			var_pretty_print = labels_pretty_print[var]
			draw_distribution(hr, var, var_pretty_print)

	if draw_correlation:
		draw_correlation_matrix(np.tril(data.corr()))

	# if draw_balance:
	# 	for feature, feature_pretty_print in zip(labels, labels_pretty_print):
	# 		data_balance(hr, feature, feature_pretty_print)

	pp.show()
