from optparse import OptionParser
import pandas as pd, numpy as np
import matplotlib.pyplot as pp
from matplotlib import mlab

from objects import HR
from settings import colors, cmap_pale_pink, cmap_pale_pink_and_green

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


def draw_discrete_distributions(hr, vars, vars_pretty_print, ):
	for var, var_pretty_print, in zip(vars, vars_pretty_print):
		draw_discrete_distribution(hr, var, var_pretty_print)


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
		c = []
		for i in range(len(var_val)):
			c.append(int(round(hr.data["time_spend_company"][i] * var_val[i])))
			i += 1
		var_val = c
		hr.discretize["last_evaluation"] = c
		vars = ['time_spend_company', 'last_evaluation']
		vars_pretty_print = ['years in company', 'last evaluation']
		draw_scatter_plots(hr, vars, vars_pretty_print)


	# add a 'best fit' lin
	#pp.scatter(var1, var2)
	#pp.showval)
	var_val_std = hr.std[var]
	var_val_mean = hr.mean[var]
	num_bins = int(np.ceil(np.log2(len(var_val))) + 1)

	figure, axes = pp.subplots()

	# add a 'best fit' line
	n, bins, patches = axes.hist(var_val, num_bins, label='Data distribution', color=colors['main'], normed=False)
	y = mlab.normpdf(bins, var_val_mean, var_val_std) * sum(n * np.diff(bins))
	axes.plot(bins, y, '.-.', color=colors['secondary'])
	axes.set_xlabel(str(var_pretty_print))
	axes.set_ylabel('Employees')
	axes.set_title(r'Distribution: $\mu$ =' + format(var_val_std, '.4f') + ', $\sigma$ =' + format(var_val_mean, '.4f'))
	pp.title(str(var_pretty_print) + ': data distribution')
	pp.savefig(str(var_pretty_print) + '.png')


def draw_discrete_distribution(hr, var, var_pretty_print):
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

	pp.bar(x_values, frequency, color=colors['main'])
	axes.set_xticks(x_values)
	axes.set_ylabel('Employees')
	axes.set_title(str(var_pretty_print))

	pp.tight_layout()
	pp.savefig(str(var_pretty_print) + '.png')


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

	bars = pp.bar(x_list, frequency)
	bars[0].set_color(colors['main'])
	bars[1].set_color(colors['secondary'])
	pp.xticks(x_list, x_values)
	axes.set_ylabel('Employees')
	axes.set_title(str(var_pretty_print))
	axes.set_xticklabels(axes.get_xticklabels(), rotation=45, ha='right')

	pp.tight_layout()
	pp.savefig(str(var_pretty_print) + '.png')


def draw_correlation_matrix(correlation_matrix, pretty_labels, normalised=False):
	figure, axes = pp.subplots(1, 1)

	#image = axes.imshow(correlation_matrix, cmap=cmap_pale_pink, vmin=0, vmax=0.5)
	mask = np.tri(correlation_matrix.shape[0], k=-1)
	matrix = np.ma.array(correlation_matrix, mask=mask)  # mask out the lower triangle
	cmap_pale_pink.set_bad('w')
	image = axes.imshow(matrix, cmap=cmap_pale_pink, interpolation='none')
	pp.colorbar(image)

	# Add labels, slightly modify for graph purposes
	axes.set_xticks(np.arange(0, correlation_matrix.shape[0], correlation_matrix.shape[0] * 1.0 / len(pretty_labels)))
	axes.set_yticks(np.arange(0, correlation_matrix.shape[1], correlation_matrix.shape[1] * 1.0 / len(pretty_labels)))
	axes.set_xticklabels(pretty_labels)
	axes.set_yticklabels(pretty_labels)

	# Adjust axes rotation
	axes.set_xticklabels(pretty_labels, rotation=45, ha='right')

	# Hide borders
	axes.spines['top'].set_visible(False)
	axes.spines['right'].set_visible(False)
	pp.draw()

	#pp.tight_layout()

	if not(normalised):
		figure.suptitle('Correlation matrix', fontsize=12, fontweight='bold')
		pp.savefig('Correlation matrix.png')
	else:
		figure.suptitle('Correlation matrix for normalised data', fontsize=12, fontweight='bold')
		pp.savefig('Correlation matrix for normalised data.png')


def draw_scatter_plots(hr, vars, vars_pretty_print):
	"""
	Draw the scatter plot of the two variables.
	:param hr: 					The data object.
	:param vars: 				The variables for the scatter plot.
	:param vars_pretty_print: 	The variables' pretty print name.
	:return: 					Nothing.
	:Note: For the discretized features there is a hr.discretize in objects. For now only last_eval (but who knows)
	"""
	var_val1 = hr.data[vars[0]]
	var_val2 = hr.data[vars[1]]

	# Fix for last_evaluation
	if vars[0] == "last_evaluation":
		var_val1 = hr.discretize["last_evaluation"]

	if vars[1] == "last_evaluation":
		var_val2 = hr.discretize["last_evaluation"]


	figure, axes = pp.subplots()

	# add a 'best fit' line
	#n, bins, patches = axes.hist(var_val, num_bins, label='Scatter Plot', color=colors['main'], normed=False)
	#y = mlab.normpdf(bins, var_val_mean, var_val_std) * sum(n * np.diff(bins))
	axes.scatter(var_val1, var_val2)
	axes.set_xlabel(str(vars_pretty_print[0]))
	axes.set_ylabel(str(vars_pretty_print[1]))
	#axes.set_title(r'Scatter plot: '+var_val1+var_val2)
	pp.title(str(vars_pretty_print[0])+str(vars_pretty_print[1]) + 'scatter')
	pp.savefig(str(vars_pretty_print[0])+str(vars_pretty_print[1]) + '.png')




def sample(values, k, bins, bin_labels, samples_number, replace=False, coverage={}):
	"""
	Sample the given dataframe var for a sample of size k from equi-width bins.
	:param var: 		The dataframe to sample.
	:param k: 			The sample size.
	:param bins:		The number of bins where to extract from.
	:param bin_labels:	The bins labels to build the returned dictionary.
	:return: 			A dictionary {bin_label: {([sample_1], bin_size),...,([sample_k], bin_size)}}
	"""
	unique_values = values.drop_duplicates()
	retbins = (pd.cut(unique_values, bins=bins, retbins=True, labels=list(map(str,range(bins)))))[0]
	value_bin = list(zip(retbins, unique_values))
	samples = {}

	for bin in bin_labels:
		bin_values = (pd.DataFrame(list(filter(lambda x: x[0] == bin, value_bin))))[1]
		buckets = list()

		for i in range(samples_number):
			sample = list(bin_values.sample(k, replace=replace))
			buckets.append(sample)

		samples[bin] = buckets
		coverage[bin] = bin_values.size


	return samples


# Main
if __name__ == '__main__':
	labels = ['satisfaction_level', 'last_evaluation', 'average_montly_hours', 'time_spend_company', 'number_project',
			  'Work_accident', 'left', 'promotion_last_5years', 'sales', 'salary']
	pretty_prints = ['Self-reported satisfaction', 'Time since last valuation, in weeks', 'AVG Monthly hours',
					 'Time in company in years', 'Projects', 'Accident', 'Left', 'Promoted (5 years)', 'Sales',
					 'Salary']
	labels_pretty_print = {k: v for k, v in zip(labels, pretty_prints)}
	continuous_labels = labels[0:3]
	discrete_labels = labels[3:5]
	categorical_labels = labels[5:]
	categorical_labels_boolean = [('Injured', 'Safe'), ('Left', 'Stayed'), ('Promoted', 'Not promoted')]

	data = pd.read_csv("./hr.csv")
	hr = HR(data, normalised=True)

	parser = OptionParser()
	parser.add_option("-d", "--distribution", dest="distributions",
					  help="List distributions to plot, comma separated", metavar="DISTRIBUTIONS")
	parser.add_option("--correlation", action="store_true", dest="correlation")

	
	(options, args) = parser.parse_args()
	draw_correlation = options.correlation

	if not (options.distributions is None):
		distributions = options.distributions.split(",")
	else:
		distributions = list()

	if distributions.count("all") > 0:
		draw_distributions(hr, continuous_labels, pretty_prints[:3])
		draw_discrete_distributions(hr, discrete_labels, pretty_prints[3:5])
		draw_categorical_distributions(hr, labels[5:], pretty_prints[5:], categorical_labels_boolean)
	else:
		discrete_vars = set(distributions).intersection(discrete_labels)
		continuous_vars = set(distributions).intersection(continuous_labels)
		for var in discrete_vars:
			var_pretty_print = labels_pretty_print[var]
			draw_discrete_distribution(hr, var, var_pretty_print)
		for var in continuous_vars:
			var_pretty_print = labels_pretty_print[var]
			draw_distribution(hr, var, var_pretty_print)

	if draw_correlation:
		normalised_data = pd.DataFrame(hr.normalised)
		draw_correlation_matrix(data.corr(), pretty_prints)
		draw_correlation_matrix(normalised_data.corr(), pretty_prints, normalised=True)

	pp.show()