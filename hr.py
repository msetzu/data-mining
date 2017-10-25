from optparse import OptionParser
from matplotlib import mlab

from analysis import *
from settings import *
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


def draw_discrete_distributions(hr, vars, vars_pretty_print):
	for var, var_pretty_print, in zip(vars, vars_pretty_print):
		draw_discrete_distribution(hr, var, var_pretty_print, labels_pretty_print[var])


def draw_categorical_distributions(hr, vars, vars_pretty_prints):
	for var, var_pretty_print in zip(vars, vars_pretty_prints):
		draw_categorical_distribution(hr, var, categorical_labels_pretty_prints[var_pretty_print], labels_pretty_print[var])


def draw_distribution(hr, var, var_pretty_print):
	"""
	Draw the variables' distributions.
	:param hr: 					The data object.
	:param var: 				The variable whose distribution to draw.
	:param var_pretty_print: 	The variable's pretty print name.
	:return: 					Nothing.
	"""
	var_val = list(hr.data[var])
	var_val_std = hr.std[var]
	var_val_mean = hr.mean[var]
	num_bins = int(np.ceil(np.log2(len(var_val))) + 1)

	figure, axes = pp.subplots()

	n, bins, patches = axes.hist(var_val, num_bins, label='Data distribution', color=palette['main'], normed=False)
	y = mlab.normpdf(bins, var_val_mean, var_val_std) * sum(n * np.diff(bins))

	axes.plot(bins, y, '.-.', color=palette['secondary'])
	axes.set_xlabel(str(var_pretty_print))
	axes.set_ylabel('Employees')
	axes.set_title(r'Distribution: $\mu$ =' + format(var_val_std, '.4f') + ', $\sigma$ =' + format(var_val_mean, '.4f'))
	pp.title(str(var_pretty_print) + ': data distribution')
	pp.savefig(str(var_pretty_print) + '.png')


def draw_categorical_distribution(hr, var, var_pretty_prints, title):
	"""
	Draw the distcrete variables' distributions.
	:param hr: 					The data object.
	:param var: 				The variable whose distribution to draw.
	:param var_pretty_prints: 	The variable's pretty print name.
	:return: 					Nothing.
	"""
	var_val = list(hr.data[var])

	figure, axes = pp.subplots()
	x_values = list(set(var_val))
	x_ticks = range(len(x_values))
	x_values.sort()
	x_values.reverse()

	frequency = list()
	for x in x_values:
		frequency.append(var_val.count(x))

	if x_values == [1, 0]:
		x_ticks = list(var_pretty_prints)
		frequency.reverse()

	bars = pp.bar(x_ticks, frequency)
	bars[0].set_color(palette["main"])
	bars[1].set_color(palette["secondary"])
	axes.set_ylabel("Employees")
	axes.set_title(str(title))
	axes.set_xticklabels(x_ticks)
	pp.title(title)

	pp.tight_layout()
	pp.savefig(str(title) + '.png')


def draw_discrete_distribution(hr, var, var_pretty_print, title):
	"""
	Draw the distcrete variables' distributions.
	:param hr: 					The data object.
	:param var: 				The variable whose distribution to draw.
	:param var_pretty_print: 	The variable's pretty print name.
	:return: 					Nothing.
	"""
	var_val = list(hr.discrete[var])

	figure, axes = pp.subplots()
	x_values = list(set(var_val))
	x_values.sort()

	frequency = list()
	for x in x_values:
		frequency.append(var_val.count(x))

	pp.bar(x_values, frequency, color=palette['main'])
	axes.set_xticks(x_values)
	axes.set_ylabel('Employees')
	axes.set_title(str(title))

	pp.tight_layout()
	pp.savefig(str(var_pretty_print) + '.png')


def draw_correlation_matrix(correlation_matrix, pretty_labels, normalised=False):
	figure, axes = pp.subplots()

	mask = np.tri(correlation_matrix.shape[0], k=-1)
	matrix = np.ma.array(correlation_matrix, mask=mask)  # mask out the lower triangle
	cmap_pale_pink.set_bad('w')
	image = axes.imshow(matrix, cmap=cmap_pale_pink, interpolation='none')
	pp.colorbar(image)

	axes.set_xticks(np.arange(0, correlation_matrix.shape[0], correlation_matrix.shape[0] * 1.0 / len(pretty_labels)))
	axes.set_yticks(np.arange(0, correlation_matrix.shape[1], correlation_matrix.shape[1] * 1.0 / len(pretty_labels)))
	axes.set_xticklabels(pretty_labels)
	axes.set_yticklabels(pretty_labels)
	axes.set_xticklabels(pretty_labels, rotation=45, ha='right')

	# Hide borders
	axes.spines['top'].set_visible(False)
	axes.spines['right'].set_visible(False)
	pp.draw()

	if not(normalised):
		axes.set_title('Correlation matrix', fontsize=12, fontweight='bold')
		pp.savefig('Correlation matrix.png')
	else:
		axes.set_title('Correlation matrix', fontsize=12, fontweight='bold')
		pp.savefig('Correlation matrix for normalised data.png')


def draw_scatter_plot(hr, var_x, var_y, vars_pretty_print):
	"""
	Draw the scatter plot of the two variables.
	:param hr: 					The data object.
	:param vars: 				The variables for the scatter plot.
	:param vars_pretty_print: 	The variables' pretty print name.
	:return: 					Nothing.
	:Note: For the discretized features there is a hr.discretize in objects. For now only last_eval (but who knows)
	"""
	x_values = hr.discrete[var_x]
	y_values = hr.discrete[var_y]

	figure, axes = pp.subplots()
	axes.scatter(x_values, y_values)
	axes.set_xlabel(str(vars_pretty_print[0]))
	axes.set_ylabel(str(vars_pretty_print[1]))
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


def parse_arguments():
	arguments = {}
	parser = OptionParser()
	parser.add_option("-d", "--distribution", dest="distributions",
					  help="List distributions to plot, comma separated", metavar="DISTRIBUTIONS")
	parser.add_option("--correlation", action="store_true", dest="correlation")

	
	(options, args) = parser.parse_args()
	arguments["draw_correlation"] = options.correlation

	if not (options.distributions is None):
		arguments["distributions"] = options.distributions.split(",")
	else:
		arguments["distributions"] = list()

	return arguments


def data_analysis():
	parsed_arguments = parse_arguments()
	distributions = parsed_arguments["distributions"]

	if distributions.count("all") > 0:
		draw_distributions(hr, continuous_labels, pretty_prints[:2])
		draw_discrete_distributions(hr, discrete_labels, pretty_prints[2:5])
		draw_categorical_distributions(hr, categorical_labels, categorical_labels_pretty_prints)
	else:
		discrete_vars = set(distributions).intersection(discrete_labels)
		continuous_vars = set(distributions).intersection(continuous_labels)
		categorical_vars = set(distributions).intersection(categorical_labels)
		for var in discrete_vars:
			var_pretty_print = labels_pretty_print[var]
			draw_discrete_distribution(hr, var, var_pretty_print)
		for var in continuous_vars:
			var_pretty_print = labels_pretty_print[var]
			draw_distribution(hr, var, var_pretty_print)
		for var in categorical_vars:
			var_pretty_prints = categorical_labels_pretty_prints[var]
			title = labels_pretty_print[var]
			draw_categorical_distribution(hr, var, var_pretty_prints, title)

	if parsed_arguments["draw_correlation"]:
		#department_salary_left_correlation = department_salary_left_correlation(hr)
		draw_correlation_matrix(hr.normal[correlated_labels].corr(), pretty_prints[:5] + [pretty_prints[-1]], normalised=True)

	promotions_per_project(hr)
	salary_per_department(hr, left=True)


# Main
if __name__ == '__main__':
	hr = HR(data, normalised=True)

	data_analysis()
	pp.show()