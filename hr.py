import pandas as pd, numpy as np, colormap
import matplotlib as mpl, matplotlib.pyplot as pp
from objects import HR

def draw_distributions(hr, var, var_pretty_print):
	""" Draw the variables' distributions. """
	var_val = hr.data[var]
	var_val_std = hr.std[var]
	var_val_mean = hr.mean[var]
	num_bins = int(np.ceil(np.log2(len(var_val))) + 1)

	figure, axes = pp.subplots()
	n, var_val_bins, patches = axes.hist(var_val, num_bins, label='Data distribution')

	axes.set_ylabel(str(var_pretty_print) + ' value')
	axes.set_title(r'Distribution: $\mu$ =' + format(var_val_std, '.4f') + ', $\sigma$ =' + format(var_val_mean, '.4f'))
	#axes.xaxis.set_visible(False)
	pp.title(str(pretty_print) + ': data distribution')


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
	data = pd.read_csv("./hr.csv")
	labels = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'left', 'promotion_last_5years', 'sales', 'salary']
	labels_pretty_print = ['Satisfaction', 'Evaluation', 'Projects', 'AVG Monthly hours', 'Time in company', 'Accident', 'Left', 'Promoted (5 years)', 'Sales', 'Salary']
	continuous_labels = labels[:-5]
	discrete_labels = labels[-5:]


	hr = HR(data)

	# Variables representation
	for var, pretty_print in zip(continuous_labels, labels_pretty_print):
		draw_distributions(hr, var, pretty_print)

	# Correlation matrix
	draw_correlation_matrix(np.tril(data.corr()))

	pp.show()
