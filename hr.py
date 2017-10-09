import pandas as pd, numpy as np, colormap
import matplotlib as mpl, matplotlib.pyplot as pp


def init():
	satisfaction_level = data["satisfaction_level"]
	last_evaluation = data["last_evaluation"]
	number_project = data["number_project"]
	average_montly_hours = data["average_montly_hours"]
	time_spend_company = data["time_spend_company"]
	Work_accident = data["Work_accident"]
	left = data["left"]
	promotion_last_5years = data["promotion_last_5years"]
	sales = data["sales"]
	salary = data["salary"]

	satisfaction_level_std = satisfaction_level.std()
	last_evaluation_std = last_evaluation.std()
	number_project_std = number_project.std()
	average_montly_hours_std = average_montly_hours.std()
	time_spend_company_std = time_spend_company.std()
	Work_accident_std = Work_accident.std()
	left_std = left.std()
	promotion_last_5years_std = promotion_last_5years.std()

	satisfaction_level_mean = satisfaction_level.mean()
	last_evaluation_mean = last_evaluation.mean()
	number_project_mean = number_project.mean()
	average_montly_hours_mean = average_montly_hours.mean()
	time_spend_company_mean = time_spend_company.mean()
	Work_accident_mean = Work_accident.mean()
	left_mean = left.mean()
	promotion_last_5years_mean = promotion_last_5years.mean()


def draw_correlation_matrix(correlation_matrix):
	figure, axes = pp.subplots(1,1)
	figure.suptitle('Correlation matrix', fontsize=12, fontweight='bold')
	
	image = axes.imshow(correlation_matrix, cmap="Blues", vmin=0, vmax=0.5)
	pp.colorbar(image)

	axes.set_xticks(np.arange(0,correlation_matrix.shape[0], correlation_matrix.shape[0]*1.0/len(labels)))
	axes.set_yticks(np.arange(0,correlation_matrix.shape[1], correlation_matrix.shape[1]*1.0/len(labels)))

	# Add labels
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
	labels = ['satisfaction','last evaluation','projects','hours (monthly)','employment time','accidents','left','recent promotions','sales','salary']

	# Correlation matrix
	draw_correlation_matrix(np.tril(data.corr()))
	pp.show()
