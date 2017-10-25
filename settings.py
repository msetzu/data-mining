import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Dataset
data = pd.read_csv("./hr.csv")
entries = len(data)

# Data analysis
analysis = {
    'bins': 10,
    'balance_threshold': 0.1
}

# Plot labels
labels = ['satisfaction_level',
          'average_montly_hours',
          'last_evaluation',
          'time_spend_company',
          'number_project',
          'Work_accident',
          'left',
          'promotion_last_5years',
          'sales',
          'salary']
pretty_prints = ['Self-reported satisfaction',
                 'AVG Monthly hours',
                 'Time since last valuation, in years',
                 'Time in company, in years',
                 'Projects',
                 'Accidents',
                 'Left',
                 'Promoted (last 5 years)',
                 'Department',
                 'Salary']
departments_pretty_prints = ["Information Technology",
                             "R&D",
                             "Accounting",
                             "Human Resources",
                             "Management",
                             "Marketing",
                             "Product Management",
                             "Sales",
                             "Support",
                             "Technical"]
labels_pretty_print = {k: v for k, v in zip(labels, pretty_prints)}

continuous_labels = labels[0:2]
discrete_labels = labels[2:5]
categorical_labels = labels[5:]
correlated_labels = continuous_labels + discrete_labels + ["salary_int"]
categorical_labels_pretty_prints = {
    "Work_accident": ("Injured", "Safe"),
    "left": ("Stayed", "Left"),
    "promotion_last_5years": ("Promoted", "Not promoted")
}
departments = set(data["sales"])

# Scatter plot
scatter = {
    'sampling_size': 10,   # size of each sample
    'samples': 5,   "   "   # number of samples to extract
    'edge_bins': 1, "   " # edge bins possibly containing outliers
    'bins': 10,
    'replace': True
}


# Graphs
palette = {
    'main': '#FE4365',
    'complementary': '#FC9D9A',
    'pr_complementary': '#F9CDAD',
    'sc_complementary': '#C8C8A9',
    'secondary': '#83AF9B'
}

large_palette = {
    "navy": "#001f3f",
    "blue": "#0074D9",
    "olive": "#3D9970",
    "orange": "#FF851B",
    "green": "#2ECC40",
    "yellow": "#FFDC00",
    "red": "#FF4136",
    "maroon": "#85144b",
    "black": "#111111",
    "grey": "#AAAAAA"
}

cmap_pale_pink = LinearSegmentedColormap.from_list('Pale pink',
                                                   [palette['pr_complementary'], palette['main']],
                                                   N=1000000)
cmap_pale_pink_and_green = LinearSegmentedColormap.from_list('Pale pink&green',
                                                            [palette['main'],
                                                             palette['complementary'],
                                                             palette['pr_complementary'],
                                                             palette['sc_complementary'],
                                                             palette['secondary']],
                                                             N=1000000)