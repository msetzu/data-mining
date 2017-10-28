import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Dataset
data = pd.read_csv("./hr.csv")
entries = len(data)
bins = 10

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
labels_pretty_print["salary_int"] = "Salary"

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
    "sampling_size": 100,    # size of each sample
    "samples": 5,           # number of samples to extract
    "edge_bins": 1,         # edge bins possibly containing outliers
    "bins": 10,
    "replace": True
}


# Graphs
palette = {
    'main': '#FE4365',
    'complementary': '#FC9D9A',
    'pr_complementary': '#F9CDAD',
    'sc_complementary': '#C8C8A9',
    'secondary': '#83AF9B'
}

round_palette = {
    "main": palette["secondary"],
    "secondary": palette["complementary"],
    "pr_complementary": palette["sc_complementary"],
    "sc_complementary": palette["secondary"]
}

large_palette = {
    "navy": "#001f3f",
    "blue": "#0074D9",
    "green": "#2ECC40",
    "olive": "#3D9970",
    "orange": "#FF851B",
    "yellow": "#FFDC00",
    "red": "#FF4136",
    "maroon": "#85144b",
    "black": "#111111",
    "grey": "#AAAAAA"
}

large_palette_full = {
    "navy": "#001f3f",
    "blue": "#0074D9",
    "aqua": "#7FDBFF",
    "teal": "#39CCCC",
    "olive": "#3D9970",
    "green": "#2ECC40",
    "lime": "#01FF70",
    "yellow": "#FFDC00",
    "orange": "#FF851B",
    "red": "#FF4136",
    "maroon": "#85144b",
    "fuchsia": "#F012BE",
    "purple": "#B10DC9",
    "black": "#111111",
    "grey": "#AAAAAA",
    "silver": "#DDDDDD"
}

large_palette_stacked = {
    "navy": "#001f3f",
    "blue": "#0074D9",
    "olive": "#3D9970",
    "orange": "#FF851B",
    "green": "#2ECC40",
    "yellow": "#FFDC00",
    "red": "#FF4136",
    "maroon": "#85144b",
    "black": "#111111",
    "grey": "#AAAAAA",
    "stack": large_palette["orange"]
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
