from matplotlib.colors import LinearSegmentedColormap

# Plotting settings

# Scatter plot
scatter = {
    'sampling_size': 10,   # size of each sample
    'samples': 5,           # number of samples to extract
    'edge_bins': 1,         # edge bins possibly containing outliers
    'bins': 10,
    'replace': True
}


# Graphs
colors = {
	'main': '#FE4365',
    'complementary': '#FC9D9A',
    'pr_complementary': '#F9CDAD',
    'sc_complementary': '#C8C8A9',
	'secondary': '#83AF9B'
}

cmap_pale_pink = LinearSegmentedColormap.from_list('Pale pink', [colors['pr_complementary'], colors['main']], N=1000000)
cmap_pale_pink_and_green = LinearSegmentedColormap.from_list('Pale pink&green', [colors['main'], colors['complementary'], colors['pr_complementary'], colors['sc_complementary'], colors['secondary']], N=1000000)