import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

colors = {
	'main': '#FE4365',
	'secondary': '#83AF9B',
    'complementary': '#FC9D9A',
    'pr_complementary': '#F9CDAD',
    'sc_complementary': '#C8C8A9'
}

cmap_pale_pink = LinearSegmentedColormap.from_list('Pale pink', [colors['pr_complementary'], colors['main']], N=1000000)