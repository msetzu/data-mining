from optparse import OptionParser
from matplotlib import mlab, patches as mpatches
import numpy as np
import matplotlib.pyplot as pp
from collections import defaultdict
import scipy.stats as stats
from scipy.stats.stats import pearsonr

def tra2rel(fileinput, fileoutput, delimiter=',', has_header=True):
    data = open(fileinput, 'r')
    if has_header:
        data.readline()
        baskets = defaultdict(list)
        for row in data:
            basket_id = row.replace('\\r\\n', '').split(delimiter)[0]
            item_id = row.replace('\\r\\n', '').split(delimiter)[1]
            baskets[basket_id].append(item_id)
        data.close()
        out = open(fileoutput, 'w')
        for k, v in baskets.items():
            s = '%s' % k
            for item in v:
                s += ',%s' % item
            out.write('%s\\n' % s)
        out.close()
    return baskets

if __name__ == '__main__':
    baskets = tra2rel('hr.csv', 'basket.csv', delimiter=',', has_header=True)
    print(baskets)