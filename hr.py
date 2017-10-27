from analysis import *
from settings import *
from objects import HR


# Main
if __name__ == '__main__':
	hr = HR(data, normalised=True)

	data_analysis(hr)
	pp.show()