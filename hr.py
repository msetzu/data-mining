from analysis import *
from clustering import *
from settings import *
from objects import HR


# Main
if __name__ == '__main__':
    hr = HR(data)

    data_analysis(hr)
    cluster(hr)
