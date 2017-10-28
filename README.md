# Data minings project
Data analysis and data mining project on the [Data mining course](http://didawiki.cli.di.unipi.it/doku.php/dm/start) at [unipi](https://www.di.unipi.it/en/).

## Dataset
The dataset is publicly available at [kaggle](https://www.kaggle.com/quentinvincenot/human-resources-analysis/data).

## Usage
```bash
$ python hr.py -h
Usage: hr.py [options]

Options:
  -h, --help            show this help message and exit
  -d DISTRIBUTIONS, --distribution=DISTRIBUTIONS
                        List distributions to plot, comma separated
  --correlation         Show correlation matrix
  --left-per-department
                        Show expected and actual left rate
  --salary-per-department
                        Show salary per department, account for left employees
  --promotions-per-project
                        Show promotions rate per number of project
  --scatter-plots       Plot scatter plots

```
use the `-h` option to get the available options.
Ex.
```
$ python hr.py -d all
$ python hr.py -d satisfaction_level,last_evaluation
$ python hr.py --correlation
```

### Save to file
The script automatically saves the plotted graphs in the working directory with a `.png` extension.

### Teamwork
In order to collaborate create a branch with your name. In order to add a feature/plot relevant to one of the project section either add it to the existing file or create one of your own, if it does not exist. For instance, plots and features relevant to the data analysis section go to `analysis.py`, future clustering functions will go to `clustering.py` and so on.