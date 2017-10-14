# Data minings project
Data analysis and data mining project on the [Data mining course](http://didawiki.cli.di.unipi.it/doku.php/dm/start) at [unipi](https://www.di.unipi.it/en/).

## Dataset
The dataset is publicly available at [kaggle](https://www.kaggle.com/quentinvincenot/human-resources-analysis/data).

## Usage
```bash
$ python hr.py -d all|satisfaction_level|last_evaluation|number_project
|average_montly_hours|time_spend_company|Work_accident|left|promotion_last_5years|sales|salary
```
with `-d` option for the distribution(s) whose graph(s) to draw. `all` draws all the distributions, otherwise you can select just one or more than one, separated by `,` (no space between them).
Ex.
```
$ python hr.py -d all
$ python hr.py -d satisfaction_level,last_evaluation
```