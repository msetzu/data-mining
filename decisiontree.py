import math
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats.stats import pearsonr
from sklearn import tree
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.tree import export_graphviz
import pydotplus 
from IPython.display import Image  
from IPython.display import display
import pydot
from graphviz import Source


def clean_data(hr):
    #solo rappresentazioni numeriche
    hr = hr.replace(["low", "medium", "high"] , [10000, 25000, 50000])
    #satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,left,promotion_last_5years,sales,salary
    hr["satisfaction_level"].astype(float)
    hr["last_evaluation"].astype(float)
    hr["number_project"].astype(int)
    hr["average_montly_hours"].astype(int)
    hr["time_spend_company"].astype(int)
    hr["Work_accident"].astype(int)
    hr["salary"].astype(int)
    hr["promotion_last_5years"].astype(int)
    print(hr.dtypes)
    return hr

def calculate_decision_tree(hr):
    hr_data = hr.values
    print(hr_data[1])
    hr_features = np.delete(hr_data, 6, axis=1)
    print(hr_features[1])
    hr_features = np.delete(hr_features, 7, axis=1)
    print(hr_features[1])
    hr_target = hr_data[:, 6]
    print(hr_target[1])
    '''
    clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=2)
    hr_target = hr_target.astype('int')
    clf = clf.fit(hr_features, hr_target)
    print(clf.feature_importances_)

    pred_target = clf.predict(hr_features)
    print(metrics.precision_score(hr_target, pred_target, average='weighted'))
    print(metrics.recall_score(hr_target, pred_target, average='weighted'))
    print(metrics.f1_score(hr_target, pred_target, average='weighted'))
    print(metrics.accuracy_score(hr_target, pred_target))
    print(metrics.precision_recall_fscore_support(hr_target, pred_target))
    '''
    train_x, test_x, train_y, test_y = train_test_split(hr_features, hr_target, test_size=0.20, random_state=0)
    print(len(train_x))
    print(len(test_x)) 
    print(len(train_x) + len(test_x))
    clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best', 
                                  max_depth=2, 
                                  min_samples_split=2, min_samples_leaf=10)
    train_y = train_y.astype(int)   
    test_y = test_y.astype(int) 
    print(train_x)                        
    clf = clf.fit(train_x, train_y)
    #train_pred = clf.predict(train_x)
    test_pred = clf.predict(test_x)
    #print(metrics.accuracy_score(train_y, train_pred))
    print(metrics.accuracy_score(test_y, test_pred))

    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"], class_names=['Not Left', 'Left'],  filled=True, rounded=True, special_characters=True)  
    
    graph = pydotplus.graph_from_dot_data(dot_data)  
    graph.write_png("prova.png")
    Image(graph.create_png())
    '''
    graph = Source(pydot.graph_from_dot_data(dot_data)) 
    graph.format = 'png'
    graph.render('dtree_render',view=True)
    '''


if __name__ == '__main__':
    hr = clean_data(pd.read_csv("hr.csv"))
    calculate_decision_tree(hr)
