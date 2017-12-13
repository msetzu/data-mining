from optparse import OptionParser
from matplotlib import mlab, patches as mpatches
import numpy as np
import matplotlib.pyplot as pp
from collections import defaultdict
import scipy.stats as stats
import pandas as pd
from scipy.stats.stats import pearsonr
import fim
from fim import apriori
import subprocess

def clean_data(hr):
    #sistemiamo satisfaction level con solo 3 valori
    clusters = [0.25, 0.55, 1]
    for i in hr['satisfaction_level']:
        if(i <= 0.25):
            hr = hr.replace(i, 0.25)
        elif(i<= 0.50 and i > 0.25):
            hr = hr.replace(i, 0.50)
        elif(i > 0.50 and i <= 0.75):
            hr = hr.replace(i, 0.75)  
        elif(i > 0.75):
            hr = hr.replace(i, 1)    
    for i in hr['number_project']:
        if(i <= 2.5):
            hr = hr.replace(i, 2)
        elif(i<=5 and i > 2):
            hr = hr.replace(i, 5)
        elif(i<=7.5 and i>5):
            hr = hr.replace(i, 7) 
        elif(i>=7 and i<100):
            hr = hr.replace(i, 10)      
    for i in hr['average_montly_hours']:
        if(i <= 150):
            hr = hr.replace(i, 150)
        elif(i<=200 and i > 150):
            hr = hr.replace(i, 200)
        elif(i <= 250 and i > 200):
            hr = hr.replace(i, 250)   
        else:
            hr.replace(i, 300)     
    hr1 = hr        
    hr1['satisfaction_level'] = hr['satisfaction_level'].astype(str) + '_Satisfaction'         
    hr1['last_evaluation'] = hr['last_evaluation'].astype(str) + '_Evaluation'  
    hr1['number_project'] = hr['number_project'].astype(str) + '_Project' 
    hr1['average_montly_hours'] = hr['average_montly_hours'].astype(str) + '_Hours'  
    hr1['time_spend_company'] = hr['time_spend_company'].astype(str) + '_Time'
    hr1['Work_accident'] = hr['Work_accident'].astype(str) + '_Accident' 
    hr1['left'] = hr['left'].astype(str) + '_Left'  
    hr1['promotion_last_5years'] = hr['promotion_last_5years'].astype(str) + '_Promotion'      

    hr1.to_csv("input_corretto.csv")    
    
def tra2rel(fileinput, fileoutput, delimiter=',', has_header=True):
    data = open(fileinput, 'r')
    if has_header:
        data.readline()
    baskets = defaultdict(list)

    for row in data:
        basket_id = row.replace('\r\n', '').split(delimiter)[0]
        for item in range(1, 10):
            item_id = row.replace('\r\n', '').split(delimiter)[item]
            baskets[basket_id].append(item_id)

    data.close()

    out = open(fileoutput, 'w')
    for k, v in baskets.items():
        s = '%s' % k
        for item in v:
            s += ',%s' % item
        out.write('%s\n' % s)
    out.close()
    
    return baskets


def call_apriori(fileinput, fileoutput, delimiter=',', target_type='s', 
                 min_nbr_items=1, min_sup=2, min_conf=2):
    # apriori
    # -t# {m: maximal, c: closed, s: frequent, r: association rules}
    # -m# minimum number of items per item set/association rule
    # -s# minimum support of an item set, positive: percentage, negative: absolute
    # -c# minimum confidence rule percentage
    # -b# line delimiter (,)
    # The default additional information output format for rules is " (%X, %C)"
    # %X relative body set support as a percentage
    # %C rule confidence as a percentage
    # %L lift

    if target_type == 'r':
        call_cmd = ['./apriori/apriori', '-b%s' % delimiter, '-t%s' % target_type, '-m%s' % min_nbr_items, 
                    '-s%s' % min_sup, '-c%s' % min_conf, '-v (%X, %C, %l)', 
                    fileinput, fileoutput]
    else:
        call_cmd = ['./apriori/apriori', '-b%s' % delimiter, '-t%s' % target_type, 
                           '-m%s' % min_nbr_items, '-s%s' % min_sup, fileinput, fileoutput]

    ret = subprocess.call(call_cmd,  stdout=open('apriori_stdout.txt', 'w'), 
                          stderr=open('apriori_stderr.txt', 'w'))
    return ret

def apriori_function(baskets):
    baskets_lists = [b for b in baskets.values()]
    itemsets = apriori(baskets_lists, supp=20, zmin=2, target='r', conf=90, report='ascl')  
    thefile = open('rules.txt', 'w')   
    for item in itemsets:
        thefile.write(str(item)+'\n\r')

def read_rules(filename):
    data = open(filename, 'r')
    rules = list()
    for row in data:
        fileds = row.rstrip('\n\r').split(' <- ')
        cons = fileds[0]
        other = fileds[1].split(' (')
        ant = other[0].split(' ')
        other2 = other[1].split(', ')
        sup = float(other2[0])
        conf = float(other2[1])
        lift = float(other2[2].replace(')', ''))
        rule = {
            'ant': ant,
            'cons': cons,
            'sup': sup,
            'conf': conf,
            'lift': lift
        }
        rules.append(rule)
    data.close()
    return rules        
     
def select_only_left(rules):
    rules0 = list()
    rules1 = list()
    file0 = open("rules_0_left", "w")
    file1 = open("rules_1_left", "w")
    for rule in rules:
        if(rule['cons'] == "0_Left"):
            rules0.append(rule)
        elif(rule['cons'] == "1_Left"):   
            rules1.append(rule)  
    rules0ord =  sorted(rules0, key=getKey) 
    rules1ord = sorted(rules1, key=getKey)           
    for r in rules0ord:
        file0.write(str(r['ant'])+' --> '+str(r['cons'])+' lift '+str(r['lift'])+' conf '+str(r['conf'])+'\r\n') 
    for r in rules1ord:
        file1.write(str(r['ant'])+' --> '+str(r['cons'])+' lift '+str(r['lift'])+' conf '+str(r['conf'])+'\r\n')                  

def getKey(item):
        return item["conf"]

if __name__ == '__main__':
    rules = read_rules("rules_5_70.txt")
    select_only_left(rules)
    '''
    baskets = tra2rel("input_corretto.csv", 'baskets2.csv', delimiter=',', has_header=True)
    delimiter=','
    target_type='r'
    min_nbr_items=2
    min_sup=5
    #ret_val = call_apriori('baskets2.csv', 'freq_patterns_2_5.txt', delimiter, target_type, min_nbr_items, min_sup)
    #hr = pd.read_csv("hr.csv")
    #num_bins1 = int(np.ceil(np.log2(len(hr['satisfaction_level']))) + 1)
    #num_bins2 = int(np.ceil(np.log2(len(hr['last_evaluation']))) + 1)
    #num_bins3 = int(np.ceil(np.log2(len(hr['time_spend_company']))) + 1)
    #num_bins4 = int(np.ceil(np.log2(len(hr['number_project']))) + 1)
    #clean_data(pd.read_csv("hr.csv"))
    min_conf=80

    ret_val = call_apriori('baskets2.csv', 'rules_5_80.txt', delimiter, target_type, min_nbr_items, min_sup, min_conf)
    rules = read_rules('rules_5_80.txt')
    file = open("rules_pretty_5_80.txt", "w")
    for r in rules:
        file.write(str(r['ant'])+' --> '+str(r['cons'])+' lift '+str(r['lift'])+' conf '+str(r['conf'])+'\r\n')                    
    '''