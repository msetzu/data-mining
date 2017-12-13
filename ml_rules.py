from matplotlib import mlab, patches as mpatches
import numpy as np
import matplotlib.pyplot as pp
from scipy.interpolate import spline
import pandas as pd

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

def ml_rules(rules, data):
    '''
    analizzo la tupla in arrivo, in base al valore che ha per ogni variabile, dico left o non left
    temporary è un temporaneo che mantiene quanti match ha ottenuto per left = 0 e per left = 1
    '''
    temporary = []
    temporary.append(0)
    temporary.append(0)
    for variable in data:
        for rule in rules:
            for i in range (0,len(rule["ant"])):
                if(rule["ant"][i] == variable):
                    if(rule["cons"] == "0_Left"):
                        temporary[0] = temporary[0] + 1
                    elif(rule["cons"] == "1_Left"):
                        temporary[1] = temporary[1] + 1    
    print(temporary)

if __name__ == '__main__':
    rules = read_rules("topk_rules.txt")
    data = list()
#0.25_Satisfaction,1.0_Evaluation,7_Project,305_Hours,5_Time,0_Accident,1_Left,0_Promotion,sales
#0.75_Satisfaction,1.0_Evaluation,5_Project,200_Hours,2_Time,0_Accident,0_Left,0_Promotion,technical
#0.75_Satisfaction,0.75_Evaluation,5_Project,250_Hours,2_Time,1_Accident,0_Left,0_Promotion,product_mng
    data.append("0.25_Satisfaction")
    data.append("1.0_Evaluation")
    data.append("7_Project")
    data.append("305_Hours")
    data.append("5_Time")
    data.append("0_Accident")
    data.append("0_Promotion")
    data.append("sales")
    ml_rules(rules, data)