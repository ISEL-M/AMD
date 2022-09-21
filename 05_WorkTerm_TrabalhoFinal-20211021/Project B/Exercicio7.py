# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 16:44:10 2022

@author: Mihail Ababii
"""

import Orange
from orangecontrib.associate.fpgrowth import *
import sys

def load_encode_file(): 
    data = Orange.data.Table("../Dataset/zz_dataset_2012_01.basket")
    X, mapping = OneHot.encode(data)
    return data, X, mapping


def getRules(X, support, confidence):
    setOf_instances = frequent_itemsets(X, support)
    dict_setOf_instances = dict(setOf_instances)
    setOf_rule = association_rules(dict_setOf_instances, confidence)
    return list(setOf_rule)
   


if __name__ == "__main__":    
    # sys..call('C:\Users\Mihail Ababii\Desktop\Universidade\AMD\05_WorkTerm_TrabalhoFinal-20211021\Project B\scripts 3-6\4_Exercicio4_subset.bat'])
    
    data, X, mapping = load_encode_file()
    support = 0.01
    confidence = 1
    maxR=10
    
    rules = getRules(X, support, confidence)
    print("number of rules for support:", support," e confidence:", confidence, "--->",len(rules))
    for rule in rules[:10]:
        LHS, RHS, support, confidence = rule
        print([var.name for _, var, _ in OneHot.decode( LHS, data, mapping )])
    maxR = int(input("number os rules (maxR):"))
    for support in range(100, 0, -1):
        support = support * 0.01
       
        for confidence in range(100, 0, -1):
          confidence = confidence * 0.01

          #Generate maxR rules
          setOf_itemset = frequent_itemsets(X, support)
          dict_setOf_itemset = dict(setOf_itemset)
          setOf_rule = association_rules(dict_setOf_itemset, confidence)
          list_setOf_rule = list(setOf_rule)
          
          if len(list_setOf_rule) <= maxR:
              if confidence == 0.01:
                if len(list_setOf_rule) != 0:
                    print("Suporte = ", support," | Confiança = ", confidence)
                    print("Regras:")
                    for rule in list_setOf_rule:
                        LHS, RHS, support, confidence = rule
                        decoded_LHS = [ var.name for _, var, _ in OneHot.decode( LHS, data, mapping ) ]
                        decoded_RHS = [ var.name for _, var, _ in OneHot.decode( RHS, data, mapping ) ]
                        tuple_rule = ( decoded_LHS, decoded_RHS, support, confidence )
                        print( tuple_rule )
                    print()   
          