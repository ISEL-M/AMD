# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 16:51:55 2021

@author: Mihail Ababii
"""
import csv

with open('dataset_long_name_ORIGINAL.csv', newline='') as f_input, open('dataset_long_name_ORIGINAL.tab', 'w', newline='') as f_output:
    r = list(csv.reader(f_input, delimiter=','))
    w = csv.writer(f_output, delimiter='\t')
    reader = list(r)
    
    domain = ["discrete"] * len(reader[0])
    reader.insert(1, domain)
    
    classe = [""] * len(reader[0])
    classe[reader[0].index("class")] = "class"
    reader.insert(2, classe)

    w.writerows(reader)
    
    
    
    
    