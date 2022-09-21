# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 16:46:06 2021

@author: Mihail Ababii
"""

# This is a string object
name = 'Swaroop'
if name.startswith('Swa'):
    print('Yes, the string starts with "Swa"')
if 'a' in name:
    print('Yes, it contains the string "a"')
if name.find('war') != -1:
    print('Yes, it contains the string "war"')
delimiter = '_*_'
mylist = ['Brazil', 'Russia', 'India', 'China']
print(delimiter.join(mylist))