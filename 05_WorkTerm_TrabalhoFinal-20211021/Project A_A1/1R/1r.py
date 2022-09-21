# to use accented characters in the code
# -*- coding: cp1252 -*-
# ===============================
# author: Paulo Trigo Silva (PTS)
# version: v07 (Python3)
# ===============================


# __________________________________________
# Orange Documentation:
# http://docs.orange.biolab.si
#
# Orange Reference Manual:
# http://docs.orange.biolab.si/3/data-mining-library/#reference
#
# Tutorial:
# http://docs.orange.biolab.si/3/data-mining-library/#tutorial
#
# details about data (attribute+class) characterization:
# http://docs.orange.biolab.si/3/data-mining-library/tutorial/data.html#data-input
# __________________________________________

# _______________________________________________________________________________
# Modules to Evaluate
from u01_util import my_print
import Orange as DM
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from numpy import set_printoptions
import os


# _______________________________________________________________________________
# Auxiliary Functions
# _______________________________________________________________________________
# load the dataset
def load(fileName):
    try:
        dataset = DM.data.Table(fileName)
    except:
        my_print("--->>> error - can not open the file: %s" % fileName)
        exit()
    return dataset


def show_data(data, numFirstRows=10):
    firstRows = data[0:numFirstRows]
    set_printoptions(precision=3)
    print(">> summarized data (max = {n:d} instances)".format(n=numFirstRows),
          firstRows, sep="\n")

# _______________________________________________________________________________
# get the variable dataset-structure given a string with its name


def get_variableFrom_str(dataset, str_name):
    variable_list = dataset.domain.variables
    for variable in variable_list:
        if(variable.name == str_name):
            return variable
    my_print(">>error>> \"{}\" is not a variable name in dataset!".format(str_name))
    return None


# _______________________________________________________________________________
# Data Analysis Functions
# _______________________________________________________________________________
# percentage of missing values of each variable
def get_missingValuePercentage(dataset):
    variable_list = dataset.domain.variables
    var_len = len(variable_list)
    missingValue_list = [0.0] * var_len
    for instance in dataset:
        for var_index in range(var_len):
            if np.isnan(instance[var_index]):
                missingValue_list[var_index] += 1
    missingValue_list = list(map(lambda x, dim=len(
        dataset): x / dim*100.0, missingValue_list))
    return (variable_list, missingValue_list)


# _______________________________________________________________________________
# contingencyMatrix; i.e., the joint-frequency table
# this implementation does not account for missing-values
# (i.e., missing-values are not included in the variables-domain)
# M(row, column)
def get_contingencyMatrix(dataset, rowVar, colVar):
    if(isinstance(rowVar, str)):
        rowVar = get_variableFrom_str(dataset, rowVar)
    if(isinstance(colVar, str)):
        colVar = get_variableFrom_str(dataset, colVar)
    if(not (rowVar and colVar)):
        return ([], [], None)
    if(not (rowVar.is_discrete and colVar.is_discrete)):
        my_print(">>error>> variables are expected to be discrete")
        return ([], [], None)

    rowDomain, colDomain = rowVar.values, colVar.values
    len_rowDomain, len_colDomain = len(rowDomain), len(colDomain)
    contingencyMatrix = np.zeros((len_rowDomain, len_colDomain))
    for instance in dataset:
        rowValue, colValue = instance[rowVar], instance[colVar]
        if(np.isnan(rowValue) or np.isnan(colValue)):
            continue

        rowIndex, colIndex = rowDomain.index(
            rowValue), colDomain.index(colValue)
        contingencyMatrix[rowIndex, colIndex] += 1
    return (rowDomain, colDomain, contingencyMatrix)


# _______________________________________________________________________________
# P( H | E )
# H means Hypothesis, E means Evidence
# a frequency approach
def get_conditionalProbability(dataset, H, E):
    if(isinstance(H, str)):
        H = get_variableFrom_str(dataset, H)
    if(isinstance(E, str)):
        E = get_variableFrom_str(dataset, E)
    if(not (H and E)):
        return ([], [], None)
    (rowDomain, colDomain, cMatrix) = get_contingencyMatrix(dataset, H, E)

    len_rowDomain, len_colDomain = len(rowDomain), len(colDomain)
    E_marginal = np.zeros(len_colDomain)
    for col in range(len_colDomain):
        E_marginal[col] = sum(cMatrix[:, col])

    for row in range(len_rowDomain):
        for col in range(len_colDomain):
            cMatrix[row, col] = cMatrix[row, col] / E_marginal[col]
    return (rowDomain, colDomain, cMatrix)


# _______________________________________________________________________________
# 1R Related Functions
# _______________________________________________________________________________
# error matrix for a given feature and considering the datatset class
def get_errorMatrix(dataset, feature):
    if(isinstance(feature, str)):
        feature = get_variableFrom_str(dataset, feature)
    the_class = dataset.domain.class_var
    (rowDomain, colDomain, cMatrix) = get_conditionalProbability(
        dataset, the_class, feature)
    if(not (rowDomain or colDomain)):
        return ([], [], None)

    errorMatrix = 1 - cMatrix
    
    
    plt.clf()
    svm = sn.heatmap(data=errorMatrix, annot=True,
                     xticklabels=colDomain, yticklabels=rowDomain)
    figure = svm.get_figure()
    figure.savefig("./ErrorMatrix/{}&Lenses".format(feature), dpi=400)
    
    
    return (rowDomain, colDomain, errorMatrix)


# _______________________________________________________________________________
# Data Presentation Functions
# _______________________________________________________________________________
# show contingency matrix
def show_contingencyMatrix(dataset, rowVar, colVar):
    my_print("{} & {} :".format(rowVar.name, colVar.name))
    (rowDomain, colDomain, cMatrix) = get_contingencyMatrix(dataset, rowVar, colVar)

    plt.clf()
    svm = sn.heatmap(data=cMatrix, annot=True,
                      xticklabels=colDomain, yticklabels=rowDomain)
    figure = svm.get_figure()
    figure.savefig(
        "./ContingencyMatrix/{}&{}".format(rowVar.name, colVar.name), dpi=400)

    print(cMatrix)
    error = 0
    for row in rowDomain:

        showStr = "<" + row + "> "
        for col in colDomain:
            showStr += col + ": " + str(cMatrix[rowDomain.index(row),
                                                colDomain.index(col)]) + " | "

        totalRow = np.sum(cMatrix[rowDomain.index(row)])
        error += np.min((totalRow-cMatrix[rowDomain.index(row)]))

    total = np.sum(cMatrix)
    print("erro total: " + str(error/total))
    return error/total


# _______________________________________________________________________________
# show all contingency matrix
# - for each feature and the class
def showAll_contingencyMatrix(dataset):
    feature_list, the_class = dataset.domain.attributes, dataset.domain.class_var
    curr_feature = ""
    curr_error = 1
    for feature in feature_list:
        error = show_contingencyMatrix(dataset, feature, the_class)
        if (error < curr_error):
            curr_feature = feature.name
            curr_error = error
        print()
    return curr_feature


# _______________________________________________________________________________
# P( H | E )
# show matriz and textual description
def show_conditionalProbability(dataset, H, E):
    (rowDomain, colDomain, cMatrix) = get_conditionalProbability(dataset, H, E)
    print(cMatrix)
    print()

    for h in rowDomain:
        for e in colDomain:
            rowIndex, colIndex = rowDomain.index(h), colDomain.index(e)
            P_h_e = cMatrix[rowIndex, colIndex]
            print("  P({} | {}) = {:.3f}".format(h, e, P_h_e))


def deploy(the_feature, rules, featureDomain0):
    featureDomain = list(map(lambda feature: feature.upper(), featureDomain0))

    while True:
        print("Available Feature({}):".format(the_feature))
        for feature in featureDomain:
            print(feature)
            
        selection = input("Selected Feature ('exit' to end) => ")
        selection = selection.upper()
        if selection == 'EXIT':
            break
        
        if(selection in featureDomain):
            idx = featureDomain.index(selection)
            feature = featureDomain0[idx]
            
            print("Result: ", rules[feature])
        
            selection = input("Try again? (y/n)\n")
            selection = selection.upper()
            if selection == 'N':
                break
        else:
            selection = input("Invalid Input! Try again? (y/n)")
            selection = selection.upper()
            if selection!='Y':
                break
        
    
    
    
# _______________________________________________________________________________
# implementation of some test cases
def rule1r():
    fileName = "../dataset/export1.tab"
    fileName = "../dataset/dataset_long_name_PTS_INPUT_v01.tab"
    dataset0 = load(fileName)
    dataset0.shuffle()
    show_data(dataset0)

    cut = int(len(dataset0)*0.7)
    dataset_train = dataset0[:cut]
    dataset_test = dataset0[cut:]
    print()
    aStr = ">> Percentage of missing values per variable <<"
    my_print(aStr)
    (variable_list, missingValue_list) = get_missingValuePercentage(dataset_train)
    for i in range(len(variable_list)):
        print("%4.1f%s %s" %
              (missingValue_list[i], '%', variable_list[i].name))

    print()
    aStr = ">> Contingency Matrix <<"
    my_print(aStr)
    the_feature = showAll_contingencyMatrix(dataset_train)

    print()
    # the_feature = "prescription" #"age"
    aStr = "(1R-approach) >>Error Matrix>> %s & %s <<" % (the_feature,
                                                          dataset_train.domain.class_var)
    my_print(aStr)
    (classDomain, featureDomain, errorMatrix) = get_errorMatrix(
        dataset_train, the_feature)
    if(not (classDomain or featureDomain)):
        return

    print(classDomain)
    print(featureDomain)
    print(errorMatrix)
    print()
    print("-->> so, the rule, and error, for the {} feature are:".format(the_feature))

    rules = {}
    f = open("oneR_OUTPUT.txt", "w")
    (rowDomain, colDomain, cMatrix) = get_contingencyMatrix(dataset0, dataset_train.domain.class_var, the_feature)
    for feature in range(len(featureDomain)):
        errorFeature = errorMatrix[:, feature]
        errorMin = min(errorFeature)
        errorMinIndex = errorFeature.tolist().index(errorMin)
        featureValue = featureDomain[feature]
        classValue = classDomain[errorMinIndex]

        showStr = "(" + the_feature + ", " + \
            featureValue + ", " + classValue + ") : "
        print(showStr + "{:.3f}".format(errorMin))
        
        rules[featureValue] = classValue
        
        
        total = sum(cMatrix[:,feature])
        errorMin = min(cMatrix[:,feature])
            
        
        to_save = "({}, {}, {}) : ({}, {})\n".format(the_feature, featureValue, classValue, int(errorMin), int(total))
        f.writelines(to_save)

    f.close()

    print("--------------------------------------------------")
    print("--------------------------------------------------")
    print("--------------------------------------------------")
    

    # for a in dataset0:
    #     print("-----------------")
    #     print(a)
    #     value = a[the_feature].value
    #     result = rules[value]
    #     print("result:")
    #     print(result)
    
    print(the_feature," rules:",rules)
    
    print("--------------------------------------------------")
    print("--------------------------------------------------")
    print("--------------------------------------------------")
    
    deploy(the_feature, rules, featureDomain)


# _______________________________________________________________________________
# the main of this module (in case this module is imported from another module)
if __name__ == "__main__":
    rule1r()
