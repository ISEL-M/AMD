# -*- coding: cp1252 -*-
# ===============================
# author: Paulo Trigo Silva (PTS)
# version: v01
# ===============================


# ______________________________________________________________________________
# The modules to import
from math import sqrt
from random import randrange
from random import seed
import csv
import matplotlib.pyplot as plt


# ______________________________________________________________________________
# The base concepts:
# - similarity-criteria
# - attraction/repulsion rules
# - learning-rate

# similarity-criteria
# - computed as the Euclidean distance between two vectors (v1 and v2)
def get_euclidean_distance(v1, v2):
    distance = 0.0
    # the feature list excludes last component because it is the "class"
    lenFeature = len(v1) - 1
    for i in range(lenFeature):
        distance+=(v2[i]-v1[i])**2
    distance=sqrt(distance)
    return distance


# attraction/repulsion rules
def apply_attraction_rule(learningRate, trainingInstance, BMU):
    return attraction_repulsion_rule(learningRate, trainingInstance, BMU,
                                     lambda a, b: a + b)


def apply_repulsion_rule(learningRate, trainingInstance, BMU):
    # <HERE> correct the next "return" to properly implement the repulsion_rule
    # hint: follow the same reasoning as in the "apply_attraction_rule"
    return attraction_repulsion_rule(learningRate, trainingInstance, BMU,
                                     lambda a, b: a - b)


def attraction_repulsion_rule(learningRate, trainingInstance, BMU,
                              operator):
    sumError = 0.0
    # the feature list excludes last component because it is the "class"
    lenFeature = len(trainingInstance) - 1
    for i in range(lenFeature):
        error = trainingInstance[i] - BMU[i]
        sumError += error**2
        BMU[i] = operator(BMU[i], learningRate * error)
    return (BMU, sumError)


## learning-rate (non-linear)
def get_learning_rate(alpha, epoch, maxEpoch):
    # return alpha
    # <HERE> return the time-dependent learning rate (cf., slides)
    return alpha * ( 1.0 -(epoch/maxEpoch) )


# ______________________________________________________________________________
# The algorithm specific methods
# ______________________________________________________________________________

# get the BMU - Best Matching Unit
def get_BMU(codebook, trainingInstance):
    # set BMU to None and minimum distance to Infnity
    (BMU, minDistance) = (None, float("inf"))
    for cBv in codebook:
        distance = get_euclidean_distance(cBv, trainingInstance)
        if(distance <= minDistance):
            (BMU, minDistance) = (cBv, distance)
    return BMU


# train a codebook, given:
# - a dataset, an initial codebook,
# - the alpha initial-learning-rate and the maximum number of epochs
def train_LVQ(dataset, codebook, alpha, maxEpoch):
    for epoch in range(maxEpoch):
        lRate = get_learning_rate(alpha, epoch, maxEpoch)
        totalError = 0.0
        for trainingInstance in dataset:
            BMU = get_BMU(codebook, trainingInstance)
            if BMU[-1] == trainingInstance[-1]:
                (BMU, error) = apply_attraction_rule(
                    lRate, trainingInstance, BMU)
            else:
                (BMU, error) = apply_repulsion_rule(
                    lRate, trainingInstance, BMU)
            totalError += error

    print("> Epoch=%d, learningRate=%.3f, totalError=%.3f" %
          (epoch, lRate, totalError))
    return (codebook, totalError)


# ______________________________________________________________________________
# Build a random codebook vector (taken from the dataset)
def build_codebook_random(dataset):
    lenDataset = len(dataset)
    lenFeature = len(dataset[0])
    codebook = [dataset[randrange(lenDataset)][i] for i in range(lenFeature)]
    return codebook


# ______________________________________________________________________________
# Auxiliary Functions

# read csv file
def read_csv(fileName, delimiter=';'):
    with open(fileName) as f_in:
        reader = csv.reader(f_in, delimiter=delimiter)
        header = next(reader)
        dataset = list()
        for row in reader:
            # ignore empty lines
            if not row:
                continue

            row_list = list()
            for col in range(len(row)-1):
                # all features are of continuous type (cast to float)
                row_list.append(float(row[col]))
            # the last column is the class of discrete type (cast to string)
            row_list.append(str(row[len(row)-1]).strip())
            dataset.append(row_list)
    return (header, dataset)


# print info
def print_info(description, aHeader, aList, aStr=None, aFloat=None):
    print()
    print(description)
    print(aHeader)
    for e in aList:
        print(e)
    if(aStr):
        print(aStr, end="")
    if(aFloat):
        print("{0: .3f}".format(aFloat), end="\n"*2)


# ______________________________________________________________________________
# The Learning Parameters
# ______________________________________________________________________________
seed(1)

_alpha = 0.7  # 0.3
_maxEpoch = 1  # 100
_codebookRandom =  True
# next variable only used for the random codebook (if _codebookRandom = True):
_dimCodebook = 5

# ______________________________________________________________________________


# ______________________________________________________________________________
# The Dataset and Codebook files in CSV Format
# ______________________________________________________________________________
_datasetFileName_csv = "01_dataset_slides.txt"
_codebookFileName_csv = "02_codebook_slides.txt"


# ______________________________________________________________________________
# The "main" function
def main():
    (data_header, dataset) = read_csv(_datasetFileName_csv)
    print_info(">> dataset:", data_header, dataset)

    (cbook_header, codebook) = (list(), list())
    if(_codebookRandom):
        codebook = [build_codebook_random(dataset)
                    for i in range(_dimCodebook)]
    else:
        (cbook_header, codebook) = read_csv(_codebookFileName_csv)
    print_info(">> initial codebook:", cbook_header, codebook)

    # learn a codebook (given an initial one)
    errorList = list()
    for maxEpoch in range(1,100):
        (codebook, totalError) = train_LVQ(dataset, codebook, _alpha, maxEpoch)
        errorList.append(totalError)
        
    plt.plot(errorList)
    plt.title("dim" + str(_dimCodebook))


# ______________________________________________________________________________
# The "main" of this module (is case this module is loaded from another module)
if __name__ == "__main__":
    main()
