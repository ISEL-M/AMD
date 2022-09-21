# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:37:00 2021

@author: Mihail Ababii
"""

from pandas import DataFrame
from numpy import array, set_printoptions
import Orange as DM

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, cohen_kappa_score

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
import joblib
import pickle

from u01_util import my_print


from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def split_dataset_Xy(D):
    # all rows and all columns except last column
    X = D.values[:, 0:-1]

    # all rows and just the last column
    y = D.values[:, -1]
    return (X, y)


def show_data(data):
    firstRows = data[0:len(data)]
    set_printoptions(precision=3)
    print(">> summarized data (max = {n:d} instances)".format(n=len(data)),
          firstRows, sep="\n")


def load(fileName):
    try:
        dataset = DM.data.Table(fileName)
    except:
        my_print("--->>> error - can not open the file: %s" % fileName)
        exit()
    return dataset


def show_score(score_all):
    if not score_all:
        return
    print("::all-evaluated-datasets::")

    all_evaluated_datasets = [i*100.0 for i in score_all]
    for v in all_evaluated_datasets:
        print(" %.2f%% " % (v), end="|")
    print()

    if isinstance(score_all, list):
        score_all = array(score_all)
    print("%.2f%% (+/- %.2f%%)" %
          (score_all.mean()*100.0, score_all.std()*100.0))


def score_recipe(model_name, classifier, X, y, f_score, **keyword_args_score):
    score_all_list = list()

    tt_split_indexes = StratifiedShuffleSplit(test_size=.2)
    for (train_index, test_index) in tt_split_indexes.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_test,  y_test = X[test_index],  y[test_index]

        # fit (build) model using classifier, X_train and y_train (training dataset)
        classifier.fit(X_train, y_train)
        # predict using the model and X_test (testing dataset)
        y_predict = classifier.predict(X_test)  

        # score the model using y_test (expected) and y_predict (predicted by model)
        # <your-code-here>
        score = f_score(y_test, y_predict, **keyword_args_score)

        score_all_list.append(score)

    joblib.dump(classifier, '{}.joblib'.format(model_name))
    pickle.dump(classifier, open('{}.pkcls'.format(model_name), 'wb'))
    return score_all_list


list_func_classifier = \
    [
        ("id3", DecisionTreeClassifier, ()),
        ("naive bayes", GaussianNB, ()),
    ]


list_score_metric = \
    [
        (accuracy_score, {}),
        (precision_score, {"average": "weighted"}),  # macro #micro #weighted
        (recall_score, {"average": "weighted"}),  # macro #micro #weighted
        (f1_score, {"average": "weighted"}),  # macro #micro #weighted
        (cohen_kappa_score, {}),  # macro #micro #weighted
    ]


def deploy(dataset):
    
    print("--------------------------------------------------")
    print("--------------------------------------------------")
    print("--------------------------------------------------")

    classifiers = {}
    for (model_name, f_classifier, args_classifier) in list_func_classifier:
        classifiers[model_name] = joblib.load('{}.joblib'.format(model_name))

    
    while True:
        classifier = classifiers["id3"]
        for model_name in classifiers.keys():
            print(model_name)
        
        tree.plot_tree(classifier,feature_names=dataset.domain.attributes, class_names=dataset.domain.class_var._values)

        
        selection = input("Select Classifier ('exit' to end) => ")
        if selection in classifiers.keys():
            classifier = classifiers[selection]
        if selection == 'EXIT':
            return
        else:
            print("Invalid Argument! Using default (ID)")
            print("-----     -----     -----     -----     -----     -----")
            
        new_tuple = []
        for variable in dataset.domain.attributes:
            print(variable.name)
            idx=0
            for value in variable.values:
                print(idx, "-", value)
                idx+=1
            
            selection = input("Select {} ('exit' to end) => ".format(variable.name))
            selection = selection.upper()
            if selection == 'EXIT':
                return
            new_tuple.append(selection)
            print("-----     -----     -----     -----     -----     -----")
            
        domain=DM.data.Domain(dataset.domain.attributes, source=dataset.domain)
        X_to_predict = DM.data.Table.from_numpy(domain, [new_tuple])
        
        predict = classifier.predict(X_to_predict)
        
        domain=DM.data.Domain(dataset.domain.class_vars, source=dataset.domain)
        result = DM.data.Table.from_numpy(domain, [predict])
        print("Result: ", result[0][0])
        print("-----     -----     -----     -----     -----     -----")
        


def main():
    fileName = "../dataset/export1.csv"
    D0 = load(fileName)
    show_data(D0)
    D = DataFrame(D0)
    (X, y) = split_dataset_Xy(D)

    for (model_name, f_classifier, args_classifier) in list_func_classifier:
        print('\n..VVV-----' + model_name + '-----VVV..')
        classifier = f_classifier(*args_classifier)
        for (f_score, keyword_args_score) in list_score_metric:
            print('::' + f_score.__name__ + '::')
            score_all = score_recipe(
                model_name, classifier, X, y, f_score, **keyword_args_score)
            show_score(score_all)

    deploy(D0)
    

if __name__ == "__main__":
    main()
