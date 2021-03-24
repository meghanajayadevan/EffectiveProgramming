#!/usr/bin/env python3

from random import choice
from numpy import array, dot, random, asfarray
import math
import pandas as pd



class TDIDTNode:
    """ This is a node for building a tree."""
    def __init__(self, parent_id=-1, left_child_id=None, right_child_id=None):
        self.parent_id = parent_id
        self.is_Left = False
        # self.direction      = direction
        self.left_child_id = left_child_id
        self.right_child_id = right_child_id
        self.is_leaf = False
        self.outcome = None
        # only needed to fullfill exercise requirements
        self.identifier = 0
        self.parent_test_outcome = None
        self.pplus = None
        self.pminus = None
        self.label = None
        self.threshold = None

    def setLeftChild(self, id):
        self.left_child_id = id

    def setRightChild(self, id):
        self.right_child_id = id

    def setpplus(self, id):
        self.pplus = id

    def setpminus(self, id):
        self.pminus = id

    def setthreshold(self, id):
        self.threshold = id

    def setlabel(self, id):
        self.label = id

    def setidentifier(self, id):
        self.identifier = id

    def setis_Left(self, id):
        self.is_Left = id

    def __str__(self):
        return "{} {} {} {} ".format(self.label, self.threshold, self.pplus, self.pminus)



def get_information_gain(ppos=335, pneg=340, npos=0, nneg=8):
    """ The function returns the information gain based on the positive and negative side split"""
    total = float(ppos + pneg + npos + nneg)
    p_total = float(ppos + pneg)
    n_total = float(npos + nneg)
    information_gain = entropy((ppos + npos) / total, (pneg + nneg) / total)
    if p_total > 0:
        information_gain -= p_total / total * entropy(ppos / p_total, pneg / p_total)
    if n_total > 0:
        information_gain -= n_total / total * entropy(npos / n_total, nneg / n_total)
    return information_gain


def entropy(p, n):
    """ This calculates the entropy"""
    if n == 0:
        return p * math.log(1.0 / p, 2)
    elif p == 0:
        return n * math.log(1.0 / n, 2)
    return p * math.log(1.0 / p, 2) + n * math.log(1.0 / n, 2)



def initialize_from_file(filename):
    """
        Read csv file into a dataframe
    """
    df = pd.read_csv(filename)
    return df

def number_of_positives(dflocal):
    """ This one calculates the total number of positive and negative outputs"""
    rowaxes, columnaxes = dflocal.axes
    number_of_positives = 0
    number_of_negatives = 0
    for i in range(len(rowaxes)):
        if (dflocal.iat[i, -1] == 1.0):
            number_of_positives += 1
        else:
            number_of_negatives += 1
    return number_of_positives, number_of_negatives


def Create_tree_TDIDT(node_list, dfa, current_node_id, tree_depth):
    """Determines the tree recursively by finding the best node heuristically"""
    current_node = node_list[current_node_id]

    rowaxes, columnaxes = dfa.axes
    pplus, pminus = number_of_positives(dfa)

    network_information_gain = 0
    final_mean = 0
    node_attribute = 0
    final_cutpoint = 0

    for current_column in range(len(columnaxes) - 1):
        df_temp = dfa.sort_values(by=[columnaxes[current_column]])
        pinnerplus = 0
        pinnerminus = 0
        max_information_gain = 0
        prev_out = 2
        for i in range(len(rowaxes)):
            if (df_temp.iat[i, -1] == 1.0):
                pinnerplus += 1
                information_gain = get_information_gain(ppos=pinnerplus, pneg=pinnerminus,
                                                        npos=(pplus - pinnerplus), nneg=(pminus - pinnerminus))
                if (information_gain > max_information_gain):
                    max_information_gain = information_gain
                    potential_cutpoint = i
                    if i > 0:
                        potential_mean = (df_temp.iat[i, current_column] +
                                          df_temp.iat[i - 1, current_column]) / 2;
                    else:
                        potential_mean = df_temp.iat[i, current_column];
            else:
                pinnerminus += 1
        if (max_information_gain > network_information_gain):
            network_information_gain = max_information_gain
            node_attribute = current_column
            final_mean = potential_mean
            final_cutpoint = potential_cutpoint


    # Updating the current array
    current_node.threshold = final_mean
    current_node.pplus = pplus
    current_node.pminus = pminus
    current_node.label = columnaxes[node_attribute]
    # The array is sorted and split
    df_temp = dfa.sort_values(by=[columnaxes[node_attribute]])
    df1 = df_temp.iloc[:final_cutpoint, :]
    df2 = df_temp.iloc[final_cutpoint:, :]

    if pplus == 0 or pminus == 0 or final_cutpoint == 0 or tree_depth >= 3:
        current_node.is_leaf = True
        current_node.outcome = (pplus > pminus)
        return
    else:
        current_node.is_leaf = False

    left_node = TDIDTNode(current_node_id)
    right_node = TDIDTNode(current_node_id)

    current_node.left_child_id = len(node_list)
    current_node.right_child_id = len(node_list) + 1

    # only needed to fullfill exercise requirements
    left_node.identifier = current_node.left_child_id
    right_node.identifier = current_node.right_child_id
    left_node.parent_test_outcome = "yes"
    right_node.parent_test_outcome = "no"

    node_list.append(left_node)
    node_list.append(right_node)
    node_list[current_node.left_child_id].identifier = current_node.left_child_id;
    node_list[current_node.right_child_id].identifier = current_node.right_child_id;
    # node_list[current_node.left_child_id].is_Left = True
    Create_tree_TDIDT(node_list, df1, current_node.left_child_id, tree_depth + 1)
    Create_tree_TDIDT(node_list, df2, current_node.right_child_id, tree_depth + 1)

    return df_temp


def classify(row, dftest, node_list):
    """Parses through the decision tree to find outcome."""
    current_node = node_list[0]

    while not current_node.is_leaf:
        if dftest._get_value(row, str(current_node.label)) < current_node.threshold:
            current_node = node_list[current_node.left_child_id]
        else:
            current_node = node_list[current_node.right_child_id]
    return current_node.outcome


def test_data_output(dftest, node_list):
    """Compares the predicted output to actual output and prints the likelihood"""
    rowaxes, columnaxes = dftest.axes
    number_of_matches = 0
    for row in range(len(rowaxes)):
        predict_op = classify(row, dftest, node_list)
        if (dftest.iat[row, -1] == predict_op):
            number_of_matches += 1
    print('Out of', len(rowaxes), 'tests run, ', number_of_matches,
          'matched the result which is at %', number_of_matches / len(rowaxes))



if __name__ == '__main__':

    # Training and test data
    train_df = pd.read_csv("new_train.csv")
    test_df = pd.read_csv("new_test.csv")

    # run decision tree algorithm
    node_list = [TDIDTNode()]
    k = Create_tree_TDIDT(node_list, train_df, 0, 0)

    # print all nodes created by TDIDT
    print('The following are the nodes created in the decision tree')
    for node in node_list:
        print(node)

    # Data set validation
    # df_validation = initialize_from_file(test_df)
    test_data_output(test_df, node_list)
