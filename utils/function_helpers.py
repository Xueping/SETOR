#################################################################
# Code written by Xueping Peng according to GRAM paper and original codes
#################################################################
from __future__ import print_function
from __future__ import division
import pickle
import numpy as np


def print2file(buf, outFile):
    print(buf)
    outfd = open(outFile, 'a')
    outfd.write(buf + '\n')
    outfd.close()


def get_rootCode(treeFile):
    tree = pickle.load(open(treeFile, 'rb'))
    rootCode = list(tree.values())[0][1]
    return rootCode


def build_tree_with_padding(treeFile, max_len_ancestors=6):
    # max_len_ancestors = 6 # the max length of code's ancestors including itself
    treeMap = pickle.load(open(treeFile, 'rb'))
    if len(treeMap) == 0:
        return [], [], []
    ancestors = np.array(list(treeMap.values())).astype('int32')
    ancSize = ancestors.shape[1]
    leaves = []
    for k in treeMap.keys():
        leaves.append([k] * ancSize)
    leaves = np.array(leaves).astype('int32')
    # add padding and mask
    if ancSize < max_len_ancestors:
        ones = np.ones((ancestors.shape[0], ancSize)).astype('int32')
        zeros = np.zeros((ancestors.shape[0], max_len_ancestors-ancSize)).astype('int32')
        leaves = np.concatenate([leaves, zeros], axis=1)
        ancestors = np.concatenate([ancestors, zeros], axis=1)
        mask = np.concatenate([ones, zeros], axis=1)
    else:
        mask = np.ones((ancestors.shape[0], max_len_ancestors))
    return leaves, ancestors, mask




