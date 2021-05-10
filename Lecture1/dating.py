"""
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)

Output:     the most popular class label

@author: pbharrin
"""
from numpy import *
import operator
from os import listdir
import numpy as np


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0], classCount


# 将读入的txt文件转为矩阵
def file2matrix(filename):
    fr = open(filename)
    lines = len(fr.readlines())  #读入txt文件的行数
    returnMat = np.zeros((lines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')  #通过制表符分隔
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))   #读取listFromLine列表的最后一个元素
        index += 1
    return returnMat, classLabelVector


# 将数据集进行最大最小标准化
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))  #对每个元素进行除法
    return normDataSet


def datingClassTest():
    hoRatio = 0.10  #取数据集的10%样本
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  #从文件中读取数据集与标签
    normMat = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        #使用classify0函数进行kNN算法
        res, _ = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("Classified as %d | Ground Truth: %d" % (res, datingLabels[i]))
        if res != datingLabels[i]:
            errorCount += 1
    print("Total error number: {}, the error rate: {}%"
          .format(errorCount, errorCount*100/numTestVecs))


datingClassTest()

