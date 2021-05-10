import operator
from os import listdir
import numpy as np
import csv
import matplotlib.pyplot as plt


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


def readFile():
    with open('Iris.csv') as f:
        reader = csv.reader(f)
        next(reader)
        returnMat = np.zeros((150, 4))
        strLabels = []
        labels = []
        index = 0
        for row in reader:
            returnMat[index, :] = row[1:5]
            strLabels.append(row[-1])
            index += 1
        for item in strLabels:
            if item == 'Iris-setosa':
                labels.append(0)
            elif item == 'Iris-versicolor':
                labels.append(1)
            elif item == 'Iris-virginica':
                labels.append(2)
        return returnMat, labels


def splitData():
    mat, labels = readFile()
    traindata = np.zeros((120, 4))
    trainlabel = []
    testdata = np.zeros((30, 4))
    testlabel = []
    ind1 = 0
    ind2 = 0
    for i in range(150):
        if i % 5 == 0:
            testdata[ind1, :] = mat[i]
            testlabel.append(labels[i])
            ind1 += 1
        else:
            traindata[ind2, :] = mat[i]
            trainlabel.append(labels[i])
            ind2 += 1
    return traindata, trainlabel, testdata, testlabel


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet


def IrisClassTest():
    # 分割数据集
    traindata, trainlabel, testdata, testlabel = splitData()
    # 用上述autoNorm方法进行归一化
    autoNorm(traindata)
    autoNorm(testdata)
    errorCount = 0
    for i in range(30):
        # 使用classify0函数进行kNN分类
        res, _ = classify0(testdata[i, :], traindata, trainlabel, 3)
        print("Classified as %d | Ground Truth: %d" % (res, testlabel[i]))
        if res != testlabel[i]:
            errorCount += 1
    print(
        "Total error number: {} error rate: {}%".format(errorCount, errorCount * 100 / 30))


def showError():
    traindata, trainlabel, testdata, testlabel = splitData()
    traindata = autoNorm(traindata)
    testdata = autoNorm(testdata)
    er = []
    for k in range(1, 11):
        errorCount = 0
        for i in range(30):
            res, _ = classify0(testdata[i, :], traindata, trainlabel, k)
            if res != testlabel[i]:
                errorCount += 1
        er.append(errorCount * 100 / 30)

    plt.plot(list(range(1, 11)), er, color='blue')
    plt.xlabel('Number of neighbor voters')
    plt.ylabel('Error rate / %')
    plt.show()


IrisClassTest()
