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
    return sortedClassCount[0][0]


def img2vector(filename):
    returnVect = np.zeros((1, 1024))  # 初始化数字
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        # 将32*32的数组转化为1*1024
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    Labels = []
    trainingFileList = listdir('trainingDigits')  # 进入训练集文件夹
    m = len(trainingFileList)  # 训练集文件夹长度
    trainingMat = np.zeros((m, 1024))  # 初始化矩阵
    for i in range(m):
        fileNameStr = trainingFileList[i]  # 获取到测试样本文件名
        fileStr = fileNameStr.split('.')[0]  # 去掉.txt后缀名
        classNumStr = int(fileStr.split('_')[0])  # 获取训练样本序号
        Labels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)  # 读取训练样本内容
    testFileList = listdir('testDigits')  # 进入测试样本文件夹
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # 去掉.txt后缀名
        classNumStr = int(fileStr.split('_')[0])  # 获取测试样本序号
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)  # 读取测试样本内容
        classifierResult = classify0(vectorUnderTest, trainingMat, Labels, 3)
        print("Classified as: %d | Ground Truth: %d" % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))


handwritingClassTest()
