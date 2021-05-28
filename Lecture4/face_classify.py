import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import re
import random
import operator


def loadData():
    #
    path = './data/project1-data-Recognition/train'
    files = os.listdir(path)
    image = []
    labels = []
    pattern = re.compile(r'(?<=subject)\d*')
    for file in files:
        img = cv2.imread(path + '/' + file, 0)
        image.append(img.flatten() / 255)
        labels.append(int(str(pattern.findall(file)[0])))
    return np.array(image), np.array(labels)


def splitTrainTest(image, N):
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for sub in range(15):
        sample = random.sample(range(sub * 11, (1 + sub) * 11), N)
        ind = set(range(sub * 11, (1 + sub) * 11))
        unsample = ind - set(sample)
        for x in image[sample]:
            train_data.append(x)
            train_label.append(sub + 1)
        for x in image[list(unsample)]:
            test_data.append(x)
            test_label.append(sub + 1)
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    return train_data, train_label, test_data, test_label


def pca(dataMat, topNFeat=1e7):
    meanVals = np.mean(dataMat, axis=0)
    # 属性减掉均值
    meanRemoved = dataMat - meanVals
    c_ = np.matmul(meanRemoved, meanRemoved.T)
    eig, vec = np.linalg.eig(np.mat(c_))
    eigVects = np.dot(meanRemoved.T, vec)

    eigValInd = np.argsort(eig)
    # 除去不需要的特征属性
    eigValInd = eigValInd[:-(topNFeat + 1):-1]
    # 将特征值逆序排列
    redEigVects = eigVects[:, eigValInd]
    lowDDataMat = np.matmul(meanRemoved, redEigVects)
    # 加回均值
    reconMat = np.matmul(lowDDataMat, redEigVects.T) + meanVals
    return lowDDataMat, redEigVects, meanVals


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.array(np.tile(inX.flatten(), (dataSetSize, 1)) - dataSet)
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


def faceClassify(image, labels, N):
    for K in [10]:
        for t in range(10):
            random.seed(33311)
            train_data, train_label, test_data, test_label = splitTrainTest(image, N)
            errorCount = 0
            er = []
            eigenTrain, eigenVec, meanface = pca(train_data, topNFeat=K)
            for i, test in enumerate(test_data):
                test = test - meanface
                eigenTest = np.dot(eigenVec.T, test)
                res, _ = classify0(eigenTest.flatten(), eigenTrain, train_label, 1)
                # print('Face #{} classified as subject {} | Ground truth: {}'.format(i+1, res, test_label[i]))
                if res != test_label[i]:
                    errorCount += 1
            print("Dim K={} | Error: {}/{} | Error Rate: {}%".format(K, errorCount, len(test_label),
                                                                     errorCount * 100 / len(test_label)))
            er.append(errorCount * 100 / len(test_label))
        print('Average Error Rate:{:.2f}%'.format(np.mean(er)))


N = 8
image, labels = loadData()
faceClassify(image, labels, N)
