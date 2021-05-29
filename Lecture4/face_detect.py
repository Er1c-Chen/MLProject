import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import re


def loadData():
    path = './data/project1-data-Recognition/train'
    files = os.listdir(path)
    image = []
    labels = []
    pattern = re.compile(r'(?<=subject)\d*')
    for file in files:
        img = cv2.imread(path + '/' + file, 0)
        image.append(img.flatten() / 255)
        labels.append(int(pattern.findall(file)[0]))
    return np.array(image), np.array(labels)


image, labels = loadData()


def pca(topNFeat=1e5):
    dataMat, labels = loadData()
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    # u, s, vt = np.linalg.svd(meanRemoved)
    # print(vt, vt.shape)
    '''
    # 计算协方差矩阵及特征值
    covMat = np.cov(meanRemoved, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    '''
    # print('Please wait......')
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
    return reconMat


def testSample():
    image = []
    for i in range(10):
        image.append(pca(topNFeat=10)[i].reshape(231, 195))
    fig = plt.figure
    for i in range(1, 10):
        plt.subplot(3, 3, i)
        plt.imshow(image[i - 1], cmap=plt.cm.gray)
        plt.title('eigenface{}'.format(i))
        plt.xticks([])
        plt.yticks([])
    plt.show()

testSample()