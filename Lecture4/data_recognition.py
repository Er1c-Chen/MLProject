import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def loadData():
    path = './data/project1-data-Recognition/train'
    files = os.listdir(path)
    image = []
    for file in files:
        img = cv2.imread(path + '/' + file, 0)
        image.append(img.flatten()/255)
    return np.array(image)


image = loadData()


def meanFace():
    meanface = []
    with open('./data/meanface.txt') as f:
        text = f.read()
        for item in text.split(','):
            meanface.append(float(item))
    '''
    m = image.shape[1]
    print(m)
    for i in range(m):
        meanface = np.mean(image, axis=0)
    '''
    meanface = np.array(meanface)
    plt.figure('Meanface')
    cv2.imshow('Meanface', meanface.reshape(231, 195))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return meanface

meanFace()
def pca(topNFeat):
    dataMat = loadData()
    meanVals = meanFace(dataMat)
    meanRemoved = dataMat - meanVals
    # 计算协方差矩阵及特征值
    covMat = np.cov(meanRemoved, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    # 将属性特征值从小到大排序
    eigValInd = np.argsort(eigVals)
    # 除去不需要的特征属性
    eigValInd = eigValInd[:-(topNFeat + 1):-1]
    # 将特征值逆序排列
    redEigVects = eigVects[:, eigValInd]
    # 将特征属性映射到新的空间中
    lowDDataMat = meanRemoved * redEigVects
    # 加回均值
    reconMat = (lowDDataMat * redEigVects.T) + meanVals