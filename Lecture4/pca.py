import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def loadDataSet(fileName, delim='\t'):
    # 导入数据集 并使用制表符分割样本属性
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return np.mat(datArr)


def PCA(dataMat, topNFeat=1e6):
    # topNFeat为返回属性的特征数量，若不指定则返回前1e6个
    # 计算平均值
    meanVals = np.mean(dataMat, axis=0)
    # 属性减掉均值
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
    return lowDDataMat, reconMat


def replaceNaNWithMean():
    # 将缺失数据替换为均值
    dataMat = loadDataSet('./data/secom.data', ' ')
    numFeat = np.shape(dataMat)[1]
    ind = []
    nanind = []
    for i in range(numFeat):
        # 找到除确实数据外的其他数据 计算均值
        for item in dataMat[:, i]:
            if not np.isnan(item):
                ind.append(item.index)
            else:
                nanind.append(item.index)
            meanVal = np.mean(dataMat[ind, i])
            dataMat[nanind, i] = meanVal

        # meanVal = np.mean(dataMat[np.nonzero(np.isnan(dataMat[:, i].A))[0], i])
        # 修正NaN为均值
        # dataMat[np.nonzero(np.isnan(dataMat[:, i].A))[0], i] = meanVal
    return dataMat


dataMat = loadDataSet('./data/testSet.txt')
lowDData, reconMat = PCA(dataMat, 1)
print(dataMat.shape, lowDData.shape)

fig = plt.figure()
ax = fig.add_subplot(111)
sc1 = ax.scatter(dataMat[:, 0].squeeze().A[0], dataMat[:, 1].squeeze().A[0],
                 marker='^', s=50, c='orange', alpha=0.2)
sc2 = ax.scatter(reconMat[:, 0].squeeze().A[0], reconMat[:, 1].squeeze().A[0],
                 marker='o', s=50, c='red', alpha=0.1)
plt.legend([sc1, sc2], ['Original', 'After PCA'], loc='lower right', scatterpoints=1)
plt.show()

secomData = replaceNaNWithMean()
meanVals = np.mean(dataMat, axis=0)
# 属性减掉均值
meanRemoved = dataMat - meanVals
# 计算协方差矩阵及特征值
covMat = np.cov(meanRemoved, rowvar=0)
eigVals, eigVects = np.linalg.eig(np.mat(covMat))
print(eigVals)
