import numpy as np


def loadSimpData():  # 读取数据集与对应的标签
    datMat = np.array([[1., 2.1],
                       [2., 1.1],
                       [1.3, 1.],
                       [1., 1.],
                       [2., 1.]])
    classLabels = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    return datMat, classLabels


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):  # just classify the data
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T  # 转置标签向量
    m, n = np.shape(dataMatrix)  # 样本的行数列数
    numSteps = 10.0  # 初始化步数
    bestStump = {}  # 初始化决策树
    bestClasEst = np.mat(np.zeros((m, 1)))  # 默认类别为0
    minError = np.inf  # 初始化错误为无穷大
    for i in range(n):  # 遍历数据所有属性
        rangeMin = dataMatrix[:, i].min()  # 在当前属性中选取最小值
        rangeMax = dataMatrix[:, i].max()  # 在当前属性中选取最大值
        stepSize = (rangeMax - rangeMin) / numSteps  # 给定步数 计算步长
        for j in range(-1, int(numSteps) + 1):  # 在当前维度内循环步数次
            for inequal in ['lt', 'gt']:  # 对值小于threshVal时分类分为为1.0还是-1.0两种情况进行遍历
                threshVal = (rangeMin + float(j) * stepSize)  # 计算划分值threshVal
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)  # 得到分类结果
                errArr = np.mat(np.ones((m, 1)))  # errArr默认为全1矩阵
                errArr[predictedVals == labelMat] = 0  # 比对分类结果与标签进行 分类正确为0 错误为1
                weightedError = D.T * errArr  # 计算加权后的误差
                if weightedError < minError:  # 如果误差值小于最小误差则更新最小误差为当前误差
                    minError = weightedError
                    bestClasEst = predictedVals.copy()  # 更新分类结果
                    # 更新最佳的决策树为当前信息
                    bestStump['dim'] = i  # i表示是以第1列还是第2列为分类依据
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal  # inequal决定值小于threshVal分类为-1.0还是1.0
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)  # init D to all equal
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)  # build Stump
        # print "D:",D.T
        alpha = float(
            0.5 * np.log(
                (1.0 - error) / max(error, 1e-16)))  # calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)  # store Stump Params in Array
        # print "classEst: ",classEst.T
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)  # exponent for D calc, getting messy
        D = np.multiply(D, np.exp(expon))  # Calc New D for next iteration
        D = D / D.sum()
        # calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha * classEst
        # print ("aggClassEst: ",aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate)
        if errorRate == 0.0: break
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    dataMatrix = np.mat(datToClass)  # do stuff similar to last aggClassEst in adaBoostTrainDS
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(  # 调用stumpClassify函数进行预测
            dataMatrix,
            classifierArr[i]['dim'],
            classifierArr[i]['thresh'],
            classifierArr[i]['ineq']
        )
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)


mat, label = loadSimpData()
adaBoostTrainDS(mat, label)
