# 逻辑回归（Logistic Regression）
# https://www.cnblogs.com/Belter/p/6128644.html
# 估计回归系数a的值的过程
from numpy import *
import os

print(os.listdir("../input"))
path = '../input/lr/'
training_sample = 'trainingSample.txt'
testing_sample = 'testingSample.txt'


# 从文件中读入训练样本的数据，同上面给出的示例数据
# 下面第20行代码中的1.0表示x0 = 1
def loadDataSet(p, file_n):
    dataMat = []
    labelMat = []
    fr = open(os.path.join(p, file_n))
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # 三个特征x0, x1, x2
        labelMat.append(int(lineArr[2]))  # 标准答案y
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# 梯度下降法求回归系数a，由于样本量少，我将迭代次数改成了1000次
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)  # convert to NumPy matrix
    labelMat = mat(classLabels).transpose()  # convert to NumPy matrix
    m, n = shape(dataMatrix)
    alpha = 0.001  # 学习率
    maxCycles = 1000
    weights = ones((n, 1))
    for k in range(maxCycles):  # heavy on matrix operations
        h = sigmoid(dataMatrix * weights)  # 模型预测值, 90 x 1
        error = h - labelMat  # 真实值与预测值之间的误差, 90 x 1
        temp = dataMatrix.transpose() * error  # 交叉熵代价函数对所有参数的偏导数, 3 x 1
        weights = weights - alpha * temp  # 更新权重
    return weights


# 分类效果展示，参数weights就是回归系数
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet(path, training_sample)
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]  # x2 = f(x1)
    ax.plot(x.reshape(1, -1), y.reshape(1, -1))
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# 下面是我自己写的测试函数
def test_logistic_regression():
    dataArr, labelMat = loadDataSet(path, training_sample)  # 读入训练样本中的原始数据
    A = gradAscent(dataArr, labelMat)  # 回归系数a的值
    h = sigmoid(mat(dataArr) * A)  # 预测结果h(a)的值
    print(dataArr, labelMat)
    print(A)
    print(h)
    plotBestFit(A)  # 取消注释显示图像


test_logistic_regression()


# 添加一个预测函数，如下：
# 直接将上面计算出来的回归系数a拿来使用，测试数据其实也是《机器学习实战》这本书中的训练数据，
# 我拆成了两份，前面90行用来做训练数据，后面10行用来当测试数据。
def predict_test_sample():
    A = [5.262118, 0.60847797, -0.75168429]  # 上面计算出来的回归系数a
    dataArr, labelMat = loadDataSet(path, testing_sample)
    h_test = sigmoid(mat(dataArr) * mat(A).transpose())  # 将读入的数据和A转化成numpy中的矩阵
    print(h_test)  # 预测的结果

# https://www.cnblogs.com/Belter/p/6128644.html
# 上面代码的输出如下：
#
# 一个元组，包含两个数组：第一个数组是所有的训练样本中的观察值，也就是X，包括x0, x1, x2；第二个数组是每组观察值对应的标准答案y。
# ([[1.0, -0.017612, 14.053064], [1.0, -1.395634, 4.662541], [1.0, -0.752157, 6.53862], [1.0, -1.322371, 7.152853],
#   [1.0, 0.423363, 11.054677], [1.0, 0.406704, 7.067335], [1.0, 0.667394, 12.741452], [1.0, -2.46015, 6.866805],
#   [1.0, 0.569411, 9.548755], [1.0, -0.026632, 10.427743]], [0, 1, 0, 0, 0, 1, 0, 1, 0, 0])
#
# 本次预测出来的回归系数a，包括a0, a1, a2
# [[1.39174871]
#  [-0.5227482]
#  [-0.33100373]]
#
# 根据回归系数a和（2）式中的模型预测出来的h(a)。这里预测得到的结果都是区间(0, 1)
# 上的实数。
#
# [[0.03730313]
#  [0.64060602]
#  [0.40627881]
#  [0.4293251]
#  [0.07665396]
#  [0.23863652]
#  [0.0401329]
#  [0.59985228]
#  [0.11238742]
#  [0.11446212]]

# 附件：
# github上的代码更新到python3.6, 2019-1-6
# 完整代码：https://github.com/OnlyBelter/MachineLearning_examples/tree/master/de_novo/regression
# 训练数据：https://github.com/OnlyBelter/MachineLearning_examples/blob/master/de_novo/data/Logistic_Regression-trainingSample.txt
# 测试数据：https://github.com/OnlyBelter/MachineLearning_examples/blob/master/de_novo/data/Logistic_Regression-testingSample.txt
