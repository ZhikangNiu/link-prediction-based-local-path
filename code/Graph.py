

"""
    a  b  c  d  e
a   0  1  0  0  1
b   1  0  1  1  0
c   0  1  0  1  0
d   0  1  1  0  1
e   1  0  0  1  0

用邻接矩阵读取

邻接矩阵的平方来进行计算

传入参数两个点(以元组形式传入)
    计算二长路，并得出数量，邻接矩阵平方项相对应的数字
    计算三长路，并得出数量，邻接矩阵立方项对应的数字

计算公式给予二阶一个参数，三阶一个参数，返回一个结果（二分类）
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def LocalPathMatrix(AdjacenceyMatrix):
    # 需要判断矩阵是否为空或者全 0
    if AdjacenceyMatrix.sum() == 0:
        scoreMatrix = np.zeros(AdjacenceyMatrix.shape)
        return scoreMatrix
    else:
        # 计算邻接矩阵的平方和立方项
        L2Matrix = np.dot(AdjacenceyMatrix,AdjacenceyMatrix)
        L3Matrix = np.dot(L2Matrix,AdjacenceyMatrix)
        # 三阶路径的参数
        # Alpha的参数不应该是一个定值，应该根据二阶和三阶的数目来确定
        L2alpha = L2Matrix.sum()
        L3alpha = L3Matrix.sum()
        if (L2alpha+L3alpha) != 0:
            Alpha = L3alpha/(L2alpha+L3alpha)
        else:
            Alpha = 0.5
        # 计算得分矩阵
        scoreMatrix = L2Matrix + Alpha * L3Matrix
        # 返回结果矩阵
        return scoreMatrix

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x/np.sum(exp_x)
    return softmax_x

def clshead(scoreMatrix,checkpoint=None):
    # 矩阵归一化
    NormalizedMatrix = softmax(scoreMatrix)
    if checkpoint is not None:
        # 获取两点的坐标
        i = checkpoint[0]
        j = checkpoint[1]
        score = NormalizedMatrix[i][j]
        return NormalizedMatrix,score
    else:
        return NormalizedMatrix


if __name__ == '__main__':
    AdjacenceyMatrix = np.array(eval(input("please input Graph AdjacenceyMatrix: ")))
    input_graph = nx.from_numpy_array(AdjacenceyMatrix)
    print(input_graph)

    scoreMatrix = LocalPathMatrix(AdjacenceyMatrix)
    flag = eval(input("if you need check points prob, please input 1: "))
    if flag == 1:
        checkpoint = tuple(eval(input("please input the point that you want to check: ")))
        NormalizedMatrix,prob = clshead(scoreMatrix,checkpoint=checkpoint)
    else:
        NormalizedMatrix = clshead(scoreMatrix)
    print('-'*6 + 'scoreMatrix' + '-'*6)
    print("ScoreMatrix = \n{}".format(scoreMatrix))
    print('-'*6 + 'Normalized scoreMatrix' + '-'*6)
    print("After Normalized ScoreMatrix = \n{}".format(NormalizedMatrix))
    print('-'*6 + 'prob of point' + '-'*6)

    # 计算平均值和矩阵的大小
    mean = 1/AdjacenceyMatrix.size
    row,col = AdjacenceyMatrix.shape
    mean = 0.002
    print("threshold = {}".format(mean))
    # 对矩阵进行copy，并对符合条件的置1，不满足置0
    addEdgeMetrix = NormalizedMatrix.copy()
    addEdgeMetrix[addEdgeMetrix>mean] = 1
    addEdgeMetrix[addEdgeMetrix<=mean] = 0
    
    # 记录下需要加点的曲线
    new_graph = input_graph.copy()
    for j in range(col):
        for i in range(row):
            if addEdgeMetrix[i][j] == 1 and i!=j:
                new_graph.add_edge(i,j,color="red")
            else:
                continue

    subax1 = plt.subplot(121)
    nx.draw(input_graph,with_labels = True)   # default spring_layout
    subax2 = plt.subplot(122)
    nx.draw(new_graph,with_labels=True)
    plt.show()
