import random
import xlrd
import xlwt
from sklearn.neighbors import NearestNeighbors
import numpy as np
 
class Smote(object):
    
    def __init__(self, N=50, k=5, r=2):
        # 初始化self.N, self.k, self.r, self.newindex
        self.N = N
        self.k = k
        # self.r是距离决定因子
        self.r = r
        # self.newindex用于记录SMOTE算法已合成的样本个数
        self.newindex = 0
        
    # 构建训练函数
    def fit(self, samples):
        # 初始化self.samples, self.T, self.numattrs
        self.samples = samples
        # self.T是少数类样本个数，self.numattrs是少数类样本的特征个数
        self.T, self.numattrs = self.samples.shape

        # 查看N%是否小于100%
        if(self.N < 100):
            # 如果是，随机抽取N*T/100个样本，作为新的少数类样本
            np.random.shuffle(self.samples)
            self.T = int(self.N*self.T/100)
            self.samples = self.samples[0:self.T,:]
            # N%变成100%
            self.N = 100

        # 查看从T是否不大于近邻数k
        if(self.T <= self.k):
            # 若是，k更新为T-1
            self.k = self.T - 1

        # 令N是100的倍数
        N = int(self.N/100)
        # 创建保存合成样本的数组
        self.synthetic = np.zeros((self.T * N, self.numattrs))

        # 调用并设置k近邻函数
        neighbors = NearestNeighbors(n_neighbors=self.k+1, 
                                     algorithm='ball_tree', 
                                     p=self.r).fit(self.samples)

        # 对所有输入样本做循环
        for i in range(len(self.samples)):
            # 调用kneighbors方法搜索k近邻
            nnarray = neighbors.kneighbors(self.samples[i].reshape((1,-1)),
                                           return_distance=False)[0][1:]

            # 把N,i,nnarray输入样本合成函数self.__populate
            self.__populate(N, i, nnarray)

        # 最后返回合成样本self.synthetic
        return self.synthetic
    
    # 构建合成样本函数
    def __populate(self, N, i, nnarray):
        # 按照倍数N做循环
        for j in range(N):
            # attrs用于保存合成样本的特征
            attrs = []
            # 随机抽取1～k之间的一个整数，即选择k近邻中的一个样本用于合成数据
            nn = random.randint(0, self.k-1)
            
            # 计算差值
            diff = self.samples[nnarray[nn]] - self.samples[i]
            # 随机生成一个0～1之间的数
            gap = random.uniform(0,1)
            # 合成的新样本放入数组self.synthetic
            self.synthetic[self.newindex] = self.samples[i] + gap*diff

            # self.newindex加1， 表示已合成的样本又多了1个
            self.newindex += 1

# samples = np.array([[3,1,2], [4,3,3], [1,3,4],
#                     [3,3,2], [2,2,1], [1,4,3]])
# smote = Smote(N=200)
# synthetic_points = smote.fit(samples)
# print(synthetic_points)

# 把xlsx文件转化成矩阵
def excel2matrix(path):
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数
    datamatrix = np.zeros((nrows, ncols))
    for i in range(ncols):
        cols = table.col_values(i)
        datamatrix[:, i] = cols
    return datamatrix

pathX = 'train_data.xlsx' 
matrix = excel2matrix(pathX) #把xlsx文件转成数组
# print(type(matrix))
samples = matrix.transpose() #数组转置
# print(samples.shape)
smote = Smote(N=200)
synthetic_points = smote.fit(samples)
print(synthetic_points)
# print(synthetic_points.shape)
example = synthetic_points.transpose() #数组转置
np.save('smote_train.npy',example)
workbook = xlwt.Workbook()
sheet = workbook.add_sheet("Sheet")

for i in range(len(example)):
    for j in range(len(example[i])):
        sheet.write(i, j, example[i][j])

workbook.save("smote_train.xls")

