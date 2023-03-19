import random
import xlrd
import xlwt  # 负责写excel
from sklearn.neighbors import NearestNeighbors
import numpy as np
import scipy.io as scio
class Smote:
    def __init__(self,samples,N=10,k=5):
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        self.k=k
        # sanples是距离决定因子
        self.samples=samples
        # smote算法已合成的样本个数
        self.newindex=0
       # self.synthetic=np.zeros((self.n_samples*N,self.n_attrs))

    def over_sampling(self):
        N=int(self.N/100)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        print ('neighbors',neighbors)
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            #print nnarray
            self._populate(N,i,nnarray)
        return self.synthetic


    # for each minority class samples,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self,N,i,nnarray):
        for j in range(N):
            nn=random.randint(0,self.k-1)
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap=random.random()
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            self.newindex+=1

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
  
pathX = 'pos_data.xlsx'  
x = excel2matrix(pathX)
a=x
s=Smote(a,N=100)
print (s.over_sampling())
print (s.over_sampling().shape)
  
filename =xlwt.Workbook() #创建工作簿
sheet1 = filename.add_sheet(u'sheet1',cell_overwrite_ok=True) #创建sheet
[h,l]=s.over_sampling().shape #h为行数，l为列数
for i in range (h):
    for j in range (l):
        sheet1.write(i,j,s.over_sampling[i,j])
filename.save('pos_data.xls')