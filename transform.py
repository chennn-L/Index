##把xlsx文件转化为npy文件
import numpy as np
import xlrd
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
matrix = excel2matrix(pathX)
np.save('pos_data.npy',matrix)