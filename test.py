# 按列读
import xlrd
import numpy as np
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

pathX = 'ma2.xlsx'  #  113.xlsx 在当前文件夹下
x = excel2matrix(pathX)
print(x)
print(x.shape)