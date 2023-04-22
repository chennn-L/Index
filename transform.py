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

# smote测试集数据
pathX = './data2/train_data2.xlsx' 
matrix = excel2matrix(pathX)
np.save('./data2/train_data2.npy',matrix)
# smote训练集数据
pathX = './data2/test_data2.xlsx' 
matrix = excel2matrix(pathX)
np.save('./data2/test_data2.npy',matrix)

# # 原始测试集数据
# pathX = './data1/train_data1.xlsx' 
# matrix = excel2matrix(pathX)
# np.save('./data1/train_data1.npy',matrix)
# # 原始训练集数据
# pathX = './data1/test_data1.xlsx' 
# matrix = excel2matrix(pathX)
# np.save('./data1/test_data1.npy',matrix)


# # 合并xlsx文件
# import pandas as pd
# df1=pd.read_excel("./data2/smote_data.xlsx")
# df2=pd.read_excel("./data2/train_data_co=0.xlsx")
# res_heng= pd.concat([df1,df2], axis=1)
# # print(type(res_heng))
# #导出文件
# res_heng.to_excel("./data2/train_data2.xlsx",index = False)


# # 把xls文件转化成xlsx文件
# import xlrd
# from openpyxl.workbook import Workbook
# def open_xls_as_xlsx(xls_path, xlsx_path):
#     # first open using xlrd
#     book = xlrd.open_workbook(xls_path)
#     index = 0
#     nrows, ncols = 0, 0
#     sheet = book.sheet_by_index(0)
#     while nrows * ncols == 0:
#         sheet = book.sheet_by_index(index)
#         nrows = sheet.nrows
#         ncols = sheet.ncols
#         index += 1
#     # prepare a xlsx sheet
#     book_new = Workbook()
#     sheet_new = book_new.create_sheet("sheet1", 0)
#     for row in range(0, nrows):
#         for col in range(0, ncols):
#             sheet_new.cell(row=row+1, column=col+1).value = sheet.cell_value(row, col)
#     book_new.save(xlsx_path)
# xls_path="./smote_data/smote_train.xls"
# xlsx_path="./data2/smote_data.xlsx"
# open_xls_as_xlsx(xls_path,xlsx_path)