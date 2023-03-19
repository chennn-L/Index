# 把excel文件转换为csv文件
import pandas as pd
data = pd.read_excel('smote.xls','Sheet',index_col=0)
data.to_csv('smote.csv',encoding='utf-8')