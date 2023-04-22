##用于可视化图表
import matplotlib.pyplot as plt
import pylab
##用于做科学计算
import numpy as np
import pandas as pd
##导入PCA库
from sklearn.decomposition import PCA
plt.switch_backend('agg')

# # 对smote过采样后的数据进行降维可视化
# smote=np.load("smote_train_data.npy")    #用于PCA降维
# smote_reshape=smote.reshape(124,1*13698)
# #加载PCA模型并训练，降维
# #注意：PCA为无监督学习 无法使用类别信息来降维
# model_pca=PCA(n_components=2) #二维
# X_pca=model_pca.fit_transform(smote_reshape)   #降维后的数据
# # print(X_pca)
# print(X_pca.shape)
# plt.scatter(X_pca[:,0],X_pca[:,1])
# #保存散点图
# plt.savefig("./smote_after.jpg")

# # 对pos_data数据进行降维可视化
# smote=np.load("train_data.npy")    #用于PCA降维
# smote_reshape=smote.reshape(62,1*13698)
# #加载PCA模型并训练，降维
# #注意：PCA为无监督学习 无法使用类别信息来降维
# model_pca=PCA(n_components=2) #二维
# X_pca=model_pca.fit_transform(smote_reshape)   #降维后的数据
# # print(X_pca.shape)
# plt.scatter(X_pca[:,0],X_pca[:,1])
# #保存散点图
# plt.savefig("./smote_before.jpg")

# 构建训练集数据
# # 对train_data2数据(smote)进行降维可视化
# smote=np.load("./data2/train_data2.npy")    #用于PCA降维
# smote_reshape=smote.reshape(246,1*13698)
# #加载PCA模型并训练，降维
# #注意：PCA为无监督学习 无法使用类别信息来降维
# model_pca=PCA(n_components=2) #二维
# X_pca=model_pca.fit_transform(smote_reshape)   #降维后的数据
# # print(X_pca.shape)
# b=np.ones(124)
# c=-np.ones(122)
# co=np.insert(b,124,values=c,axis=0)
# # print(co)
# train_data1=np.column_stack((X_pca,co))
# np.savetxt("./data2/train_data2.txt", train_data1, fmt='%.6f', delimiter='   ')

# # 对train_data1数据进行降维可视化
# smote=np.load("./data1/train_data1.npy")    #用于PCA降维
# smote_reshape=smote.reshape(184,1*13698)
# #加载PCA模型并训练，降维
# #注意：PCA为无监督学习 无法使用类别信息来降维
# model_pca=PCA(n_components=2) #二维
# X_pca=model_pca.fit_transform(smote_reshape)   #降维后的数据
# # print(X_pca.shape)
# b=np.ones(62)
# c=-np.ones(122)
# co=np.insert(b,62,values=c,axis=0)
# # print(co)
# train_data1=np.column_stack((X_pca,co))
# # print(train_data1)
# np.savetxt("./data1/train_data1.txt", train_data1, fmt='%.6f', delimiter='   ')


# # 构建测试集数据
# 对test_data2数据进行降维可视化
smote=np.load("./data2/test_data2.npy")    #用于PCA降维
smote_reshape=smote.reshape(92,1*13698)
#加载PCA模型并训练，降维
#注意：PCA为无监督学习 无法使用类别信息来降维
model_pca=PCA(n_components=2) #二维
X_pca=model_pca.fit_transform(smote_reshape)   #降维后的数据
# print(X_pca.shape)
b=np.ones(31)
c=-np.ones(61)
co=np.insert(b,31,values=c,axis=0)
# print(co)
test_data1=np.column_stack((X_pca,co))
# print(test_data1)
np.savetxt("./data2/test_data2.txt", test_data1, fmt='%.6f',  delimiter='   ')

# # 对test_data1数据进行降维可视化
# smote=np.load("./data1/test_data1.npy")    #用于PCA降维
# smote_reshape=smote.reshape(92,1*13698)
# #加载PCA模型并训练，降维
# #注意：PCA为无监督学习 无法使用类别信息来降维
# model_pca=PCA(n_components=2) #二维
# X_pca=model_pca.fit_transform(smote_reshape)   #降维后的数据
# # print(X_pca.shape)
# b=np.ones(31)
# c=-np.ones(61)
# co=np.insert(b,31,values=c,axis=0)
# # print(co)
# test_data1=np.column_stack((X_pca,co))
# # print(test_data1)
# np.savetxt("./data1/test_data1.txt", test_data1, fmt='%.6f',  delimiter='   ')
# # 对test_data1数据进行降维可视化
# smote=np.load("./data1/test_data1.npy")    #用于PCA降维
# smote_reshape=smote.reshape(92,1*13698)
# #加载PCA模型并训练，降维
# #注意：PCA为无监督学习 无法使用类别信息来降维
# model_pca=PCA(n_components=2) #二维
# X_pca=model_pca.fit_transform(smote_reshape)   #降维后的数据
# # print(X_pca.shape)
# b=np.ones(31)
# c=-np.ones(61)
# co=np.insert(b,31,values=c,axis=0)
# # print(co)
# test_data1=np.column_stack((X_pca,co))
# # print(test_data1)
# np.savetxt("./data1/test_data1.txt", test_data1, fmt='%.6f',  delimiter='   ')
