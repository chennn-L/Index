##用于可视化图表
import matplotlib.pyplot as plt
import pylab
##用于做科学计算
import numpy as np
##导入PCA库
from sklearn.decomposition import PCA
plt.switch_backend('agg')
# # 对smote过采样后的数据进行降维可视化
# smote=np.load("smote_test.npy")    #用于PCA降维
# smote_reshape=smote.reshape(186,1*13698)

# #加载PCA模型并训练，降维
# #注意：PCA为无监督学习 无法使用类别信息来降维
# model_pca=PCA(n_components=2) #二维
# X_pca=model_pca.fit_transform(smote_reshape)   #降维后的数据
# # print(X_pca)

# plt.scatter(X_pca[:,0],X_pca[:,1])
# #保存散点图
# plt.savefig("./smote_after.jpg")

# # 对pos_data数据进行降维可视化
smote=np.load("pos_data.npy")    #用于PCA降维
smote_reshape=smote.reshape(93,1*13698)

#加载PCA模型并训练，降维
#注意：PCA为无监督学习 无法使用类别信息来降维
model_pca=PCA(n_components=2) #二维
X_pca=model_pca.fit_transform(smote_reshape)   #降维后的数据
# print(X_pca)

plt.scatter(X_pca[:,0],X_pca[:,1])
#保存散点图
plt.savefig("./smote_before.jpg")