import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
"""
每个像素点做成一张图
例如：将3x3数据立方体分成9个长条作为RNN的输入
一个数据立方体是一个样本点 （9，220）
"""
mat = sio.loadmat('data/Indian_pines.mat')['indian_pines']
label = sio.loadmat('data/Indian_pines_gt.mat')['indian_pines_gt']
Height = 145
Width = 145
ratio = 0.1
pca = PCA(n_components='mle')


def sample_wise_standardization(data):
    import math
    _mean = np.mean(data)
    _std = np.std(data)
    npixel = np.size(data) * 1.0
    min_stddev = 1.0 / math.sqrt(npixel)

    return (data - _mean) / max(_std, min_stddev)


def class_seperation(data, region_size):
    m, n = label.shape
    r = region_size
    labels = []
    site_feature = []
    for i in range(m):
        for j in range(n):
            if label[i][j] != 0:
                labels.append(label[i][j]-1)
                cube = []
                for idx in range(i-r, i+r+1):
                    for idy in range(j-r, j+r+1):
                        idx = (idx+Width) % Width
                        idy = (idy+Height) % Height
                        cube.append(data[idx][idy][:])
                site_feature.append(cube)

    Data = np.array(site_feature)
    Data = sample_wise_standardization(Data)
    Label = np.array(labels)
    Label = to_categorical(Label)

    TrnData, TestData, TrnLabel, TestLabel = train_test_split(Data, Label, test_size=(1-ratio), random_state=6)
    return TrnData, TrnLabel, TestData, TestLabel


def batch_format(Data, Label, batch_size):
    sample_nums = int(len(Data))
    S = Data.shape
    T = Label.shape
    Epoch = sample_nums // batch_size
    Data = Data[:Epoch*batch_size]
    Label = Label[:Epoch*batch_size]
    Temp = Data.reshape(Epoch, batch_size, S[1], S[2])  # Temp = (Epoch, batch_size, features, channel)
    Input = Temp
    Input_label = Label.reshape(Epoch, batch_size, T[1])
    dataLoader = [(Input[i], Input_label[i]) for i in range(Epoch)]

    return dataLoader


def DataLoad(Batch):
    trainData, trainLabel, test_X, test_Y = class_seperation(mat, 1)
    trainLoader = batch_format(trainData, trainLabel, Batch)

    return trainLoader, {'data': test_X, 'target': test_Y}
