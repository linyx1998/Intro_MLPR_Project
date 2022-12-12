import numpy as np
from sklearn import svm
from read import read_data

from skimage.feature import hog
from PIL import Image
from skimage import feature as ft
from skimage import data, exposure
import sys
import os
import time
import matplotlib.pyplot as plt


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



def CreatData():
    x = []
    y = []
    for i in range(1, 6):
        batch_path = '../cnn/cifar10_data/cifar-10-batches-py/data_batch_%d' % (i)
        batch_dict = unpickle(batch_path)
        train_batch = batch_dict[b'data']
        train_labels = np.array(batch_dict[b'labels'])
        x.append(train_batch)
        y.append(train_labels)

    traindata = np.concatenate(x)
    trainlabels = np.concatenate(y)
    test_dict = unpickle('../cnn/cifar10_data/cifar-10-batches-py/test_batch')
    testdata = test_dict[b'data']
    testlabels = np.array(test_dict[b'labels'])

    return traindata, trainlabels, testdata, testlabels

traindata1, trainlabels1, testdata1, testlabels1 = CreatData()

num_train = 10000
num_test = 5000

traindata = np.reshape(traindata1[:num_train], [num_train, 3, 32, 32])
trainlabels = trainlabels1[:num_train]
testdata = np.reshape(testdata1[:num_test], [num_test, 3, 32, 32])
testlabels = testlabels1[:num_test]
print(traindata.shape)


# extracting HOG input:[num,3,32,32],size means cell size , return data_hogfeature:[num,2916]
def hog_extraction(data, size):
    num = data.shape[0]
    data1_hogfeature = []
    for i in range(num):
        x = data[i]
        r = Image.fromarray(x[0])
        g = Image.fromarray(x[1])
        b = Image.fromarray(x[2])
        img = Image.merge("RGB", (r, g, b))
        gray = img.convert('L')
        gray_array = np.array(gray)

        hogfeature = ft.hog(gray_array, pixels_per_cell=(size, size), cells_per_block= (3, 3))
        data1_hogfeature.append(hogfeature)

    data_hogfeature = np.reshape(np.concatenate(data1_hogfeature), [num, -1])
    return data_hogfeature


size = 8


train_hogfeature = hog_extraction(traindata, size)

test_hogfeature = hog_extraction(testdata, size)
num_hogfeature = test_hogfeature.shape[1]
#
# print(train_hogfeature.shape)
for k in (['rbf']):
    for c in (0.0001, 0.001, 0.01, 0.1, 1, 10, 100):
        clf = svm.SVC(kernel=k, C=c).fit(train_hogfeature, trainlabels)
        print(k, c, "=> train acc:", np.mean(clf.predict(train_hogfeature) == trainlabels), \
              "test acc:", np.mean(clf.predict(test_hogfeature) == testlabels))
