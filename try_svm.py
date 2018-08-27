'''
Created on 2018/08/28

@author: code-440
'''
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib import cm
import math 

digits = datasets.load_digits()  # load data set

train_data_start = 0
train_data_end = -30

test_data_start = train_data_end

train_data = digits.data[:train_data_end]
train_label = digits.target[:train_data_end]

test_data = digits.data[test_data_start:]
test_label = digits.target[test_data_start:]
test_images = digits.images[test_data_start:]

if __name__ == '__main__':
    clf = svm.SVC(gamma=0.001, C=100.)
    clf.fit(train_data, train_label)
    
    list_result = clf.predict(test_data)
    
    for index, num in enumerate(list_result):
        # math.ceil1で切り上げしてNx5枚で並べる
        plt.subplot(math.ceil(len(list_result)/5), 5, index + 1)
        plt.axis('off')
        plt.imshow(test_images[index], cmap=cm.gray_r, interpolation='nearest')
        title = "res: " + str(num) + " ans: " + str(test_label[index])
        plt.title(title)
    plt.show()    