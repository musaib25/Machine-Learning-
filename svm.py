# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 09:24:27 2022

@author: Raisa
"""

import numpy as np
from sklearn.datasets import load_digits
dataset = load_digits()
print(dataset.data)
print(dataset.target)

print(dataset.data.shape)
print(dataset.images.shape)

dataimagelength = len(dataset.images)
print(dataimagelength)

n=1795 #no of samples put of 1797
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(dataset.images[n])
#plt.show()

dataset.images[n]
X = dataset.images.reshape((dataimagelength,-1))
Y = dataset.target
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train, Y_test = train_test_split(X, Y,test_size=0.25,random_state=0)
print(X_train.shape)
print(X_test.shape)

from sklearn import svm
model = svm.SVC(kernel='linear')
model.fit(X_train,Y_train)

n=15
result = model.predict(dataset.images[n].reshape((1,-1)))
plt.imshow(dataset.images[n], cmap=plt.cm.gray_r,interpolation='nearest')
print(result)
print("\n")
plt.axis('off')
plt.title('%i' %result)
plt.show()


