import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import seaborn as sns
# # %matplotlib inline

# np.random.seed(2)

from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
# import itertools

from keras.utils.np_utils import to_categorical 
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
# from keras.optimizers import RMSprop
# from keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import ReduceLROnPlateau


sns.set(style='white', context='notebook', palette='deep')

# load the data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

Y_train = train["label"]
# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 
# free some space
del train 

# normalize the data
X_train /= 255
test /= 255

# reshape => images are 28x28, reshape in 28x28x1 3D matrix
# image as 3D with channels-last [rows][cols][channels]
# for gray scaled is just 1 channel
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

# convert to one-hot-encoding
# label encoding => one-hot vectors
# 4 => [0,0,0,0,1,0,0,0,0]
Y_train = to_categorical(Y_train, num_classes=10)

# use 10% of data for testing an d90% fro training
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)



