import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, PReLU
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf


def data_Preprocessing():
    # load the data
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    # Drop 'label' column
    X_train = train.drop(labels = ["label"],axis = 1)
    Y_train = train["label"]
    X_test = test.values
    # free some space
    del train 

    # normalize the data
    # reshape => images are 28x28, reshape in 28x28x1 3D matrix
    # image as 3D with channels-last [rows][cols][channels]
    # for gray scaled is just 1 channel
    X_train = X_train.values.reshape(-1,28,28,1)/255.0
    X_test = X_test.reshape(-1,28,28,1)/255.0

    # convert to one-hot-encoding
    # label encoding => one-hot vectors
    # 4 => [0,0,0,0,1,0,0,0,0]
    Y_train = to_categorical(Y_train, num_classes=10)

    # use 10% of data for testing an d90% fro training
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
    return Y_train,X_train,X_val,Y_val,X_test

def model_Generator():
    model = Sequential()
    model.add(Conv2D(32, (3, 3),  input_shape=(28, 28, 1)))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3)))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5, 5), strides=2, padding='same'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64, (3, 3)))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3)))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), strides=2, padding='same'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (4, 4)))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(10, activation="softmax", kernel_regularizer= tf.keras.regularizers.l2(0.01)))
    return model

def plot_Loss_Acc(history):
    # Plot the loss and accuracy curves for training and validation 
    fig, ax = plt.subplots(2,1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    # fig.show()
    plt.savefig('loss_accuracy.png')
    plt.close()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1
    
    # plt.show()
    plt.savefig("errors.png")
    plt.close()

Y_train, X_train, X_val, Y_val,X_test = data_Preprocessing()
# Data Augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

model = model_Generator()

# initial
epochs = 2
batch_size = 32

# Compile the model
model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])

# Add early stopping, model checkpointing, and learning rate reduction
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint("best_model.h5", save_best_only=True),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.00001, verbose=1)
]

datagen.fit(X_train)

# Fit the model
history = model.fit(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val)
                              , callbacks=callbacks)

# Evaluate model on validation set
_, accuracy = model.evaluate(X_val,Y_val)
print(f"Validation Accuracy: {accuracy}")

plot_Loss_Acc(history)

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 

# Display some error results 

# Errors are difference between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
plot_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)

# predict results
prediction = model.predict(X_test)

# select the index with the maximum probability
Y_pred = np.argmax(prediction,axis = 1)

results = pd.Series(Y_pred,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission.csv",index=False)
