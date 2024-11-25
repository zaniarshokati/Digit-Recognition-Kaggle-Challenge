import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, PReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import itertools

def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    # Load data
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    # Split features and labels
    X_train = train.drop(columns=['label'])
    Y_train = to_categorical(train['label'])
    X_test = test.values

    # Normalize data
    X_train = X_train.values.reshape(-1, 28, 28, 1) / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

    # Split training data for validation
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=2)

    return X_train, X_val, Y_train, Y_val, X_test

def build_model():
    """Build and compile the CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation=PReLU()),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation=PReLU()),
        BatchNormalization(),
        Conv2D(32, (5, 5), padding='same', activation=PReLU()),
        MaxPooling2D(pool_size=2),
        BatchNormalization(),
        Dropout(0.4),
        Conv2D(64, (3, 3), activation=PReLU()),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation=PReLU()),
        BatchNormalization(),
        Conv2D(64, (5, 5), padding='same', activation=PReLU()),
        MaxPooling2D(pool_size=2),
        BatchNormalization(),
        Dropout(0.4),
        Conv2D(128, (4, 4), activation=PReLU()),
        BatchNormalization(),
        Flatten(),
        Dropout(0.4),
        Dense(10, activation='softmax', kernel_regularizer=l2(0.01))
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_model(model, X_train, Y_train, X_val, Y_val):
    """Train the model and evaluate it on the validation set."""
    datagen = ImageDataGenerator(
        rotation_range=10, zoom_range=0.1,
        width_shift_range=0.1, height_shift_range=0.1)

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint("best_model.h5", save_best_only=True),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.00001)
    ]

    datagen.fit(X_train)
    history = model.fit(datagen.flow(X_train, Y_train, batch_size=32),
                        epochs=100, validation_data=(X_val, Y_val),
                        callbacks=callbacks)

    # Evaluate the model
    _, accuracy = model.evaluate(X_val, Y_val)
    print(f"Validation Accuracy: {accuracy}")

    return history

def plot_metrics(history):
    """Plot training and validation metrics."""
    fig, ax = plt.subplots(2, 1, figsize=(8, 12))
    ax[0].plot(history.history['loss'], label="Training loss")
    ax[0].plot(history.history['val_loss'], label="Validation loss")
    ax[0].set_title('Loss Metrics')
    ax[0].legend()

    ax[1].plot(history.history['accuracy'], label="Training accuracy")
    ax[1].plot(history.history['val_accuracy'], label="Validation accuracy")
    ax[1].set_title('Accuracy Metrics')
    ax[1].legend()

    plt.savefig('metrics.png')
    plt.close()

def main():
    """Run the data preprocessing, model training, and evaluation."""
    X_train, X_val, Y_train, Y_val, X_test = load_and_preprocess_data()
    model = build_model()
    history = train_and_evaluate_model(model, X_train, Y_train, X_val, Y_val)
    plot_metrics(history)

if __name__ == "__main__":
    main()
