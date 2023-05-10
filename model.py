from tensorflow import keras
from keras.layers import Dense, BatchNormalization, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, InputLayer
from keras.models import Sequential
from keras.optimizers import Adam

def build_model(input_shape, num_classes = 8):
    """
    Parameters
    ----------
    input_shape : tuple of int
      The shape of the input tensor, e.g. (height, width, channels).

    Returns
    -------
    model : keras.Model
      The compiled baseline convolutional neural network model.

    """
    # Build the model

    # Initialize the model
    model = Sequential()

    # Add the first convolutional layer
    model.add(Conv2D(16, (3, 3), strides=(2, 2), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    # Add the second convolutional layer
    model.add(Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding="same"))

    # Add the third convolutional layer
    model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    # Flatten the output of the convolutional layers
    model.add(Flatten())
    model.add(BatchNormalization())

    # Add the first dense layer
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))

    # Add the second dense layer
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))

    # Add the output layer
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    optimizer = Adam(learning_rate=0.0002)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def dataset_split(dataset, train_size = 1000, val_size = 300):
    features_train = dataset[:train_size]
    features_val = dataset[train_size: train_size + val_size]
    features_test = dataset[train_size + val_size:]

    return features_train, features_val, features_test

def label_split(source_locations, train_size = 1000, val_size = 300):
    labels = source_locations[:, 0]
    labels_train = labels[:train_size] / 45
    labels_val = labels[train_size: train_size + val_size] / 45
    labels_test = labels[train_size + val_size:] / 45

    return labels_train, labels_val, labels_test


import numpy as np
import matplotlib.pyplot as plt

def plot_loss(history, title, figsize=(10, 5)):
    """
    Plot the training and validation loss and accuracy.

    Parameters
    ----------
    history : keras.callbacks.History
      The history object returned by the `fit` method of a Keras model.

    Returns
    -------
    None
    """
    plt.rcParams.update({'font.size': 18})
    plt.rc('axes', labelsize=18)  
    plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=16)

    # Plot the training and validation loss side by side
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # Plot the training and validation loss
    ax[0].plot(history.history['loss'], label='train')
    ax[0].plot(history.history['val_loss'], label='val')
    ax[0].set_xlabel('Epoch',)
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    # Plot the training and validation accuracy
    ax[1].plot(history.history['accuracy'], label='train')
    ax[1].plot(history.history['val_accuracy'], label='val')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_prediction_analysis(y_pred, y_true, title=''):
    plt.rc('figure', labelsize=18)

    # num_centers, 4 (all, correct, front back, adjacent)
    count = np.zeros((8, 4))
    
    fb = {0:4, 1:3, 2:2, 3:1, 4:0, 5:7, 6:6, 7:5}
    
    for i in range(len(y_pred)):
        count[y_true[i]][0] += 1
        
        if y_pred[i] == y_true[i]:
            count[y_true[i]][1] += 1
        elif np.abs(y_pred[i]-y_true[i]) < 2 or np.abs(y_pred[i] - y_true[i])==7:
            count[y_true[i]][2] += 1
        elif fb[y_pred[i]]==y_true[i]:
            count[y_true[i]][3] +=1
    
    plt.figure(figsize=(16, 6))
    plt.rcParams.update({'font.size': 12})
    labels = ['Count', 'Correct', 'Adjacent', 'Front-Back']
    colors = ['b', 'g', 'y', 'r']
    width=0.15

    x_axis = ['0','45','90','135', '180', '225', '270', '315']

    x = np.arange(8)
    for i in range(4):
        bar = plt.bar(x + (i-1.5)*width, count[:, i], color=colors[i],
                width=width, label=labels[i])
        plt.bar_label(bar)
    plt.rc('legend', fontsize=12)
    plt.legend()
    plt.xticks(x, x_axis)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return count


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, normalize = 'true', title=''):
    font_size = 16
    matrix_label = np.array(['0', '45', '90', '135', '180', '225', '270', '315'])

    cm = confusion_matrix(y_true, y_pred, normalize = 'true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=matrix_label)

    plt.rc('xtick', labelsize=font_size) 
    plt.rc('ytick', labelsize=font_size) 
    disp.plot()
    disp.ax_.set_title(title)

    fig = disp.ax_.get_figure() 
    fig.set_figwidth(18)
    fig.set_figheight(18)  
    plt.show(disp)
    plt.tight_layout()