import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt


def loadData( prefix, folder ):
    intType = np.dtype( 'int32' ).newbyteorder( '>' )
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile( folder + "/" + prefix + '-images-idx3-ubyte', dtype = 'ubyte' )
    magicBytes, nImages, width, height = np.frombuffer( data[:nMetaDataBytes].tobytes(), intType )
    data = data[nMetaDataBytes:].astype( dtype = 'float32' ).reshape( [ nImages, width, height, 1 ] )

    labels = np.fromfile( folder + "/" + prefix + '-labels-idx1-ubyte',
                          dtype = 'ubyte' )[2 * intType.itemsize:] -1

    return data, labels
    
def oneHot(classification):
    # emulates the functionality of tf.keras.utils.to_categorical( y )
    hotEncoding = np.zeros( [ len( classification ), 
                              np.max( classification ) + 1 ] )
    hotEncoding[ np.arange( len( hotEncoding ) ), classification ] = 1
    return hotEncoding

batch_size = 128
num_classes = 26
epochs = 2

(x_train, y_train)= loadData("train", "datasets")
(x_test, y_test) =loadData("test","datasets")
print(len(y_train))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = .2, stratify=y_train)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)




input_shape = (28,28, 1)

model = Sequential()
model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
model.add(Conv2D(128, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.loss = []
        self.val_loss = []
        self.val_acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

history = AccuracyHistory()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_val, y_val),
          callbacks=[history])
score = model.evaluate(x_test, y_test, verbose=0)
y_pred = model.predict_classes(x_test)
Y_test = np.argmax(y_test, axis=1)


print(classification_report(Y_test, y_pred))
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(history.acc)
print(history.loss)
print(history.val_acc)
print(history.val_loss)
plt.figure(1)
plt.plot(range(1, epochs+1), history.acc, range(1, epochs+1), history.val_acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(labels = ['Test Accuracy', 'Validation Accuracy'])
plt.figure(2)
plt.plot(range(1, epochs+1), history.loss,range(1, epochs+1), history.val_loss)
plt.xlabel('Accuracy')
plt.ylabel('Loss')
plt.legend(labels = ['Test Accuracy', 'Validation Accuracy'])
plt.show()