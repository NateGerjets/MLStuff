import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
from sklearn.utils import compute_class_weight
from sklearn.metrics import classification_report

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
num_classes = 75
epochs = 12

(x_train, y_train)= loadData("train", "datasets")
(x_test, y_test) =loadData("test","datasets")

x2_train = np.load('datasets/k49-train-imgs.npz')['arr_0']
y2_train = np.load('datasets/k49-train-labels.npz')['arr_0'] + 26
x2_test = np.load('datasets/k49-test-imgs.npz')['arr_0']
y2_test = np.load('datasets/k49-test-labels.npz')['arr_0'] + 26




nImages, width, height = x2_train.shape
x2_train = x2_train.reshape( [ nImages, width, height, 1 ] )

nImages, width, height = x2_test.shape
x2_test = x2_test.reshape( [ nImages, width, height, 1 ] )


x_train = np.append(x_train, x2_train, axis = 0)
x_test = np.append(x_test, x2_test, axis = 0)
y_train = np.append(y_train, y2_train, axis = 0)
y_test = np.append(y_test, y2_test, axis=0)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = .20, stratify=y_train)

weights = compute_class_weight('balanced', np.unique(y_train), y_train)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_test /= 255
x_val /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)




input_shape = (28,28, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
print(len(model.layers))

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
          
y_pred = model.predict_classes(x_test)
Y_test = np.argmax(y_test, axis=1)
print(classification_report(Y_test, y_pred))

eng_eng = 0
jap_eng = 0
eng_jap = 0
jap_jap = 0

for i,x in enumerate(Y_test):
    if x < 26: #English
        if y_pred[i] < 26:
            eng_eng += 1
        else:
            eng_jap += 1
    else:
        if y_pred[i] < 26:
            jap_eng += 1
        else:
            jap_jap += 1

print('Confusion Matrix:    as English     as Japanese')
print('English              ' + str(eng_eng) + '      ' + str(eng_jap))
print('Japanese              ' + str(jap_eng) + '      ' + str(jap_jap))
score = model.evaluate(x_test, y_test, verbose=0)
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
plt.legend(labels = ['Train Accuracy', 'Validation Accuracy'], loc = 4)
plt.figure(2)
plt.plot(range(1, epochs+1), history.loss,range(1, epochs+1), history.val_loss)
plt.xlabel('Accuracy')
plt.ylabel('Loss')
plt.legend(labels = ['Train Loss', 'Validation Loss'])
plt.show()