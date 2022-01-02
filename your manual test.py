#
#
# import tensorflow as tf
# from tensorflow.keras import models, datasets, layers, optimizers
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
#
#
# (X_train, Y_train), (X_test, Y_test) = datasets.cifar10.load_data()
# print(X_train.shape)
# print(X_test.shape)
# datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
# datagen.fit(X_train)
#
#
# print(X_train.shape)
# print(X_test.shape)
# classes = 10
# input_shape = (32,32,3)
# def model(classes, input_shape):
#     model = models.Sequential()
#     model.add(layers.Convolution2D(32,(3,3), activation="relu", input_shape = input_shape, padding="same"))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Convolution2D(32,(3,3), activation="relu", padding="same"))
#     model.add(layers.BatchNormalization())
#     model.add(layers.MaxPooling2D(pool_size=(2,2)))
#     model.add(layers.Dropout(0.2))
#
#
#     model.add(layers.Convolution2D(64,(3,3), activation="relu", padding="same"))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Convolution2D(64,(3,3), activation="relu", padding="same"))
#     model.add(layers.BatchNormalization())
#     model.add(layers.MaxPooling2D(pool_size=(2,2)))
#     model.add(layers.Dropout(0.3))
#
#     model.add(layers.Convolution2D(128,(3,3), activation="relu", padding="same"))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Convolution2D(128,(3,3), activation="relu", padding="same"))
#     model.add(layers.BatchNormalization())
#     model.add(layers.MaxPooling2D(pool_size=(2,2)))
#     model.add(layers.Dropout(0.4))
#
#     model.add(layers.Flatten())
#     model.add(layers.Dense(10, activation = "softmax"))
#     return model
# #
# X_train = X_train.reshape((50000, 32,32,3))
# X_test = X_test.reshape((10000, 32,32,3))
# X_train = X_train/255
# X_test = X_test/255
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
#
# Y_train = tf.keras.utils.to_categorical(Y_train, 10)
# Y_test = tf.keras.utils.to_categorical(Y_test, 10)
# #
# model = model(10, input_shape=input_shape)
# model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
# model.summary()
#
# callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs')]
#
# model.fit_generator(datagen.flow(X_train, Y_train,
# batch_size=64), epochs=20,
# verbose=1, validation_data=(X_test, Y_test),
# callbacks=callbacks)
# score = model.evaluate(X_test, Y_test, verbose=1)
# print("\nTest score:", score[0])
# print('Test accuracy:', score[1])
#
# model.save('model.h5')
# # test
# scores = model.evaluate(X_test, Y_test, batch_size=128, verbose=1)
# print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))



##############

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks

data = datasets.cifar10.load_data()
(X_train, Y_train),(X_test, Y_test) = data
print(len(X_train))

X_train = X_train/255
X_test = X_test/255

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

Y_train = tf.keras.utils.to_categorical(Y_train)
Y_test = tf.keras.utils.to_categorical(Y_test)

num_classes = 10
epochs = 10
VERBOSE = 1

model = models.Sequential()

print(X_train.shape)
print(X_test.shape)
classes = 10
input_shape = (32,32,3)
def model(classes, input_shape):
    model = models.Sequential()
    model.add(layers.Convolution2D(32,(3,3), activation="relu", input_shape = input_shape, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Convolution2D(32,(3,3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.2))


    model.add(layers.Convolution2D(64,(3,3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Convolution2D(64,(3,3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Convolution2D(128,(3,3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Convolution2D(128,(3,3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.4))

    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation = "softmax"))
    return model

model = model(10,input_shape)
print(model.summary())

model.compile(optimizer="Adam", loss = "categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, Y_train, epochs = 10, validation_split=0.1, use_multiprocessing=True, batch_size = 32)

model.save("simple_conv.h5")