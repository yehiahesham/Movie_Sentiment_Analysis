from keras.models import Model
from keras.optimizers import SGD, Adam, Nadam, RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout ,Flatten
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.constraints import maxnorm
from keras.models import load_model
from keras.callbacks import ModelCheckpoint,TensorBoard, ReduceLROnPlateau
from keras.applications import Xception
import keras
from keras import backend as K
import numpy as np
import cv2
import os
import glob
import math

img_size = 299 #224 #299
model = Sequential()
model.add(Dense(8, input_shape=(kX_train2.shape[1:])))
predictions = Dense(200, activation='softmax', name='predictions')(output_model)

new_model = Model(input=model.input, output=predictions)
new_model.summary()



# Loading best weights
new_model.load_weights("./weights/weights_3.hdf5")



sgd = SGD(lr=learning_rate, nesterov=True, momentum=0.7, decay=1e-4)
new_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

#--------------------------

datagen_test = ImageDataGenerator(
    featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=False
    )

datagen_train = ImageDataGenerator(
    featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

board = keras.callbacks.TensorBoard(log_dir='./logs/log_1', histogram_freq=0, write_graph=True, write_images=False)
checkpointer = ModelCheckpoint(filepath="./weights/weights_1.hdf5", verbose=1, save_best_only=True, monitor="val_loss")
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=0, verbose=1, mode='auto', cooldown=0, min_lr=1e-6)

datagen_train.fit(X_train)
datagen_test.fit(X_test)

train_gen_trainData,  train_gen_testData

print ("=============================== Training =========================================")

# uncommit this to train
# tensorboard --logdir=./logs

history = new_model.fit_generator(datagen_train.flow(X_train, Y_train,batch_size=batchSize), steps_per_epoch= X_train.shape[0] / batchSize ,
                             epochs=epochs,validation_data=datagen_test,validation_steps =10000 / batchSize ,verbose=1,
                             callbacks=[board,checkpointer, reduce_lr] )

#print ("=============================== Predicting =========================================")
