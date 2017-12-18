from keras.optimizers import SGD, Adam, Nadam, RMSprop
from keras.models import Sequential,Model,load_model
from keras.layers.core import Dense, Activation,Dropout ,Flatten
from keras.layers import Embedding,Conv1D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence,one_hot,Tokenizer
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint,TensorBoard, ReduceLROnPlateau
from keras.applications import Xception
from keras import regularizers
from keras import backend as K
import keras
import numpy as np
import pandas as pd
import cv2
import os
import glob
import math
seed = 7
np.random.seed(seed)

def load_TrainingData(path):     #loads data , caluclate Mean & subtract it data, gets the COV. Matrix.
    D = pd.read_csv(path, sep='\t', header=0)
    feature_names  = np.array(list(D.columns.values))
    Y_train = np.array(list(D['Sentiment']));
    X_train = np.array(list(D['Phrase']))
    return  X_train, Y_train, feature_names

def load_TestingData(path):     #loads data , caluclate Mean & subtract it data, gets the COV. Matrix.
    D = pd.read_csv(path, sep='\t', header=0)
    X_test=D.ix[:,:3]
    return  X_test

def myfunction( x ):
    return sum(x)

X_train, Y_train, feature_names = load_TrainingData('./train.tsv')
X_test = load_TestingData('./test.tsv')
print '============================== Training data shapes =============================='
print 'X_train.shape is ', X_train.shape
print 'Y_train.shape is ',Y_train.shape

#masking
num_test = 10000
mask = range(num_test)
Y_Val = Y_train[:num_test]
X_Val = X_train[:num_test]

# X_Val = X_train[]
# Y_Val = Y_train
X_train = X_train[num_test:]
Y_train = Y_train[num_test:]

# X_train_words=[]
#
# # for i in xrang(num_test+1,X_train.shape[0])
# for i in xrange(num_test+1,10):
    # X_train_words = X_train_words | set(text_to_word_sequence(X_train[i]))


#
# encoded = one_hot(X_train, round(len(X_train_words)*1.3))
# print encoded

maxWordCount= 37
maxDictionary_size=5000

# create the tokenizer(s)
train_Tokenizer = Tokenizer()
Val_Tokenizer = Tokenizer()
# fit the tokenizer on the documents
train_Tokenizer.fit_on_texts(X_train)
Val_Tokenizer.fit_on_texts(X_Val)
train_Tokenizer_vocab_size = len(train_Tokenizer.word_index) + 1
Val_Tokenizer_vocab_size = len(train_Tokenizer.word_index) + 1

encoded_words = train_Tokenizer.texts_to_sequences(X_train)
encoded_words2 = Val_Tokenizer.texts_to_sequences(X_Val)


#padding all text to same size
X_Train_encodedPadded_words = sequence.pad_sequences(encoded_words, maxlen=maxWordCount)
X_Val_encodedPadded_words = sequence.pad_sequences(encoded_words2, maxlen=maxWordCount)
# X_test = sequence.pad_sequences(X_test, maxlen=maxWordCount)



# One Hot Encoding
Y_train = keras.utils.to_categorical(Y_train, 5)
Y_Val   = keras.utils.to_categorical(Y_Val, 5)


print Y_Val.shape
print Y_Val
print '-0-------'
print X_Train_encodedPadded_words[0]
print 'Featu  re are ',feature_names
print '============================== After extracting a validation set of '+ str(num_test)+' ============================== '
print '============================== Training data shapes =============================='
print 'X_train.shape is ', X_train.shape
print 'Y_train.shape is ',Y_train.shape
print '============================== Validation data shapes =============================='
print 'Y_Val.shape is ',Y_Val.shape
print 'X_Val.shape is ', X_Val.shape
print '============================== Test data shape =============================='
print 'X_test.shape is ', X_test.shape


learning_rate=0.0001
epochs = 50
batch_size = 128 #32
sgd = SGD(lr=learning_rate, nesterov=True, momentum=0.7, decay=1e-4)




print '============================== After padding all text to same size of '+ str(maxWordCount)+' =============================='
print '============================== Training data shapes =============================='
print 'X_train.shape is ', X_train.shape
print 'Y_train.shape is ',Y_train.shape
print '============================== Validation data shapes =============================='
print 'Y_Val.shape is ',Y_Val.shape
print 'X_Val.shape is ', X_Val.shape
print '============================== Test data shape =============================='
print 'X_test.shape is ', X_test.shape

#model
model = Sequential()

model.add(Embedding(maxDictionary_size, 32, input_length=maxWordCount)) #to change words to ints
# model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
 #hidden layers
# model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.5))

 #output layer
model.add(Dense(5, activation='softmax'))

# Compile model
# adam=Adam(lr=learning_rate, beta_1=0.7, beta_2=0.999, epsilon=1e-08, decay=0.0000001)

model.summary()
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

#
# #--------------------------
#
# datagen_test = ImageDataGenerator(
#     featurewise_center=True,
#     samplewise_center=False,
#     featurewise_std_normalization=True,
#     samplewise_std_normalization=False
#     )
#
# datagen_train = ImageDataGenerator(
#     featurewise_center=True,
#     samplewise_center=False,
#     featurewise_std_normalization=True,
#     samplewise_std_normalization=False,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)
#
tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/log_1', histogram_freq=0, write_graph=True, write_images=False)
checkpointer = ModelCheckpoint(filepath="./weights/weights_1.hdf5", verbose=1, save_best_only=True, monitor="val_loss")
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=0, verbose=1, mode='auto', cooldown=0, min_lr=1e-6)
#
# datagen_train.fit(X_train)
# datagen_test.fit(X_test)
#
# train_gen_trainData,  train_gen_testData
#
print ("=============================== Training =========================================")
#
# # uncommit this to train
# # tensorboard --logdir=./logs
#
# history = new_model.fit_generator(datagen_train.flow(X_train, Y_train,batch_size=batchSize), steps_per_epoch= X_train.shape[0] / batchSize ,
#                              epochs=epochs,validation_data=datagen_test,validation_steps =10000 / batchSize ,verbose=1,
#                              callbacks=[board,checkpointer, reduce_lr] )

history  = model.fit(X_Train_encodedPadded_words, Y_train, epochs = epochs, batch_size=batch_size, verbose=1,
                    validation_data=(X_Val_encodedPadded_words, Y_Val), callbacks=[tensorboard, reduce_lr])

print ("=============================== Predicting =========================================")

score = model.evaluate(X_Val, Y_Val, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
