from keras.optimizers import SGD, Adam, Nadam, RMSprop
from keras.models import Sequential,Model,load_model
from keras.layers import Embedding,Conv1D,MaxPooling1D
from keras.layers.core import Dense, Activation,Dropout ,Flatten
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence,one_hot,Tokenizer
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint,TensorBoard, ReduceLROnPlateau,EarlyStopping
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
    X_train = np.array(list(D['Phrase']))
    Y_train = np.array(list(D['Sentiment']));
    return  X_train, Y_train, feature_names

def load_TestingData(path):     #loads data , caluclate Mean & subtract it data, gets the COV. Matrix.
    D = pd.read_csv(path, sep='\t', header=0)
    X_test=np.array(list(D['Phrase']))
    X_test_PhraseID=np.array(list(D['PhraseId']))
    return  X_test,X_test_PhraseID

def shuffle_2(a, b): # Shuffles 2 arrays with the same order
    s = np.arange(a.shape[0])
    np.random.shuffle(s)
    return a[s], b[s]



X_train, Y_train, feature_names = load_TrainingData('./train.tsv')
X_test,X_test_PhraseID = load_TestingData('./test.tsv')
print '============================== Training data shapes =============================='
print 'X_train.shape is ', X_train.shape
print 'Y_train.shape is ',Y_train.shape


Tokenizer = Tokenizer()
Tokenizer.fit_on_texts(np.concatenate((X_train, X_test), axis=0))
# Tokenizer.fit_on_texts(X_train)
Tokenizer_vocab_size = len(Tokenizer.word_index) + 1
print "Vocab size",Tokenizer_vocab_size

#masking
num_test = 32000
mask = range(num_test)

Y_Val = Y_train[:num_test]
Y_Val2 = Y_train[:num_test]
X_Val = X_train[:num_test]


X_train = X_train[num_test:]
Y_train = Y_train[num_test:]


maxWordCount= 60
maxDictionary_size=Tokenizer_vocab_size




encoded_words = Tokenizer.texts_to_sequences(X_train)
encoded_words2 = Tokenizer.texts_to_sequences(X_Val)
encoded_words3 = Tokenizer.texts_to_sequences(X_test)


#padding all text to same size
X_Train_encodedPadded_words = sequence.pad_sequences(encoded_words, maxlen=maxWordCount)
X_Val_encodedPadded_words = sequence.pad_sequences(encoded_words2, maxlen=maxWordCount)
X_test_encodedPadded_words = sequence.pad_sequences(encoded_words3, maxlen=maxWordCount)

# One Hot Encoding
Y_train = keras.utils.to_categorical(Y_train, 5)
Y_Val   = keras.utils.to_categorical(Y_Val, 5)


#shuffling the traing Set
shuffle_2(X_Train_encodedPadded_words,Y_train)

print 'Featu are ',feature_names
print '============================== After extracting a validation set of '+ str(num_test)+' ============================== '
print '============================== Training data shapes =============================='
print 'X_train.shape is ', X_train.shape
print 'Y_train.shape is ',Y_train.shape
print '============================== Validation data shapes =============================='
print 'Y_Val.shape is ',Y_Val.shape
print 'X_Val.shape is ', X_Val.shape
print '============================== Test data shape =============================='
print 'X_test.shape is ', X_test.shape





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
# model.add(Dropout(0.5))
# model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
 #hidden layers
model.add(LSTM(10))
# model.add(Flatten())
model.add(Dropout(0.6))
model.add(Dense(1200, activation='relu',W_constraint=maxnorm(1)))
# model.add(Dropout(0.6))
model.add(Dense(500, activation='relu',W_constraint=maxnorm(1)))

# model.add(Dropout(0.5))
 #output layer
model.add(Dense(5, activation='softmax'))

# Compile model
# adam=Adam(lr=learning_rate, beta_1=0.7, beta_2=0.999, epsilon=1e-08, decay=0.0000001)

model.summary()


learning_rate=0.0001
epochs = 60
batch_size = 32 #32
sgd = SGD(lr=learning_rate, nesterov=True, momentum=0.7, decay=1e-4)
Nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(loss='categorical_crossentropy', optimizer=Nadam, metrics=['accuracy'])


tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/log_25', histogram_freq=0, write_graph=True, write_images=False)
checkpointer = ModelCheckpoint(filepath="./weights/weights_25.hdf5", verbose=1, save_best_only=True, monitor="val_loss")
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=0, verbose=1, mode='auto', cooldown=0, min_lr=1e-6)
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)

# Loading best weights
# model.load_weights("./weights/weights_19.hdf5")

print ("=============================== Training =========================================")

# # uncommit this to train
# # tensorboard --logdir=./logs

history  = model.fit(X_Train_encodedPadded_words, Y_train, epochs = epochs, batch_size=batch_size, verbose=1,
                    validation_data=(X_Val_encodedPadded_words, Y_Val), callbacks=[tensorboard, reduce_lr,checkpointer,earlyStopping])

print ("=============================== Score =========================================")

scores = model.evaluate(X_Val_encodedPadded_words, Y_Val, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
print ("=============================== Predicting =========================================")

f = open('Submission.csv', 'w')
f.write('PhraseId,Sentiment\n')


# predictions = model.predict(X_test_encodedPadded_words)
predicted_classes = model.predict_classes(X_test_encodedPadded_words, batch_size=batch_size, verbose=1)
# print np.sum(predicted_classes==Y_Val2)/(1.0*Y_Val2.shape[0])
# print predicted_classes
# preds = new_model.predict(x)
# print predicted_classes
for i in range(0,X_test_PhraseID.shape[0]):
    # pred =np.argmax(predictions[i])
    f.write(str(X_test_PhraseID[i])+","+str(predicted_classes[i])+'\n')
    # print predictions[i],"=>",pred

f.close()
