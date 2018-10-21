# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.layers import Dot
import attention_keras_mask
from attention_keras_mask import Attention
from attention_keras_mask import Position_Embedding
import numpy as np
import pickle
from keras.utils import np_utils
import sys
from keras import backend as K
import tensorflow as tf

#python transformer_nsh_ver3 299 40
if __name__ == '__main__':
    OUTPUT_UNIT = 38
    max_features = 1000
    BATCHSIZE=16
    maxlen= 10
    #OUTPUT_UNIT = 299
    FILE_PATH='../data/'


    from keras.models import Model
    from keras.layers import *
    S_inputs_x = Input(shape=(maxlen,), dtype='float64')
    S_inputs_time = Input(shape=(maxlen,), dtype='float64')
    #S_inputs =  K.reshape(S_inputs,(1,-1,S_inputs.get_shape()[1]))

    X_array_1 = np.ones((maxlen,maxlen))
    X_array_2 = np.zeros((maxlen,maxlen))
    #print('aaaaaaaaaaaaa:',X_array_1.shape,X_array_2.shape)

    # X_array = np.concatenate([X_array_1,X_array_2])
    # T_array = np.concatenate([X_array_2,X_array_1])
    # print('aaaaaaaaaaaaaaaaaaaaaaaa:',type(S_inputs),type(X_array))
    # X_tensor = tf.convert_to_tensor(X_array)
    # T_tensor = tf.convert_to_tensor(T_array)

    # X_inputs = tf.matmul(S_inputs,X_array)
    # T_inputs = tf.matmul(S_inputs,T_array)
    #X_inputs = S_inputs[:,:maxlen]
    #T_input = S_inputs[:,maxlen:]
    # print('aaaaaaaaaaaaaaaaaaaaaaaaaaa:',type(X_inputs))
    embeddings = Embedding(max_features, 32)(S_inputs_x)
    embeddings = Position_Embedding()(embeddings) # 增加Position_Embedding能轻微提高准确率

    O_seq = Attention(8,16,S_inputs_time)([embeddings,embeddings,embeddings])
    O_seq = Flatten()(O_seq)
    #O_seq = GlobalAveragePooling1D()(O_seq)
    O_seq = Dropout(0.2)(O_seq)
    O_seq = Dense(OUTPUT_UNIT, activation='relu', name='dense')(O_seq)
    O_seq = Dropout(0.2)(O_seq)
    outputs = Dense(OUTPUT_UNIT,activation='softmax')(O_seq)

    model = Model(inputs=[S_inputs_x,S_inputs_time], outputs=outputs)
    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    print('Train...')

    X_file = open(FILE_PATH + 'all_datatrain_seq' + str(maxlen + 1) + '.pkl', 'rb')
    x_train_list = pickle.load(X_file)
    y_file = open(FILE_PATH + 'all_labeltrain_seq' + str(maxlen + 1) + '.pkl', 'rb')
    y_train_list= pickle.load(y_file)

    X_file = open(FILE_PATH + 'all_datatest_seq' + str(maxlen + 1) + '.pkl', 'rb')
    x_test_list = pickle.load(X_file)
    y_file = open(FILE_PATH + 'all_labeltest_seq' + str(maxlen + 1) + '.pkl', 'rb')
    y_test_list = pickle.load(y_file)

    X_file = open(FILE_PATH + 'all_timetrain_seq' + str(maxlen + 1) + '.pkl', 'rb')
    x_train_time_list = pickle.load(X_file)
    x_train_time = np.array(x_train_time_list)
    X_file = open(FILE_PATH + 'all_timetest_seq' + str(maxlen + 1) + '.pkl', 'rb')
    x_test_time_list = pickle.load(X_file)
    x_test_time = np.array(x_test_time_list)



    x_train = np.array(x_train_list)
    x_test = np.array(x_test_list)
    # x_train = np.concatenate((x_train, x_train_time), axis=0)
    # x_test = np.concatenate((x_test, x_test_time), axis=0)
    y_train = np_utils.to_categorical(y_train_list,OUTPUT_UNIT)
    y_test = np_utils.to_categorical(y_test_list,OUTPUT_UNIT)

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print('-------------------------')
    model.fit([x_train, x_train_time], y_train,epochs=20,batch_size=BATCHSIZE,validation_data=([x_test,x_test_time], y_test))

