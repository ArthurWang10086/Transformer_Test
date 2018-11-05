# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.layers import Dot
import attention_keras
from attention_keras import Attention
from attention_keras import Position_Embedding
from point_process_layer import CustomPPLayer
import numpy as np
import pickle
from keras.utils import np_utils
from hyperparams import Hyperparams as hp
import sys
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

#python transformer_nsh_ver3 299 40
if __name__ == '__main__':
    OUTPUT_UNIT = hp.output_unit
    max_features = hp.vocab_size
    BATCHSIZE=hp.batch_size
    maxlen=hp.maxlen
    #OUTPUT_UNIT = 299
    FILE_PATH=hp.FILE_PATH

    config = tf.ConfigProto(intra_op_parallelism_threads=hp.intra_op_parallelism_threads,inter_op_parallelism_threads=hp.inter_op_parallelism_threads)
    set_session(tf.Session(config=config))


    from keras.models import Model
    from keras.layers import *
    S_inputs_x = Input(shape=(maxlen,), dtype='float64')
    S_inputs_time = Input(shape=(maxlen,), dtype='float64')
    #S_inputs =  K.reshape(S_inputs,(1,-1,S_inputs.get_shape()[1]))

    #X_array_1 = np.ones((maxlen,maxlen))
    #X_array_2 = np.zeros((maxlen,maxlen))
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

    outputs = CustomPPLayer()([S_inputs_x,S_inputs_time_label,O_seq])

    # O_seq = Dropout(0.2)(O_seq)
    # outputs = Dense(OUTPUT_UNIT,activation='softmax')(O_seq)

    model = Model(inputs=[S_inputs_x,S_inputs_time,S_inputs_time_label], outputs=outputs)
    # try using different optimizers and different optimizer configs
    model.compile(loss=None,optimizer='adam')

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

    X_file = open(FILE_PATH + 'all_timetrain_label_seq' + str(maxlen + 1) + '.pkl', 'rb')
    x_train_time_list = pickle.load(X_file)
    x_train_time_label = np.array(x_train_time_list)

    X_file = open(FILE_PATH + 'all_timetest_seq' + str(maxlen + 1) + '.pkl', 'rb')
    x_test_time_list = pickle.load(X_file)
    x_test_time = np.array(x_test_time_list)

    X_file = open(FILE_PATH + 'all_timetest_label_seq' + str(maxlen + 1) + '.pkl', 'rb')
    x_test_time_list = pickle.load(X_file)
    x_test_time_label = np.array(x_test_time_list)



    x_train = np.array(x_train_list)
    x_test = np.array(x_test_list)
    # x_train = np.concatenate((x_train, x_train_time), axis=0)
    # x_test = np.concatenate((x_test, x_test_time), axis=0)
    y_train = np.array([[1,0] if x==1 else [0,1] for x in y_train_list]) if OUTPUT_UNIT==2 else np_utils.to_categorical(np.array(y_train_list)-1,OUTPUT_UNIT)
    y_test = np.array([[1,0] if x==1 else [0,1] for x in y_test_list]) if OUTPUT_UNIT==2 else np_utils.to_categorical(np.array(y_test_list)-1,OUTPUT_UNIT)


    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print('-------------------------')
    model.fit([x_train, x_train_time], [y_train,x_train_time_label],epochs=20,batch_size=BATCHSIZE,validation_data=([x_test,x_test_time], [y_test,x_test_time_label]))

