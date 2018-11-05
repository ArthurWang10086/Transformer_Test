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
import sys
from keras import backend as K
from hyperparams import Hyperparams as hp
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from util_runformimic import util_runformimic

SINGLE_ATTENTION_VECTOR = False
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, hp.maxlen))(a)  # this line is not useful. It's just to know which dimension is what.
    a = Dense(hp.maxlen, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = concatenate([inputs, a_probs], name='attention_mul')
    return output_attention_mul

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
    # S_inputs_x = Input(shape=(maxlen,), dtype='float64')

    #S_inputs =  K.reshape(S_inputs,(1,-1,S_inputs.get_shape()[1]))

    X_array_1 = np.ones((maxlen,maxlen))
    X_array_2 = np.zeros((maxlen,maxlen))


    S_inputs = Input(shape=(hp.maxlen,), dtype='int32')
    S_inputs_time = Input(shape=(maxlen,), dtype='float32')
    embeddings = Embedding(max_features, 32)(S_inputs)
    input_lstm = Bidirectional(LSTM(units=64, return_sequences=True))(embeddings)
    dropout_lstm = Dropout(0.2)(input_lstm)
    attention_mul = attention_3d_block(dropout_lstm)
    attention_mul = Flatten()(attention_mul)
    # attention_mul = tf.cast(attention_mul,tf.float32)
    # attention_mul = K.concatenate([attention_mul,S_inputs_time],axis=0)
    attention_mul = Concatenate(axis=-1)([attention_mul, S_inputs_time])
    # attention_mul = tf.concat([attention_mul,S_inputs_time],0)
    # dropout_attention = Dropout(0.2)(attention_mul)
    # dense = Dense(hp.output_unit, activation='relu', name='dense')(dropout_attention)
    dropout_dense = Dropout(0.2)(attention_mul)
    output = Dense(1, activation='sigmoid')(dropout_dense)
    model = Model(input=[S_inputs,S_inputs_time], output=[output])
    # if os.path.exists("model/lstm1.1.ALL.1.32.64.64.weights.014-0.9754.hdf5"):
    #     model_final.load_weights("model/lstm1.1.ALL.1.32.64.64.weights.014-0.9754.hdf5")
    model.compile(loss='mse',
                        optimizer='adam',
                        metrics=['mae','mse'])
    print(model.summary())


    # checkpoint
    # checkpoint = ModelCheckpoint(self.model_file + '.weights.{epoch:03d}-{val_f1_score:.4f}.hdf5', monitor='val_f1_score', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]
    print('Train...')

    if 'fold' in hp.FILE_PATH:
        util_runformimic(model)
        exit(0)
    else:
        pass
    print('Train...')

    X_file = open(FILE_PATH + 'all_datatrain_seq' + str(maxlen + 1) + '.pkl', 'rb')
    x_train_list = pickle.load(X_file)
    y_file = open(FILE_PATH + 'all_timetrain_label_seq' + str(maxlen + 1) + '.pkl', 'rb')
    y_train_list= pickle.load(y_file)

    X_file = open(FILE_PATH + 'all_datatest_seq' + str(maxlen + 1) + '.pkl', 'rb')
    x_test_list = pickle.load(X_file)
    y_file = open(FILE_PATH + 'all_timetest_label_seq' + str(maxlen + 1) + '.pkl', 'rb')
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
    y_train = np.array(y_train_list)
    y_test = np.array(y_test_list)
    # y_train = np.array([[1,0] if x==1 else [0,1] for x in y_train_list]) if OUTPUT_UNIT==2 else np_utils.to_categorical(np.array(y_train_list)-1,OUTPUT_UNIT)
    # y_test = np.array([[1,0] if x==1 else [0,1] for x in y_test_list]) if OUTPUT_UNIT==2 else np_utils.to_categorical(np.array(y_test_list)-1,OUTPUT_UNIT)

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print('-------------------------')
    model.fit([x_train, x_train_time], y_train,epochs=hp.epochs,batch_size=BATCHSIZE,validation_data=([x_test,x_test_time], y_test))

