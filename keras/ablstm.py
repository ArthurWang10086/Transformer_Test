#!/usr/bin/python
# -*- coding:utf-8 -*-


__author__ = "Qian Zhong"
__copyright__ = ""
__version__ = "1.0.0"
__email__ = "zhongqian@corp.netease.com"
__phone__ = "18758090359"
__description__ = "通用时间预测模型训练"
__usage1__ = "python3 ablstm_nsh.py  ALL 100000 maxlen(40) 32 64 64 38"
#python3 ablstm_nsh.py  ALL 100000 maxlen embedding_size_logid embedding_size_logdesignid lstmsize dense_size

FILE_PATH='/home/zhongqian/yuyi-prediction-dataset/bookorder/data_3/'

import sys
import os
import gc
import random
import linecache
import numpy as np
from keras.utils import np_utils
import pickle
#D
#n3 ablstm_nsh.py  ALL 100000 40 32 64 64 38
import keras.backend as K
from time import time
from keras.preprocessing import sequence
from keras.layers import Input, Embedding, LSTM, Dense, merge, Bidirectional, Dropout, Flatten,concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.layers.core import *

# np.random.seed(7)
max_features=1000
TEST_SIZE = 0.2
BATCH_SIZE = 16
DROPOUT_SIZE = 0.2
MAX_LOGID = 10002
MAX_LOGDESIGNID = 10002
EPOCHS = 100
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False


def f1_score(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0
    # How many selected items are relevant?
    precision = c1 / c2
    # How many relevant items are selected?
    recall = c1 / c3
    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def precision(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0
    # How many selected items are relevant?
    precision = c1 / c2
    return precision


def recall(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0
    # How many relevant items are selected?
    recall = c1 / c3
    return recall




class xiaosanhuan_lstm_attention_train:
    def __init__(self, normal_file, waigua_file, model_file, maxlen):
        self.normal_file = normal_file     # 正常玩家序列文件
        self.waigua_file = waigua_file      # 异常玩家序列文件
        self.model_file = model_file        # 模型存储文件
        self.logid_data = list()                   # 大类事件序列
        self.logid_train = list()                   # 大类事件序列训练集
        self.logid_test = list()                    # 大类事件序列测试集
        self.logdesignid_data = list()        # 大小类组合事件序列
        self.logdesignid_train = list()        # 大小类组合事件序列训练集
        self.logdesignid_test = list()          # 大小类组合事件序列测试集
        # self.num_data = list()                   # 数量序列
        # self.num_train = list()                    # 数量序列训练集
        # self.num_test = list()                      # 数量序列测试集
        # self.grade_data = list()                  # 等级序列
        # self.grade_train = list()                   # 等级序列训练集
        # self.grade_test = list()                    # 等级序列测试集
        # self.time_data = list()                    # 时间间隔序列
        # self.time_train = list()                    # 时间间隔序列训练集
        # self.time_test = list()                      # 时间间隔序列测试集
        self.y_data = list()                          # 标签
        self.y_train = list()                         # 标签训练集
        self.y_test = list()                           # 标签测试集
        self.maxlen = maxlen                    # Pad Sequence最大长度
        self.max_logid = MAX_LOGID
        self.max_logdesignid = MAX_LOGDESIGNID
        self.test_size = TEST_SIZE
        self.batch_size = BATCH_SIZE
        self.embedding_size_logid = int(sys.argv[4])
        self.embedding_size_logdesignid = int(sys.argv[5])
        self.embedding_dropout_size = DROPOUT_SIZE
        self.lstm_size = int(sys.argv[6])
        self.lstm_dropout_size = DROPOUT_SIZE
        self.dense_size = int(sys.argv[7])
        self.dense_dropout_size = DROPOUT_SIZE
        self.attention_dropout_size = DROPOUT_SIZE

        # 加载数据
    def load_data(self):
        #X_file = open(FILE_PATH + 'all_data_seq'+str(self.maxlen+1)+'.pkl', 'rb')
        #X = pickle.load(X_file)
        #y_file = open(FILE_PATH + 'all_label_seq'+str(self.maxlen+1)+'.pkl', 'rb')
        #Y = pickle.load(y_file)

        #x_train_list = X[:int(len(X)*0.7)]
        #x_test_list = X[int(len(X)*0.7):]
        #y_train_list = Y[:int(len(X)*0.7)]
        #y_test_list = Y[int(len(X)*0.7):]


        X_file = open('all_datatrain_seq4.pkl', 'rb')
        x_train_list = pickle.load(X_file)
        y_file = open('all_labeltrain_seq4.pkl', 'rb')
        y_train_list= pickle.load(y_file)

        X_file = open('all_datatest_seq4.pkl', 'rb')
        x_test_list = pickle.load(X_file)
        y_file = open('all_labeltest_seq4.pkl', 'rb')
        y_test_list = pickle.load(y_file)
        
        
        x_train = np.array(x_train_list)
        x_test = np.array(x_test_list)

        self.y_train = np_utils.to_categorical(y_train_list, self.dense_size)
        self.y_test = np_utils.to_categorical(y_test_list, self.dense_size)

        print(len(x_train), 'train sequences')
        print(len(x_test), 'test sequences')

        print('Pad sequences (samples x time)')
        self.x_train = sequence.pad_sequences(x_train, maxlen=self.maxlen)
        self.x_test = sequence.pad_sequences(x_test, maxlen=self.maxlen)

        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)





    def pad_sequence_and_reshape(self):
        self.logid_train = sequence.pad_sequences(self.logid_train, maxlen=self.maxlen)
        self.logid_test = sequence.pad_sequences(self.logid_test, maxlen=self.maxlen)
        self.logdesignid_train = sequence.pad_sequences(self.logdesignid_train, maxlen=self.maxlen)
        self.logdesignid_test = sequence.pad_sequences(self.logdesignid_test, maxlen=self.maxlen)
        # self.num_train = sequence.pad_sequences(self.num_train, maxlen=self.maxlen)
        # self.num_test = sequence.pad_sequences(self.num_test, maxlen=self.maxlen)
        # self.grade_train = sequence.pad_sequences(self.grade_train, maxlen=self.maxlen)
        # self.grade_test = sequence.pad_sequences(self.grade_test, maxlen=self.maxlen)
        # self.time_train = sequence.pad_sequences(self.time_train, maxlen=self.maxlen)
        # self.time_test = sequence.pad_sequences(self.time_test, maxlen=self.maxlen)

        # self.num_train = np.reshape(self.num_train, (len(self.num_train), self.maxlen, 1))
        # self.num_test = np.reshape(self.num_test, (len(self.num_test), self.maxlen, 1))
        # self.grade_train = np.reshape(self.grade_train, (len(self.grade_train), self.maxlen, 1))
        # self.grade_test = np.reshape(self.grade_test, (len(self.grade_test), self.maxlen, 1))
        # self.time_train = np.reshape(self.time_train, (len(self.time_train), self.maxlen, 1))
        # self.time_test = np.reshape(self.time_test, (len(self.time_test), self.maxlen, 1))

        # print(self.logid_train[0])
        # print(self.logdesignid_train[0])
        # print(self.num_train[0])
        # print(self.grade_train[0])
        # print(self.time_train[0])
        # print(self.y_train[0])


    def attention_3d_block(self, inputs):
        # inputs.shape = (batch_size, time_steps, input_dim)
        input_dim = int(inputs.shape[2])
        a = Permute((2, 1))(inputs)
        a = Reshape((input_dim, self.maxlen))(a)  # this line is not useful. It's just to know which dimension is what.
        a = Dense(self.maxlen, activation='softmax')(a)
        if SINGLE_ATTENTION_VECTOR:
            a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
            a = RepeatVector(input_dim)(a)
        a_probs = Permute((2, 1), name='attention_vec')(a)
        output_attention_mul = concatenate([inputs, a_probs], name='attention_mul')
        return output_attention_mul


    def model_train(self):
        print('Building model...')
        S_inputs = Input(shape=(self.maxlen,), dtype='int32')
        embeddings = Embedding(max_features, 32)(S_inputs)
        input_lstm = Bidirectional(LSTM(units=self.lstm_size, return_sequences=True))(embeddings)
        dropout_lstm = Dropout(self.lstm_dropout_size)(input_lstm)
        attention_mul = self.attention_3d_block(dropout_lstm)
        attention_mul = Flatten()(attention_mul)
        dropout_attention = Dropout(self.attention_dropout_size)(attention_mul)
        dense = Dense(self.dense_size, activation='relu', name='dense')(dropout_attention)
        dropout_dense = Dropout(self.dense_dropout_size)(dense)
        output = Dense(self.dense_size, activation='softmax', name='output')(dropout_dense)
        model_final = Model(input=S_inputs, output=[output])
        # if os.path.exists("model/lstm1.1.ALL.1.32.64.64.weights.014-0.9754.hdf5"):
        #     model_final.load_weights("model/lstm1.1.ALL.1.32.64.64.weights.014-0.9754.hdf5")
        model_final.compile(loss='categorical_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy',recall, f1_score])
        print(model_final.summary())


        # checkpoint
       # checkpoint = ModelCheckpoint(self.model_file + '.weights.{epoch:03d}-{val_f1_score:.4f}.hdf5', monitor='val_f1_score', verbose=1, save_best_only=True, mode='max')
       # callbacks_list = [checkpoint]
        print('Train...')
        model_final.fit(self.x_train, self.y_train,
                        batch_size=self.batch_size,
                        epochs=EPOCHS,
                        validation_data=(self.x_test, self.y_test))


    def run(self):
        self.load_data()
        self.pad_sequence_and_reshape()
        self.model_train()


if __name__ == '__main__':

    start = time()
    # 参数1：数据集名称
    # 参数2：1 / 2 / 3
    # 参数3：序列最大长度
    # 参数4：大小类组合事件Embedding层节点数
    # 参数5：双向LSTM层节点数
    # 参数6：全连接层节点数
   # waigua_file = 'waigua/' + sys.argv[1] + '_date_seq_all/' + sys.argv[2] + '.csv'
   # normal_file = 'normal/' + sys.argv[1] + '_date_seq_all/' + sys.argv[2] + '.csv'
   # model_file = 'model/lstm_attention2.2.' + sys.argv[1] + '.' + sys.argv[2] + '.' + sys.argv[4] + '.' + sys.argv[5] + '.' + sys.argv[6] + '.' + sys.argv[7]
    maxlen = int(sys.argv[3])

    waigua_file = ''
    normal_file =''
    model_file =''


    xiaosanhuan_lstm_attention_train_ins = xiaosanhuan_lstm_attention_train(normal_file, waigua_file, model_file, maxlen)
    xiaosanhuan_lstm_attention_train_ins.run()

    stop = time()
    print("模型训练时间: " + str(stop - start) + "秒")
