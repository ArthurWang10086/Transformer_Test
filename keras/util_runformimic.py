# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.preprocessing import sequence
from keras.datasets import imdb
import attention_keras
from attention_keras import Attention
from attention_keras import Position_Embedding
import numpy as np
from hyperparams import Hyperparams as hp
import pickle
from keras.utils import np_utils

def getAcc(model,ttt,datatype):
    FOLD = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']
    y_pred_dataset_file = open(hp.FILE_PATH + FOLD[ttt] + '_%s.pkl'%(datatype), 'rb')
    y_pred_dataset = pickle.load(y_pred_dataset_file)
    y_pred_label_dataset = open(hp.FILE_PATH + FOLD[ttt] + '_%s_label.pkl'%datatype, 'rb')
    y_pred_label = pickle.load(y_pred_label_dataset)

    y_pred_dataset= np.array(y_pred_dataset)
    y_pred_dataset= sequence.pad_sequences(y_pred_dataset, maxlen=hp.maxlen)
    #y_pred_label = np_utils.to_categorical(y_pred_label,76)

    y_pred = model.predict(y_pred_dataset)
    y_pred_list = y_pred.tolist()

    true_count=0
    #    print(y_pred_list)
    #   print(y_pred_label)

    for k in range(len(y_pred_list)):
        leibie=y_pred_list[k].index(max(y_pred_list[k]))
        #        print(k,leibie,y_pred_label[k][0])
        if leibie==y_pred_label[k][0]:
            true_count=true_count+1

    print('accuuracy for %s%d:'%(datatype,ttt),true_count,len(y_pred_list),(0.0+true_count)/len(y_pred_list),sep='\t')

def util_runformimic(model):
    FOLD = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']
    for ttt in range(5):
        # print('Loading data for ',ttt)
        # (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

        X_train = open(hp.FILE_PATH+FOLD[ttt]+'_train.pkl', 'rb')
        x_train_list = pickle.load(X_train)
        X_test = open(hp.FILE_PATH+ FOLD[ttt] + '_dev.pkl', 'rb')
        x_test_list = pickle.load(X_test)

        y_train = open(hp.FILE_PATH+FOLD[ttt]+'_train_label.pkl', 'rb')
        y_train_list = pickle.load(y_train)
        y_test = open(hp.FILE_PATH + FOLD[ttt] + '_dev_label.pkl', 'rb')
        y_test_list = pickle.load(y_test)

        x_train = np.array(x_train_list)
        x_test = np.array(x_test_list)

        y_train = np_utils.to_categorical(y_train_list,hp.output_unit)
        y_test = np_utils.to_categorical(y_test_list,hp.output_unit)

        # print(len(x_train), 'train sequences')
        # print(len(x_test), 'test sequences')

        # print('Pad sequences (samples x time)')
        x_train = sequence.pad_sequences(x_train, maxlen=hp.maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=hp.maxlen)
        # print('x_train shape:', x_train.shape)
        # print('x_test shape:', x_test.shape)

        model.fit(x_train, y_train,epochs=hp.epochs,batch_size=hp.BATCHSIZE,validation_data=(x_test, y_test),verbose=0)
        getAcc(model,ttt,'train')
        getAcc(model,ttt,'dev')
        getAcc(model,ttt,'test')





