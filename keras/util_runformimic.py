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
    y_pred_dataset_file = open(hp.FILE_PATH + 'all_data%s_seq'%(datatype) + str(hp.maxlen + 1) + '.pkl.' + FOLD[ttt], 'rb')
    y_pred_dataset = pickle.load(y_pred_dataset_file)
    y_pred_label_dataset = open(hp.FILE_PATH + 'all_label%s_seq'%(datatype) + str(hp.maxlen + 1) + '.pkl.' + FOLD[ttt], 'rb')
    y_pred_label = pickle.load(y_pred_label_dataset)
    X_file = open(hp.FILE_PATH + 'all_time%s_seq'%(datatype) + str(hp.maxlen + 1) + '.pkl.' + FOLD[ttt], 'rb')
    x_train_time_list = pickle.load(X_file)


    y_pred_dataset= np.array(y_pred_dataset)
    y_pred_dataset= sequence.pad_sequences(y_pred_dataset, maxlen=hp.maxlen)
    x_train_time = np.array(x_train_time_list)
    #y_pred_label = np_utils.to_categorical(y_pred_label,76)

    y_pred = model.predict([y_pred_dataset,x_train_time])
    y_pred_list = y_pred.tolist()
    # print(y_pred_list[0])
    # print(y_pred_label[k])

    true_count=0
    #    print(y_pred_list)
    #   print(y_pred_label)

    for k in range(len(y_pred_list)):
        leibie=y_pred_list[k].index(max(y_pred_list[k]))
        #        print(k,leibie,y_pred_label[k][0])
        if leibie==y_pred_label[k]:
            true_count=true_count+1

    print('accuuracy for %s%d:'%(datatype,ttt),true_count,len(y_pred_list),(0.0+true_count)/len(y_pred_list),sep='\t')

def util_runformimic(model):
    FOLD = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']
    for ttt in range(5):
        X_file = open(hp.FILE_PATH + 'all_data%s_seq'%('train') + str(hp.maxlen + 1) + '.pkl.' + FOLD[ttt], 'rb')
        x_train_list = pickle.load(X_file)
        y_file = open(hp.FILE_PATH + 'all_label%s_seq'%('train') + str(hp.maxlen + 1) + '.pkl.' + FOLD[ttt], 'rb')
        y_train_list= pickle.load(y_file)

        X_file = open(hp.FILE_PATH + 'all_data%s_seq'%('dev') + str(hp.maxlen + 1) + '.pkl.' + FOLD[ttt], 'rb')
        x_test_list = pickle.load(X_file)
        y_file = open(hp.FILE_PATH + 'all_label%s_seq'%('dev') + str(hp.maxlen + 1) + '.pkl.' + FOLD[ttt], 'rb')
        y_test_list = pickle.load(y_file)

        X_file = open(hp.FILE_PATH + 'all_time%s_seq'%('train') + str(hp.maxlen + 1) + '.pkl.' + FOLD[ttt], 'rb')
        x_train_time_list = pickle.load(X_file)
        x_train_time = np.array(x_train_time_list)
        X_file = open(hp.FILE_PATH + 'all_time%s_seq'%('dev') + str(hp.maxlen + 1) + '.pkl.' + FOLD[ttt], 'rb')
        x_test_time_list = pickle.load(X_file)
        x_test_time = np.array(x_test_time_list)

        x_train = np.array(x_train_list)
        x_test = np.array(x_test_list)
        # x_train = np.concatenate((x_train, x_train_time), axis=0)
        # x_test = np.concatenate((x_test, x_test_time), axis=0)
        y_train = np_utils.to_categorical(y_train_list,hp.output_unit)
        y_test = np_utils.to_categorical(y_test_list,hp.output_unit)

        model.fit([x_train, x_train_time], y_train,epochs=20,batch_size=hp.batch_size,validation_data=([x_test,x_test_time], y_test),verbose=0)
        getAcc(model,ttt,'train')
        getAcc(model,ttt,'dev')
        getAcc(model,ttt,'test')





