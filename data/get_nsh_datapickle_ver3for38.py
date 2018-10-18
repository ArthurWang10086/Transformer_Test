import json
import pickle
import sys
from hyperparams import Hyperparams as hp
import os

if __name__ == '__main__':
    YUZHI=int(hp.maxlen)+1
    file = open('dataset/savefile_event_train1.txt','r')
    final_data=[]
    final_label=[]
    count=0
    for line in file.readlines():
        count=count+1
        if count%10000==0:
            print(count)
        list_str=line.split(' ')
        list_str = list_str[:len(list_str)-1]
        data = [int(i) for i in list_str]
        if len(data)< YUZHI:
            continue
        for i in range(len(data)-YUZHI+1):
            final_data.append(data[i:i+YUZHI-1])
            final_label.append(data[i+YUZHI-1])
        if len(final_label)>70000:
            break

    file = open('dataset/savefile_time_train1.txt','r')
    final_time=[]
    count=0
    for line in file.readlines():
        count=count+1
        if count%10000==0:
            print(count)
        list_str=line.split(' ')
        list_str = list_str[:len(list_str)-1]
        data = [int(i) for i in list_str]
        if len(data)< YUZHI:
            continue
        for i in range(len(data)-YUZHI+1):
            final_time.append(data[i:i+YUZHI-1])
        if len(final_time)>70000:
            break


    print(len(final_label))
    pickle.dump(final_data, open('all_datatrain_seq'+str(YUZHI)+'.pkl', "wb"))
    pickle.dump(final_time, open('all_timetrain_seq'+str(YUZHI)+'.pkl', "wb"))
    pickle.dump(final_label, open('all_labeltrain_seq'+str(YUZHI)+'.pkl', "wb"))


    file = open('dataset/savefile_event_test1.txt','r')
    final_data=[]
    final_label=[]
    count=0
    for line in file.readlines():
        count=count+1
        if count%10000==0:
            print(count)
        list_str=line.split(' ')
        list_str = list_str[:len(list_str)-1]
        data = [int(i) for i in list_str]
        if len(data)< YUZHI:
            continue
        for i in range(len(data)-YUZHI+1):
            final_data.append(data[i:i+YUZHI-1])
            final_label.append(data[i+YUZHI-1])
        if len(final_label)>30000:
            break

    file = open('dataset/savefile_time_test1.txt','r')
    final_time=[]
    count=0
    for line in file.readlines():
        count=count+1
        if count%10000==0:
            print(count)
        list_str=line.split(' ')
        list_str = list_str[:len(list_str)-1]
        data = [int(i) for i in list_str]
        if len(data)< YUZHI:
            continue
        for i in range(len(data)-YUZHI+1):
            final_time.append(data[i:i+YUZHI-1])
        if len(final_time)>70000:
            break

    print(len(final_label))
    pickle.dump(final_data, open('all_datatest_seq'+str(YUZHI)+'.pkl', "wb"))
    pickle.dump(final_time, open('all_timetest_seq'+str(YUZHI)+'.pkl', "wb"))
    pickle.dump(final_label, open('all_labeltest_seq'+str(YUZHI)+'.pkl', "wb"))