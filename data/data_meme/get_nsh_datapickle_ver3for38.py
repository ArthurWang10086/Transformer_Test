import json
import pickle
import sys
import math
import os

if __name__ == '__main__':
    YUZHI=3+1
    time_norm = 10000.0
    file = open('dataset/all_raw_trainevent.txt','r')
    final_data=[]
    final_label=[]
    count=0
    for line in file.readlines():
        count=count+1
        if count%10000==0:
            print(count)
        list_str=line.split(' ')
        list_str = list_str[:len(list_str)-1]
        data = [float(i) for i in list_str]
        if len(data)< YUZHI:
            continue
        for i in range(len(data)-YUZHI+1):
            final_data.append(data[i:i+YUZHI-1])
            final_label.append(data[i+YUZHI-1])
        if len(final_label)>70000:
            break

    file = open('dataset/all_raw_traintime.txt','r')
    final_time=[]
    final_time_gap = []
    final_time_label=[]
    final_time_raw=[]
    count=0
    for line in file.readlines():
        count=count+1
        if count%10000==0:
            print(count)
        list_str=line.split(' ')
        list_str = list_str[:len(list_str)-1]
        data = [float(i) for i in list_str]
        if len(data)< YUZHI:
            continue
        for i in range(len(data)-YUZHI+1):
            L = data[i:i+YUZHI-1]
            final_time_raw.append(L[:])
            final_time_label.append((data[i+YUZHI-1]-data[i+YUZHI-2])/time_norm)
            if i<1:
                final_time_gap.append([0]+[(L[i+1]-L[i])/time_norm for i in range(0,len(L)-1)][:])
            else:
                final_time_gap.append([(data[i]-data[i-1])/time_norm]+[(L[i+1]-L[i])/time_norm for i in range(0,len(L)-1)][:])
            final_time.append([math.exp((x-L[0])/time_norm) for x in L][:])
           # final_time.append([float(x-L[0])/max(L) for x in L][:])
            #final_time.append(L)
        if len(final_time)>70000:
            break


    print(len(final_label),len(final_time))
    pickle.dump(final_data, open('all_datatrain_seq'+str(YUZHI)+'.pkl', "wb"))
    pickle.dump(final_time, open('all_timetrain_seq'+str(YUZHI)+'.pkl', "wb"))
    pickle.dump(final_time_gap, open('all_timetrain_gap_seq'+str(YUZHI)+'.pkl', "wb"))
    pickle.dump(final_time_label, open('all_timetrain_label_seq'+str(YUZHI)+'.pkl', "wb"))
    pickle.dump(final_time_raw, open('all_timetrain_raw_seq'+str(YUZHI)+'.pkl', "wb"))
    pickle.dump(final_label, open('all_labeltrain_seq'+str(YUZHI)+'.pkl', "wb"))


    file = open('dataset/all_raw_testevent.txt','r')
    final_data=[]
    final_label=[]
    count=0
    for line in file.readlines():
        count=count+1
        if count%10000==0:
            print(count)
        list_str=line.split(' ')
        list_str = list_str[:len(list_str)-1]
        data = [float(i) for i in list_str]
        if len(data)< YUZHI:
            continue
        for i in range(len(data)-YUZHI+1):
            final_data.append(data[i:i+YUZHI-1])
            final_label.append(data[i+YUZHI-1])
        if len(final_label)>30000:
            break
    file.close()

    file = open('dataset/all_raw_testtime.txt','r')
    final_time=[]
    final_time_gap=[]
    final_time_label=[]
    final_time_raw=[]
    count=0
    for line in file.readlines():
        count=count+1
        if count%10000==0:
            print(count)
        list_str=line.split(' ')
        list_str = list_str[:len(list_str)-1]
        data = [float(i) for i in list_str]
        if len(data)< YUZHI:
            continue
        for i in range(len(data)-YUZHI+1):
            L = data[i:i+YUZHI-1]
            final_time_raw.append(L[:])
            final_time_label.append((data[i+YUZHI-1]-data[i+YUZHI-2])/time_norm)
            if i<1:
                final_time_gap.append([0]+[(L[i+1]-L[i])/time_norm for i in range(0,len(L)-1)][:])
            else:
                final_time_gap.append([(data[i]-data[i-1])/time_norm]+[(L[i+1]-L[i])/time_norm for i in range(0,len(L)-1)][:])
            final_time.append([math.exp((x-L[0])/time_norm)for x in L][:])
        if len(final_time)>30000:
            break

    print(len(final_data),len(final_time))
    pickle.dump(final_data, open('all_datatest_seq'+str(YUZHI)+'.pkl', "wb"))
    pickle.dump(final_time, open('all_timetest_seq'+str(YUZHI)+'.pkl', "wb"))
    pickle.dump(final_time_gap, open('all_timetest_gap_seq'+str(YUZHI)+'.pkl', "wb"))
    pickle.dump(final_time_label, open('all_timetest_label_seq'+str(YUZHI)+'.pkl', "wb"))
    pickle.dump(final_time_raw, open('all_timetest_raw_seq'+str(YUZHI)+'.pkl', "wb"))
    pickle.dump(final_label, open('all_labeltest_seq'+str(YUZHI)+'.pkl', "wb"))