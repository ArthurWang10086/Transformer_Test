# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import pickle
if __name__ == '__main__':
    BPTT=3
    FOLD=['fold1','fold2','fold3','fold4','fold5']
    print('Loading data...')
    file_path='dataset/'
    for fold_num in range(len(FOLD)):
        for name in ['train','test','dev']:
            f = open(file_path+FOLD[fold_num]+'/%s.pkl'%(name),'rb')
            data=pickle.load(f)
            data=data[name]
            with open(file_path+FOLD[fold_num]+'/event_mimic_%s.txt'%(name), 'wb') as file:
                file.write('\n'.join([' '.join([str(y['type_event']) for y in x ]) for x in data ]))
            with open(file_path+FOLD[fold_num]+'/time_mimic_%s.txt'%(name), 'wb') as file:
                file.write('\n'.join([' '.join([str(y['time_since_start']) for y in x ]) for x in data ]))

            maxlen = 3
            YUZHI=maxlen+1
            file = open(file_path+FOLD[fold_num]+'/event_mimic_%s.txt'%(name),'r')
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

            file = open(file_path+FOLD[fold_num]+'/time_mimic_%s.txt'%(name),'r')
            final_time=[]
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
                    final_time_label.append((data[i+YUZHI-1]-L[0]))
                    final_time.append([(x-L[0]) for x in L][:])
                    #final_time.append(L)


            print(len(final_label),len(final_time))
            pickle.dump(final_data, open('all_data%s_seq'%(name)+str(YUZHI)+'.pkl.'+FOLD[fold_num], "wb"))
            pickle.dump(final_time, open('all_time%s_seq'%(name)+str(YUZHI)+'.pkl.'+FOLD[fold_num], "wb"))
            pickle.dump(final_time_label, open('all_time%s_label_seq'%(name)+str(YUZHI)+'.pkl.'+FOLD[fold_num], "wb"))
            pickle.dump(final_time_raw, open('all_time%s_raw_seq'%(name)+str(YUZHI)+'.pkl.'+FOLD[fold_num], "wb"))
            pickle.dump(final_label, open('all_label%s_seq'%(name)+str(YUZHI)+'.pkl.'+FOLD[fold_num], "wb"))





