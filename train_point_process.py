# -*- coding: utf-8 -*-
# /usr/bin/python2

__author__ = "Zhong Qian"
__version__ = "0.2-end2end vision"

from hyperparams import Hyperparams as hp
from data_load import *
from modules import *
import random
import tensorflow as tf
import time
import json
import os
from keras.utils import np_utils
import pickle
class Transformer_Graph():
    def __init__(self,is_training,train_size):
        self.graph = tf.Graph()
        self.is_training = is_training
        self.train_size = train_size
        self.acc_count = 0
        self.acc_true = 0
        self.acc_count_test = 0
        self.acc_true_test = 0
        self.loss_sum = 0
        self.loss_sum_test = 0
        self.mse = 0
        self.mse_test = 0

    def get_model_inputs(self):
        with tf.name_scope('inputs'):
            # input_data_logdesignid_enc = tf.placeholder(tf.int32, [hp.batch_size, hp.maxlen+1], name='input_logdesignid_enc')
            input_data_logdesignid_enc = tf.placeholder(tf.int32, [None, hp.maxlen],name='input_logdesignid_enc')
            time = tf.placeholder(tf.float32, [None, hp.maxlen],name='times')
            time_gap = tf.placeholder(tf.float32, [None, hp.maxlen],name='time_gap')
            target = tf.placeholder(tf.float32, [None], name='targets')
            time_label = tf.placeholder(tf.float32, [None], name='time_label')
            is_training = tf.placeholder(tf.bool, name='is_training')

        return input_data_logdesignid_enc, target, is_training, time , time_label, time_gap

    def generator_batches(self,datatype):
        if datatype == 'train':
            X_file = open(hp.X_file_train, 'rb')
            x_list = pickle.load(X_file)
            Time_file = open(hp.Time_file_train, 'rb')
            Time_list = pickle.load(Time_file)
            Time_gap_file = open(hp.Time_gap_file_train, 'rb')
            Time_gap_list = pickle.load(Time_gap_file)
            Time_file_label = open(hp.Time_file_train_label, 'rb')
            Time_list_label = pickle.load(Time_file_label)
            y_file = open(hp.y_file_train, 'rb')
            y_list = pickle.load(y_file)
        else:
            X_file = open(hp.X_file_test, 'rb')
            x_list = pickle.load(X_file)
            Time_file = open(hp.Time_file_test, 'rb')
            Time_list = pickle.load(Time_file)
            Time_gap_file = open(hp.Time_gap_file_test, 'rb')
            Time_gap_list = pickle.load(Time_gap_file)
            Time_file_label = open(hp.Time_file_test_label, 'rb')
            Time_list_label = pickle.load(Time_file_label)
            y_file = open(hp.y_file_test, 'rb')
            y_list = pickle.load(y_file)

        X_ndarray = np.array(x_list)
        Time_ndarray = np.array(Time_list)
        Time_ndarray_gap = np.array(Time_gap_list)
        Time_ndarray_label = np.array(Time_list_label)
        y_ndarray = np.array(y_list)   #根据Loss定
        count = 0
        shuffleIndex0 = np.random.permutation(len(y_ndarray))

        while (1):
            if (count+1)*hp.batch_size < X_ndarray.shape[0]:
                pad_enc_logdesignid_batch = X_ndarray[shuffleIndex0][count*hp.batch_size:(count+1)*hp.batch_size,:]
                batch_time = Time_ndarray[shuffleIndex0][count*hp.batch_size:(count+1)*hp.batch_size,:]
                batch_time_gap = Time_ndarray_gap[shuffleIndex0][count*hp.batch_size:(count+1)*hp.batch_size,:]
                batch_target = y_ndarray[shuffleIndex0][count*hp.batch_size:(count+1)*hp.batch_size]
                batch_time_label = Time_ndarray_label[shuffleIndex0][count*hp.batch_size:(count+1)*hp.batch_size]
                count = count + 1
            else:
                count=0
                pad_enc_logdesignid_batch = X_ndarray[shuffleIndex0][count * hp.batch_size:(count + 1) * hp.batch_size, :]
                batch_time = Time_ndarray[shuffleIndex0][count * hp.batch_size:(count + 1) * hp.batch_size, :]
                batch_time_gap = Time_ndarray_gap[shuffleIndex0][count * hp.batch_size:(count + 1) * hp.batch_size, :]
                batch_target = y_ndarray[shuffleIndex0][count * hp.batch_size:(count + 1) * hp.batch_size]
                batch_time_label = Time_ndarray_label[shuffleIndex0][count * hp.batch_size:(count + 1) * hp.batch_size]
                shuffleIndex0 = np.random.permutation(len(y_ndarray))

            shuffleIndex = np.random.permutation(len(batch_target))

            yield pad_enc_logdesignid_batch[shuffleIndex], batch_target[shuffleIndex], batch_time[shuffleIndex], batch_time_label[shuffleIndex], batch_time_gap[shuffleIndex]

    def transformer(self,enc_embed_input,action_length,target,time,time_gap,issin = True,is_training=True):
        # Encoder
        # Embedding
        print('[transformer_model] enc_input', enc_embed_input.get_shape())
        with tf.variable_scope("encoder"):

            enc = embedding(enc_embed_input,
                            vocab_size=action_length,
                            num_units=hp.hidden_units,
                            scale=False,
                            scope='enc_embed')
            if issin:
                # Position Encoding
                enc += positional_encoding(enc_embed_input,
                                           num_units=hp.hidden_units,
                                           scale=False,
                                           scope="enc_pe")
            else:
                enc += embedding(
                    tf.tile(tf.expand_dims(tf.range(tf.shape(enc_embed_input)[1]), 0),
                            [tf.shape(enc_embed_input)[0], 1]),
                    vocab_size=hp.maxlen,
                    num_units=hp.hidden_units,
                    scale=False,
                    scope="enc_pe")

            ## Dropout
            if is_training:
                enc = tf.layers.dropout(enc,
                                        rate=hp.dropout_rate,
                                        training=tf.convert_to_tensor(self.is_training))
            ## Blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    enc1,align_score = multihead_attention(queries=enc,
                                              keys=enc,
                                              num_units=hp.hidden_units,
                                              num_heads=hp.num_heads,
                                              dropout_rate=hp.dropout_rate,
                                              is_training=self.is_training,
                                              T_input=time)
                    enc2 = enc1
                    # enc2 = feedforward(enc1, num_units=[4 * hp.hidden_units, hp.hidden_units])

        with tf.variable_scope("point_process"):

            # enc3 = tf.layers.dense(enc2,hp.hidden_units/4,activation=tf.nn.relu)
            # print('------------------enc_shape',enc.get_shape())
            #enc3=enc2
            #flatten = tf.contrib.layers.flatten(enc3)
            #print('------------------enc_shape', flatten.get_shape())
            #enc = tf.reshape(enc,[hp.batch_size,(hp.maxlen+1)*(hp.hidden_units/4)])
            #logits = tf.layers.dense(flatten, hp.output_unit,activation=tf.nn.softmax)


            #shape
            #enc2    [batch_size,hp.maxlen,hp.hidden_units]
            # b = tf.get_variable('pp_b',[hp.output_unit,1],tf.float32,initializer=tf.glorot_uniform_initializer)
            # Wt = tf.get_variable('pp_Wt',[hp.output_unit,1],tf.float32,initializer=tf.glorot_uniform_initializer)
            # Wd = tf.get_variable('pp_Wd',[hp.output_unit,hp.maxlen],tf.float32,initializer=tf.glorot_uniform_initializer)
            # Wh = tf.get_variable('pp_Wh',[hp.output_unit,hp.maxlen*hp.hidden_units,1],tf.float32,initializer=tf.glorot_uniform_initializer)
            b = tf.Variable(tf.random_uniform([hp.output_unit, 1], -1.0, 1.0))
            Wt = tf.Variable(tf.random_uniform([hp.output_unit, 1], 0.0, 1.0))
            Wd = tf.Variable(tf.random_uniform([hp.output_unit, hp.maxlen], -1.0, 1.0))
            Wh = tf.Variable(tf.random_uniform([hp.output_unit, hp.maxlen*hp.hidden_units, 1], -1.0, 1.0))

            context = tf.reshape(enc2,[-1,hp.maxlen*hp.hidden_units])  #enc2 shape:(-1,maxlen,hidden_units)
            context = tf.expand_dims(context,1)
            context = tf.tile(context,[1,hp.output_unit,1])#(-1,38,maxlen*hidden_units)
            #除了wt*t之外的项相加   lambda_all_0 shape[batch_size,output_unit]
            lambda_all_0 = tf.squeeze(tf.squeeze(tf.matmul(tf.expand_dims(context,-1),tf.tile(tf.expand_dims(Wh,0),[tf.shape(context)[0],1,1,1]),transpose_a=True),-1) +b,-1) \
                            +tf.squeeze(tf.matmul(tf.tile(tf.expand_dims(Wd,0),[tf.shape(context)[0],1,1]),tf.expand_dims(time_gap,-1)),-1)



        # print('enc,logits shape:',enc.get_shape(),logits.get_shape())

        return enc1,enc2,enc2,enc2,align_score,lambda_all_0,Wt

    def model_train(self):
        #self.action_length = len(self.id2action)

        with self.graph.as_default():
            input_data_logdesignid_enc, batch_target, is_training, batch_time, batch_time_label, batch_time_gap = self.get_model_inputs()

            with tf.name_scope("optimization"):
                # Create the training and inference logits
                enc1,enc2,enc3,logits,align_score,lambda_all_0,Wt = self.transformer(input_data_logdesignid_enc,hp.output_unit,batch_target,batch_time, batch_time_gap)
                tf.add_to_collection('latent_enc1', enc1)
                tf.add_to_collection('latent_enc2', enc2)
                tf.add_to_collection('latent_enc3', enc3)
                tf.add_to_collection('latent_logits', logits)
                #shape
                #lambda_all_0  [batch_size,output_unit]
                #loss_time_part_index   [batch_size*output_unit]
                #loss_event_part_wt [batch_size,output_unit]
                #loss_event [batch_size,]
                #cost         [1]


                loss_event_part_lambda_all = tf.exp((lambda_all_0+tf.matmul(tf.expand_dims(batch_time_label,-1),Wt*hp.time_scale,transpose_b=True))/hp.exp_constant)
                loss_event_part_index = tf.range(0,tf.shape(batch_target)[0])*hp.output_unit+tf.cast(batch_target,tf.int32)
                #取出未来事件yj的条件概率
                loss_event = tf.log(tf.reshape(tf.gather(tf.reshape(loss_event_part_lambda_all,[-1]),loss_event_part_index),[tf.shape(batch_target)[0]]))

                loss_time_part_wt = tf.matmul(tf.expand_dims(batch_time_label,-1),Wt*hp.time_scale,transpose_b=True)
                #计算lambda
                loss_time_part_lambda_all = tf.exp((lambda_all_0+loss_time_part_wt)/hp.exp_constant)
                loss_time_part_1 = loss_time_part_lambda_all
                loss_time_part_2 = tf.exp(lambda_all_0/hp.exp_constant)
                loss_time = tf.reduce_sum(tf.multiply(tf.expand_dims(loss_time_part_lambda_all-tf.exp(lambda_all_0/hp.exp_constant),-1),hp.exp_constant/(Wt*hp.time_scale)),axis=[1,2])

                cost = -tf.reduce_mean(loss_event-loss_time)
                tf.summary.scalar('loss', cost)

                # Optimizer
                global_steps = tf.Variable(0, name='global_step', trainable=False)
                train_op = tf.train.AdamOptimizer(hp.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(cost,global_step=global_steps)

                # 2.Start train
                checkpoint = hp.transformer_model_file + "best_model.ckpt"

                with tf.Session(graph=self.graph) as sess:
                    config = tf.ConfigProto(inter_op_parallelism_threads=hp.inter_op_parallelism_threads,intra_op_parallelism_threads=hp.intra_op_parallelism_threads)
                    sess = tf.Session(config=config)
                    merged = tf.summary.merge_all()
                    train_writer = tf.summary.FileWriter(hp.log_file + 'train', sess.graph)
                    test_writer = tf.summary.FileWriter(hp.log_file + 'test')
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())

                    max_batchsize = self.train_size // hp.batch_size
                    epoch_i = 1
                    test_generator = self.generator_batches(datatype='test')
                    for batch_i, (pad_enc_logdesignid_batch, train_targets_batch, train_times_batch, train_times_label_batch,train_times_gap_batch) in enumerate(self.generator_batches(datatype='train')):
                        if (batch_i % max_batchsize) + 1 == max_batchsize:
                            epoch_i += 1
                            self.acc_count = 0
                            self.acc_true = 0
                            self.loss_sum = 0
                            self.acc_count_test = 0
                            self.acc_true_test = 0
                            self.loss_sum_test = 0
                            self.mse_test = self.mse = 0
                            if epoch_i >= hp.epochs:
                                break

                        # Training step
                        with tf.name_scope('loss'):
                            summary, _, enc1_,enc2_,enc3_,logits_, loss,loss_time_,loss_event_,Wt_,lambda_all_0_, lambda_all_ = sess.run(
                                [merged, train_op, enc1,enc2,enc3,logits, cost,loss_time,loss_event,Wt,lambda_all_0,loss_event_part_lambda_all],
                                {input_data_logdesignid_enc: pad_enc_logdesignid_batch,
                                 batch_target: train_targets_batch,
                                 is_training:True,
                                 batch_time:train_times_batch,
                                 batch_time_label:train_times_label_batch,
                                 batch_time_gap:train_times_gap_batch
                                 })
                            self.acc_count  += len(train_targets_batch)
                            yy = np.argmax(lambda_all_,axis=1)
                            xx = train_targets_batch
                            self.acc_true += sum([xx[i]==yy[i] for i in range(0,len(xx))])
                            self.loss_sum += loss
                            num=300
                            time_maximum=0.2
                            batch_num = len(train_targets_batch)
                            t_ = np.linspace(0, time_maximum, num=num)
                            t_ = np.tile(np.reshape(t_,[1,-1,1]),[batch_num,1,1])#(16,1000,1)

                            loss_event_part_wt_ = np.matmul(t_,np.transpose(Wt_*hp.time_scale))#(16,1000,38)
                            #print('check',t_[0][-1],(Wt_*hp.time_scale)[0][0],loss_event_part_wt_[0][-1][0])
                            lambda_all_0_ = np.tile(np.expand_dims(lambda_all_0_,1),[1,num,1])
                            lambda_all_ = np.exp((lambda_all_0_+loss_event_part_wt_)/hp.exp_constant)#(16,1000,38)

                            Wt_ = np.tile(np.expand_dims(Wt_,0),[num,1,1])#(1000,38,1)

                            temp = np.exp(np.sum(np.multiply(np.expand_dims(lambda_all_-np.exp(lambda_all_0_/hp.exp_constant),-1),-hp.exp_constant/(Wt_*hp.time_scale)),axis=(2,3)))#(16,1000)
                            time_pred_ = time_maximum * np.average(np.multiply(np.multiply(np.expand_dims(temp,-1),np.expand_dims(np.sum(lambda_all_,2),-1)),t_),(1,2))
                            self.mse += sum([x if x<100 else 100 for x in (time_pred_-train_times_label_batch)**2])

                            for i in range(len(time_pred_)):
                                if ((time_pred_-train_times_label_batch)**2)[i]>100:
                                    print(time_pred_[i],train_times_label_batch[i])

                            if(self.mse>300000):
                                print(time_pred_,train_times_label_batch)

                            train_writer.add_summary(summary, batch_i)

                        # Debug message updating us on the status of the training
                        if batch_i % hp.display_step == 0:
                            (pad_enc_valid_logdesignid_batch, valid_targets_batchs, valid_times_batchs, valid_times_label_batchs, valid_times_gap_batchs) = next(test_generator)

                            # Calculate validation cost
                            summary, _, enc1_,enc2_,enc3_,logits_, loss,loss_time_,loss_event_,Wt_,lambda_all_0_, lambda_all_ = sess.run(
                                [merged, train_op, enc1,enc2,enc3,logits, cost,loss_time,loss_event,Wt,lambda_all_0,loss_event_part_lambda_all],
                                {input_data_logdesignid_enc: pad_enc_valid_logdesignid_batch,
                                 batch_target: valid_targets_batchs,
                                 is_training:False,
                                 batch_time:valid_times_batchs,
                                 batch_time_label:valid_times_label_batchs,
                                 batch_time_gap:valid_times_gap_batchs
                                 })

                            self.acc_count_test  += len(valid_targets_batchs)
                            yy = np.argmax(lambda_all_,axis=1)
                            xx = valid_targets_batchs
                            # yy = [np.argmax(i) for i in preds_]
                            self.acc_true_test += sum([xx[i]==yy[i] for i in range(0,len(xx))])
                            self.loss_sum_test += loss

                            batch_num = len(valid_targets_batchs)
                            t_ = np.linspace(0, time_maximum, num=num)
                            t_ = np.tile(np.reshape(t_,[1,-1,1]),[batch_num,1,1])#(16,1000,1)

                            loss_event_part_wt_ = np.matmul(t_,np.transpose(Wt_*hp.time_scale))#(16,1000,38)
                            #print('check',t_[0][-1],(Wt_*hp.time_scale)[0][0],loss_event_part_wt_[0][-1][0])
                            lambda_all_0_ = np.tile(np.expand_dims(lambda_all_0_,1),[1,num,1])
                            lambda_all_ = np.exp((lambda_all_0_+loss_event_part_wt_)/hp.exp_constant)#(16,1000,38)

                            Wt_ = np.tile(np.expand_dims(Wt_,0),[num,1,1])#(1000,38,1)

                            temp = np.exp(np.sum(np.multiply(np.expand_dims(lambda_all_-np.exp(lambda_all_0_/hp.exp_constant),-1),-hp.exp_constant/(Wt_*hp.time_scale)),axis=(2,3)))#(16,1000)
                            time_pred_ = time_maximum * np.average(np.multiply(np.multiply(np.expand_dims(temp,-1),np.expand_dims(np.sum(lambda_all_,2),-1)),t_),(1,2))
                            self.mse_test += sum([x if x<100 else 100 for x in (time_pred_-valid_times_label_batchs)**2])

                            test_writer.add_summary(summary, batch_i)
                            if(batch_i%300==0 or (batch_i % max_batchsize) > max_batchsize-3):
                                print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f} - Train acc: {:>6.3f}  - Train rmse: {:>6.3f} - TestLoss: {:>6.3f} - Test acc: {:>6.3f}  - Test rmse: {:>6.3f}'
                                      .format(epoch_i,
                                              hp.epochs,
                                              (batch_i % max_batchsize) + 1,
                                              max_batchsize,
                                              (0.0+self.loss_sum)/self.acc_count,
                                              (0.0+self.acc_true)/self.acc_count,
                                              np.sqrt((0.0+self.mse)/self.acc_count),
                                              (0.0+self.loss_sum_test)/self.acc_count_test,
                                              (0.0+self.acc_true_test)/self.acc_count_test,

                                              np.sqrt((0.0+self.mse_test)/self.acc_count)
                                              ))

                                print('pred',time_pred_)
                                print('true',train_times_label_batch)


                        if ((batch_i % max_batchsize) + 1) % hp.saver_step == 0:
                            saver = tf.train.Saver()
                            saver.save(sess, os.path.join(os.getcwd(),
                                                          hp.transformer_model_file + "epoch" + str(epoch_i) + "batch" + str(
                                                              (batch_i % max_batchsize) + 1) + ".ckpt"))

                    # Save Model
                    saver = tf.train.Saver()
                    saver.save(sess, checkpoint)

                    print('Model Trained and Saved')

    def run(self):
        #self.id2action, self.action2id = create_vocab()
        #print('--------------CREATE VOCAB FINISH!--------------')
        self.model_train()

def countsize(mainpath):
    count = 0
    for fn in os.listdir(mainpath):
        count = count+1
    return count

def dir_check(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == '__main__':
    start = time.time()

    # 参数1 数据集名称
    # train_dataset_file = hp.train_dataset_file  # 玩家序列文件训练集
    # test_dataset_file = hp.test_dataset_file   # 玩家序列文件验证集

    logid_freq_file = hp.logid_freq_file
    logdesignid_freq_file = hp.logdesignid_freq_file

    model_file = hp.transformer_model_file
    dir_check(model_file)
    log_file = hp.log_file
    dir_check(log_file)

    train_size = hp.train_size
    test_size = hp.test_size

    print('train_size', train_size)
    print('test_size', test_size)

    transformer_trainn = Transformer_Graph(is_training = True,train_size = train_size)
    transformer_trainn.run()

    stop = time.time()
    print("模型训练时间: " + str(stop - start) + "秒")
    print("memory: ", hpy().heap())