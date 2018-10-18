# -*- coding: utf-8 -*-
# /usr/bin/python2

__author__ = "Luo Yifan"
__version__ = "0.2-end2end vision"

from hyperparams import Hyperparams as hp
from data_load import *
from modules import *
import random
import tensorflow as tf
import time
import json
import os
import numpy
class Transformer_Graph():
    def __init__(self,is_training,train_size):
        self.graph = tf.Graph()
        self.is_training = is_training
        self.train_size = train_size

    def get_model_inputs(self):
        with tf.name_scope('inputs'):
            input_data_logdesignid_enc = tf.placeholder(tf.int32, [hp.batch_size, hp.maxlen+1], name='input_logdesignid_enc')
            target = tf.placeholder(tf.float32, [hp.batch_size,1], name='targets')
            is_training = tf.placeholder(tf.bool, name='is_training')

        return input_data_logdesignid_enc, target, is_training

    def generator_batches(self,datatype):
        if datatype == 'train':
            dataset_file_normal,dataset_file_plugin = hp.normal_dataset_file_train,hp.plug_dataset_file_train
        else: 
            dataset_file_normal,dataset_file_plugin = hp.normal_dataset_file_test,hp.plug_dataset_file_test
        count = 0
        batch_player_logdesignid_data_enc = list()
        batch_target_list = list()
        while (1):
            for filename in os.listdir(dataset_file_normal):
                logdesignid_data_enc = list()
                player_event_dir = dataset_file_normal + '/' + filename
                with open(player_event_dir, "r") as load_f:
                    player_event = json.load(load_f)
                    if player_event == []:
                        continue
                    for each_player_event in player_event:
                        letter_log = each_player_event.split('#')[3]
                        letter = self.action2id.get(letter_log, self.action2id['<UNK>'])
                        logdesignid_data_enc.append(letter)
                    batch_target_list.append(0.0)
                        
                batch_player_logdesignid_data_enc.append(logdesignid_data_enc)
                count += 1
                if count == hp.batch_size/2:
                    break
            
            for filename in os.listdir(dataset_file_plugin):
            	logdesignid_data_enc = list()
                player_event_dir = dataset_file_plugin + '/' + filename
                with open(player_event_dir, "r") as load_f:
                    player_event = json.load(load_f)
                    if player_event == []:
                        continue
                    for each_player_event in player_event:
                        letter_log = each_player_event.split('#')[3]
                        letter = self.action2id.get(letter_log, self.action2id['<UNK>'])
                        logdesignid_data_enc.append(letter)
                    batch_target_list.append(1.0)
                        
                batch_player_logdesignid_data_enc.append(logdesignid_data_enc)
                count += 1
                if count == hp.batch_size:
                    pad_enc_logdesignid_batch = np.array(
                        pad_sentence_batch(batch_player_logdesignid_data_enc,
                                           self.action2id['<PAD>'],
                                           self.action2id['<EOS>'],
                                           self.action2id['<GO>'], codertype='encoder'))
                    batch_target = np.array(batch_target_list)
                    batch_target = np.reshape(batch_target,(-1,1))
                    count = 0  # init again
                    batch_player_logdesignid_data_enc = list()
                    batch_target_list = list()
                    shuffle_list = list(zip(pad_enc_logdesignid_batch, batch_target))
                    random.shuffle(shuffle_list)
                    pad_enc_logdesignid_batch[:], batch_target[:] = zip(*shuffle_list)
                    break
            print(' pad_enc_logdesignid_batch, batch_target type:', type(pad_enc_logdesignid_batch), type(batch_target))
            print(' pad_enc_logdesignid_batch, batch_target shape:', pad_enc_logdesignid_batch.shape,batch_target.shape)
            print(' pad_enc_logdesignid_batch',pad_enc_logdesignid_batch)
            print('batch_target type:', batch_target)
            yield pad_enc_logdesignid_batch, batch_target

    def transformer(self,enc_embed_input,action_length,target,issin = False,is_training=True):
        # Encoder
        # Embedding
        print('[transformer_model] enc_input', enc_embed_input.get_shape())
        with tf.variable_scope("encoder"):

            enc = embedding(enc_embed_input,
                            vocab_size=action_length,
                            num_units=hp.hidden_units,
                            scale=True,
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
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              num_units=hp.hidden_units,
                                              num_heads=hp.num_heads,
                                              dropout_rate=hp.dropout_rate,
                                              is_training=self.is_training,
                                              causality=False)
                    enc = feedforward(enc, num_units=[4 * hp.hidden_units, hp.hidden_units])

        
        with tf.variable_scope("softmax"):
            enc = tf.layers.dense(enc,hp.hidden_units/4)
            enc = tf.reshape(enc,[hp.batch_size,(hp.maxlen+1)*(hp.hidden_units/4)])
            logits = tf.layers.dense(enc, 1,activation=tf.nn.softmax)
           # print('enc,logits shape:',enc.get_shape(),logits.get_shape())

        return logits

    def model_train(self):
        self.action_length = len(self.id2action)

        with self.graph.as_default():
            input_data_logdesignid_enc, batch_target, is_training = self.get_model_inputs()

            with tf.name_scope("optimization"):
                # Create the training and inference logits
                logits = self.transformer(input_data_logdesignid_enc,self.action_length,batch_target)
                # Predicr label
                preds = tf.nn.sigmoid(logits)
                print('[transformer_model] logits', logits.get_shape())
                print('[transformer_model] logits', logits)
                print('=======================================')

                print('[transformer_model] preds', preds.get_shape())
                print('[transformer_model] preds', preds)
                print('=======================================')
                #preds=logits

                # Loss
               # logloss = tf.nn.sparse_softmax_cross_entropy_with_logits()
                logloss = tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_target, logits=logits)
                cost = tf.reduce_sum(logloss) / (hp.batch_size)

                output = tf.round(preds)
                acc = tf.reduce_sum(tf.to_float(tf.equal(output, batch_target)))/ (hp.batch_size)
                TP = tf.count_nonzero(output * batch_target)
                TN = tf.count_nonzero((output - 1) * (batch_target - 1))
                FP = tf.count_nonzero(output * (batch_target - 1))
                FN = tf.count_nonzero((output - 1) * batch_target)
                precision = tf.divide(TP, tf.add(TP, FP))
                recall = tf.divide(TP, tf.add(TP, FN))
                f1 = tf.divide(tf.multiply(tf.constant(2.0,dtype=tf.float64), tf.multiply(precision, recall)), tf.add(precision, recall))
                
                # Summary 
                tf.summary.scalar('loss', cost)
                tf.summary.scalar('acc', acc)
                tf.summary.scalar('precision', precision)
                tf.summary.scalar('recall', recall)
                tf.summary.scalar('f1', f1)

                # Loss function
                print('[model_train] targets', batch_target.get_shape())
                #batch_target_smooth = label_smoothing(batch_target)
                #print('[model_train] targets smoth',batch_target_smooth.get_shape())
        
                # Optimizer
                global_steps = tf.Variable(0, name='global_step', trainable=False)
                train_op = tf.train.AdamOptimizer(hp.learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-8).minimize(cost,global_step=global_steps)

                # 2.Start train
                checkpoint = hp.transformer_model_file + "best_model.ckpt"

                with tf.Session(graph=self.graph) as sess:
                    merged = tf.summary.merge_all()
                    train_writer = tf.summary.FileWriter(hp.log_file + 'train', sess.graph)
                    test_writer = tf.summary.FileWriter(hp.log_file + 'test')
                    sess.run(tf.global_variables_initializer())

                    max_batchsize = self.train_size // hp.batch_size
                    epoch_i = 1
                    test_generator = self.generator_batches(datatype='test')
                    for batch_i, (pad_enc_logdesignid_batch, train_targets_batch) in enumerate(self.generator_batches(datatype='train')):
                        #print('train_targets_batch',pad_enc_logdesignid_batch)
                        #print('train_targets_batch',train_targets_batch)
                        if (batch_i % max_batchsize) + 1 == max_batchsize:
                            epoch_i += 1
                            if epoch_i >= hp.epochs:
                                break

                        # Training step
                        with tf.name_scope('loss'):
                            summary, _, logits_, loss, logloss_, preds_ = sess.run(
                                [merged, train_op, logits, cost, logloss, preds],
                                {input_data_logdesignid_enc: pad_enc_logdesignid_batch,
                                 batch_target: train_targets_batch,
                                 is_training:True
                                 })
                            #print('logits',logits_)
                            #print('logloss',logloss_)
                            #print('cost_',loss)
                            #print('preds_',preds_)

                            train_writer.add_summary(summary, batch_i)

                        # Debug message updating us on the status of the training
                        if batch_i % hp.display_step == 0:
                            (pad_enc_valid_logdesignid_batch, valid_targets_batchs) = next(test_generator)
                            
                            # Calculate validation cost
                            summary, validation_loss, validation_precision, validation_recall, validation_f1 = sess.run(
                                [merged, cost, precision, recall, f1],
                                {input_data_logdesignid_enc: pad_enc_valid_logdesignid_batch,
                                 batch_target: valid_targets_batchs,
                                 is_training:False
                                 })

                            test_writer.add_summary(summary, batch_i)

                            print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}  - Validation loss: {:>6.3f} - Precision: {:>6.3f} - Recall: {:>6.3f} - f1: {:>6.3f}'
                                  .format(epoch_i,
                                          hp.epochs,
                                          (batch_i % max_batchsize) + 1,
                                          max_batchsize,
                                          loss,
                                          validation_loss,
                                          validation_precision,
                                          validation_recall,
                                          validation_f1))
                    
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
        self.id2action, self.action2id = create_vocab()
        print('--------------CREATE VOCAB FINISH!--------------')
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
    train_dataset_file = hp.train_dataset_file  # 玩家序列文件训练集
    test_dataset_file = hp.test_dataset_file   # 玩家序列文件验证集

    logid_freq_file = hp.logid_freq_file
    logdesignid_freq_file = hp.logdesignid_freq_file

    model_file = hp.transformer_model_file
    dir_check(model_file)
    log_file = hp.log_file
    dir_check(log_file)

    train_size = countsize(train_dataset_file)
    test_size = countsize(test_dataset_file)
    print('train_size', train_size)
    print('test_size', test_size)

    transformer_trian = Transformer_Graph(is_training = True,train_size = train_size)
    transformer_trian.run()

    stop = time.time()
    print("模型训练时间: " + str(stop - start) + "秒")
    print("memory: ", hpy().heap())
