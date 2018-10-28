# -*- coding: utf-8 -*-
# /usr/bin/python2

__author__ = "Luo Yifan"
__version__ = "0.2"

from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np

def normalize(inputs, epsilon=1e-8, scope="ln", reuse=None):
    #layer normalize
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta
        return outputs


def embedding(inputs, vocab_size, num_units, scale=True, scope="embedding", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size + 1, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())

        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)
    return outputs


def positional_encoding(inputs, num_units, scale=True, scope="positional_encoding", reuse=None):
    inputs = tf.convert_to_tensor(inputs, np.int32)
    N=hp.batch_size  # N:32,T:length
    T = hp.maxlen

    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i - i % 2) / num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc, np.float32)

        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units ** 0.5

        return outputs


def feedforward(inputs, num_units=[4 * hp.hidden_units, hp.hidden_units], scope="multihead_attention", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs #(N, T, C)


def multihead_attention(queries, keys, num_units=None, num_heads=4, dropout_rate=0, is_training=True,
                        causality=False, scope="multihead_attention", reuse=None, T_input=None):
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (h*N, T_q, T_k)

        # Dropouts
        outputs2 = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs2 = tf.matmul(outputs2, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs2 = tf.concat(tf.split(outputs2, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs2 += queries

        # Normalize
        outputs2 = normalize(outputs2)  # (N, T_q, C)

    return outputs2,outputs


def multihead_attention_keras(queries, keys, size_per_head=None, nb_head=4, dropout_rate=0, is_training=True,
                        causality=False, scope="multihead_attention", reuse=None, T_input=None):
    with tf.variable_scope(scope):

        WQ = tf.Variable(tf.random_uniform([hp.hidden_units,size_per_head*nb_head], -1.0, 1.0))
        WK = tf.Variable(tf.random_uniform([hp.hidden_units,size_per_head*nb_head], -1.0, 1.0))
        WV = tf.Variable(tf.random_uniform([hp.hidden_units,size_per_head*nb_head], -1.0, 1.0))


        # WQ = tf.get_variable('WQ',[32,size_per_head*nb_head],initializer=tf.glorot_uniform_initializer)
        # WK = tf.get_variable('WK',[tf.shape(queries)[2],size_per_head*nb_head],tf.float32,initializer=tf.glorot_uniform_initializer)
        # WV = tf.get_variable('WV',[tf.shape(queries)[2],size_per_head*nb_head],tf.float32,initializer=tf.glorot_uniform_initializer)
        import keras
        Q_seq = keras.backend.dot(queries, WQ)
        # print('T_seq shape:',Q_seq.shape,self.WQ.shape)
        Q_seq = tf.reshape(Q_seq, (-1, hp.maxlen, 8, size_per_head))
        Q_seq = tf.transpose(Q_seq, [0,2,1,3])
        K_seq = keras.backend.dot(queries, WK)
        K_seq = tf.reshape(K_seq, (-1, hp.maxlen, 8, size_per_head))
        K_seq = tf.transpose(K_seq, [0,2,1,3])
        V_seq = keras.backend.dot(queries, WV)
        V_seq = tf.reshape(V_seq, (-1, hp.maxlen, nb_head, size_per_head))
        V_seq = tf.transpose(V_seq, [0,2,1,3])

        A = keras.backend.batch_dot(Q_seq, K_seq, axes=[3,3]) / size_per_head**0.5
        # print('T_seq shape:',tf.matmul(Q_seq, K_seq, adjoint_a=None, adjoint_b=True).shape)
        #print('T_seq shape:',Q_seq.shape,K_seq.shape,A.shape)
        Q_len,V_len = None,None
        def Mask(inputs, seq_len, mode='mul'):
            if seq_len == None:
                return inputs
            else:
                mask = tf.one_hot(seq_len[:,0], tf.shape(inputs)[1])
                mask = 1 - tf.cumsum(mask, 1)
                for _ in range(len(inputs.shape)-2):
                    mask = tf.expand_dims(mask, 2)
                if mode == 'mul':
                    return inputs * mask
                if mode == 'add':
                    return inputs - (1 - mask) * 1e12

        A = tf.transpose(A, [0,3,2,1])
        A = Mask(A, V_len, 'add')
        A = tf.transpose(A, [0,3,2,1])
        A = tf.nn.softmax(A)
        #输出并mask
        O_seq = keras.backend.batch_dot(A, V_seq, axes=[3,2])
        O_seq = tf.transpose(O_seq, [0,2,1,3])
        O_seq = tf.reshape(O_seq, (-1, hp.maxlen, size_per_head*nb_head))
        O_seq = Mask(O_seq, Q_len, 'mul')
        return O_seq





def multihead_attention_time(queries, keys, num_units=None, num_heads=4, dropout_rate=0, is_training=True,
                        causality=False, scope="multihead_attention", reuse=None, T_input=None):
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        T = tf.reshape(T_input,(-1,queries.shape[1],1))
        T = T*np.ones((1,queries.shape[2]))
        T = tf.cast(T,tf.float32)
        queries = tf.multiply(queries, T)

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs2 = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs2 = tf.matmul(outputs2, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs2 = tf.concat(tf.split(outputs2, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs2 += queries

        # Normalize
        outputs2 = normalize(outputs2)  # (N, T_q, C)

    return outputs2,outputs

def multihead_attention_time_mask(queries, keys, num_units=None, num_heads=4, dropout_rate=0, is_training=True,
                             causality=True, scope="multihead_attention", reuse=None, T_input=None):
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        T = tf.reshape(T_input,(-1,queries.shape[1],1))
        T = T*np.ones((1,queries.shape[2]))
        T = tf.cast(T,tf.float32)
        queries = tf.multiply(queries, T)

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            #########Causality=True#########
            tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        #outputs2 = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        # outputs2=tf.transpose(outputs2, perm=[0,2,1,3])

        outputs2 = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
        print(tf.shape(outputs2))

        # Restore shape
        outputs2 = tf.concat(tf.split(outputs2, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs2 += queries

        # Normalize
        outputs2 = normalize(outputs2)  # (N, T_q, C)

    return outputs2,outputs

def point_process(context,is_training=False):
    b = tf.Variable(tf.random_uniform([hp.output_unit, 1], -1.0, 1.0))
    Wt = tf.Variable(tf.random_uniform([hp.output_unit, 1], -1.0, 1.0))
    Wh = tf.Variable(tf.random_uniform([hp.output_unit, hp.maxlen*hp.hidden_units, 1], -1.0, 1.0))

    #(-1,maxlen,hidden_units)
    #print(tf.shape(context))
    context = tf.reshape(context,[-1,hp.maxlen*hp.hidden_units])
    context = tf.expand_dims(context,1)
    #(-1,38,maxlen*hidden_units)
    context = tf.tile(context,[1,hp.output_unit,1])
    lambda_all_0 = tf.squeeze(tf.matmul(tf.expand_dims(context,-1),tf.tile(tf.expand_dims(Wh,0),[tf.shape(context)[0],1,1,1]),transpose_a=True),-1) +b


def point_process_loss(batch_y,batch_t,lambda_all_0,Wt):

    loss_time = tf.log(tf.reduce_sum(tf.exp(lambda_all_0+tf.matmul(batch_t,Wt,transpose_b=True)),axis=0))
    #loss_time = tf.scan(lambda a,t: tf.log(tf.reduce_sum(tf.exp(lambda_all_0+tf.multiply(Wt,t)),axis=0)),batch_t)

    loss_event = tf.reduce_sum(tf.scan(lambda a,(event,t): tf.multiply((tf.exp(lambda_all_0+tf.multiply(Wt,t))[event]-tf.exp(lambda_all_0)[event]),1/Wt[event]), (batch_y,batch_t)),axis=1)
    return tf.reduce_sum(loss_time-loss_event)




def label_smoothing(inputs, epsilon=0.1):
    K = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)
