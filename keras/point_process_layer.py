from keras.layers import Multiply
from keras import backend as K
from keras.engine.topology import Layer
from keras.engine import InputSpec
import numpy as np
import tensorflow as tf
from hyperparams import Hyperparams as hp
try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations

class CustomPPLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        self.b_init = initializations.RandomNormal()
        self.wt_init = initializations.RandomNormal()
        self.wh_init = initializations.RandomNormal()
        super(CustomPPLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=(-1,10)),InputSpec(shape=(-1,10)),InputSpec(shape=(-1,38))]
        self.b = K.variable(self.b_init((38,1)), name='{}_b'.format(self.name))
        self.wt = K.variable(self.wt_init((38,1)), name='{}_wt'.format(self.name))
        self.wh = K.variable(self.wh_init((38,hp.maxlen*hp.hidden_units,1)), name='{}_wh'.format(self.name))
        self.trainable_weights = [self.b, self.wt,self.wh]

    def call(self,inputs):
        batch_y = inputs[0]
        batch_t = inputs[1]
        batch_y = tf.cast(batch_y, tf.float32)
        batch_t = tf.cast(batch_t, tf.float32)
        context = inputs[2]
        context = K.reshape(context,[-1,hp.maxlen*hp.hidden_units])
        context = K.expand_dims(context,1)
        context = K.tile(context,[1,38,1])
        lambda_all_0 = tf.squeeze(K.batch_dot(K.expand_dims(context,-1),K.tile(K.expand_dims(self.wh,0),[K.shape(context)[0],1,1,1]),axes=[2,2]),-1) +self.b

        #loss_time = tf.scan(lambda a,t: K.log(K.sum(K.exp(lambda_all_0+tf.multiply(self.wt,t[0])),axis=0)),batch_t,infer_shape=False)
        loss_time = tf.log(tf.reduce_sum(tf.exp(lambda_all_0+tf.matmul(batch_t,self.wt,transpose_b=True)),axis=0))
        print(loss_time.get_shape())
        loss_event = K.sum(tf.scan(lambda a,(event,t): tf.multiply((K.exp(lambda_all_0+K.multiply(self.wt,t[0]))[event]-K.exp(lambda_all_0)[event]),1/self.wt[event]), (batch_y,batch_t)),axis=1)
        loss = tf.reduce_mean(loss_time-loss_event)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return batch_y


