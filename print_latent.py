import tensorflow as tf
from hyperparams import Hyperparams as hp
import pickle
from keras.utils import np_utils
from data_load import *

class tfModel:
    def __init__(self, modelname='transformer'):
        self.graph = tf.Graph()
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(hp.print_model+'.meta')
        self.sess = tf.Session(graph=self.graph)
        with self.sess.as_default():
            with self.graph.as_default():
                saver.restore(self.sess, hp.print_model)

    def predict(self,input):
        #print(self.graph.get_operations())
        input_data_logdesignid_enc = self.graph.get_operation_by_name('inputs/input_logdesignid_enc').outputs[0]
        time = self.graph.get_operation_by_name('inputs/times').outputs[0]
        target = self.graph.get_operation_by_name('inputs/targets').outputs[0]
        is_training = self.graph.get_operation_by_name('inputs/is_training').outputs[0]
        enc1 = self.graph.get_collection("latent_enc1",scope='optimization')[0]
        enc2 = self.graph.get_collection("latent_enc2",scope='optimization')[0]
        enc3 = self.graph.get_collection("latent_enc3",scope='optimization')[0]
        logits = self.graph.get_collection("latent_logits",scope='optimization')[0]
        enc1_,enc2_,enc3_,logits_ = self.sess.run([enc1,enc2,enc3,logits], feed_dict={input_data_logdesignid_enc: input[0],
                                                                            target: input[1],
                                                                            is_training:False,
                                                                            time: input[2]})
        print('enc1',enc1_.shape)
        print('enc2',enc2_.shape)
        print('enc3',enc3_.shape)
        print('logits',logits_.shape)
        return enc1_,enc2_,enc3_,logits_

if __name__ == '__main__':
    p = tfModel()
    datatype='train'
    if datatype == 'train':
        X_file = open(hp.X_file_train, 'rb')
        x_list = pickle.load(X_file)
        Time_file = open(hp.Time_file_train_raw, 'rb')
        Time_list = pickle.load(Time_file)
        y_file = open(hp.y_file_train, 'rb')
        y_list = pickle.load(y_file)
    else:
        X_file = open(hp.X_file_test, 'rb')
        x_list = pickle.load(X_file)
        Time_file = open(hp.Time_file_test_raw, 'rb')
        Time_list = pickle.load(Time_file)
        y_file = open(hp.y_file_test, 'rb')
        y_list = pickle.load(y_file)

    X_ndarray = np.array(x_list)
    Time_ndarray = np.array(Time_list)
    y_ndarray = np_utils.to_categorical(y_list, hp.output_unit)
    (pad_enc_valid_logdesignid_batch, valid_targets_batchs, valid_times_batchs) = (x_list, y_ndarray, Time_ndarray)

    enc1_,enc2_,enc3_,logits_ = p.predict([pad_enc_valid_logdesignid_batch, valid_targets_batchs, valid_times_batchs])
    L=[]
    for i in range(0,len(pad_enc_valid_logdesignid_batch)):
        L.append('\t'.join([','.join([str(x) for x in pad_enc_valid_logdesignid_batch[i]])
                           ,','.join([str(x) for x in valid_times_batchs[i]])
                           ,str(np.argmax(valid_targets_batchs[i]))
                           ,'@'.join([','.join([str(z) for z in y]) for y in enc1_[i]])
                           ,'@'.join([','.join([str(z) for z in y]) for y in enc2_[i]])
                           ,'@'.join([','.join([str(z) for z in y]) for y in enc3_[i]])
                           ,','.join([str(z) for z in logits_[i]])]))
    with open('data/output/all_latenttrain_seq'+str(hp.maxlen+1)+'.txt','w') as f:
        f.write('\n'.join(L))




