# -*- coding: utf-8 -*-
# /usr/bin/python2

__author__ = "Luo Yifan"
__version__ = "0.2"

class Hyperparams:

    # data
    #这部分数据集用在train.py中
   # train_dataset_file = '/home/luoyifan/dataset/nsh/sanhuangua/transformer/v_3/dataset/train/'
   # test_dataset_file = '/home/luoyifan/dataset/nsh/sanhuangua/transformer/v_3/dataset/test/'
    
    #这部分数据集用在Transformer_latent_vector.py中
    #FILE_PATH = '/home/zhongqian/yuyi-prediction-dataset/nsh2/newplayer/'
#    FILE_PATH = ''
    #normal_dataset_file_train = '/home/luoyifan/dataset/nsh/sanhuangua/transformer/v_3/dataset/normal_train/'
    #plug_dataset_file_train = '/home/luoyifan/dataset/nsh/sanhuangua/transformer/v_3/dataset/plugin_train/'
    #normal_dataset_file_test = '/home/luoyifan/dataset/nsh/sanhuangua/transformer/v_3/dataset/normal_test/'
    #plug_dataset_file_test = '/home/luoyifan/dataset/nsh/sanhuangua/transformer/v_3/dataset/plugin_test/'

#    X_file_train = FILE_PATH + 'all_datatrain_seq' + str(maxlen + 1) + '.pkl'
#    y_file_train = FILE_PATH + 'all_labeltrain_seq' + str(maxlen + 1) + '.pkl'  
#    X_file_test = FILE_PATH + 'all_datatest_seq' + str(maxlen + 1) + '.pkl'
#    y_file_test = FILE_PATH + 'all_labeltest_seq' + str(maxlen + 1) + '.pkl'

    # model_file log_file
    transformer_model_file = 'model/transformer/'  # 模型存储文件
    log_file = 'log/'

    logid_freq_file = ''
    logdesignid_freq_file = '/home/luoyifan/dataset/nsh/sanhuangua/transformer/v_2/freq_id/logid_freq'
    
    #log file dir
    log_file_dir = 'log/train_log.log'

    # training
    inter_op_parallelism_threads=5
    intra_op_parallelism_threads=5

    # model
    vocab_size = 1000
    min_cnt = 3  # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 64
    num_blocks = 1  # number of encoder/decoder blocks
    num_heads = 8
    dropout_rate = 0.2
    sinusoid = False
    exp_constant = 20
    time_scale = 1

    # other
    display_step = 1  # Check training loss after every display_step batches
    saver_step = 1000
    print_model = transformer_model_file+'epoch20batch%s.ckpt'%(str(saver_step))

#FILE_PATH = '/home/zhongqian/yuyi-prediction-dataset/nsh2/newplayer/'
    #normal_dataset_file_train = '/home/luoyifan/dataset/nsh/sanhuangua/transformer/v_3/dataset/normal_train/'
    #plug_dataset_file_train = '/home/luoyifan/dataset/nsh/sanhuangua/transformer/v_3/dataset/plugin_train/'
    #normal_dataset_file_test = '/home/luoyifan/dataset/nsh/sanhuangua/transformer/v_3/dataset/normal_test/'
    #plug_dataset_file_test = '/home/luoyifan/dataset/nsh/sanhuangua/transformer/v_3/dataset/plugin_test/'

    ########################mimic mimic mimic mimic#################
    # output_unit=76
    # # training
    # batch_size = 4
    # learning_rate = 0.001
    # maxlen = 3
    # FILE_PATH = 'data/mimic/'
    # epochs = 50
    # train_size = 440
    # test_size = 135

    ########################mimic_fold mimic_fold mimic_fold mimic_fold#################
    # output_unit=76
    # # training
    # batch_size = 4
    # learning_rate = 0.001
    # maxlen = 3
    # FILE_PATH = 'data/mimic_fold/'
    # epochs = 30

    # ########################meme meme meme meme meme meme#################
    output_unit=5000
    vocab_size = 10000
    batch_size = 64
    learning_rate = 0.0016
    maxlen = 3  # max length of Pad Sequence
    FILE_PATH = 'data/data_meme/'
    epochs = 30
    train_size=70011
    test_size=6374
    ########################reweet reweet reweet reweet reweet#################
    # output_unit = 3
    # batch_size = 128
    # learning_rate = 0.002
    # maxlen = 3  # max length of Pad Sequence
    # FILE_PATH = 'data/data_reweet/'
    # epochs = 30
    # train_size = 70133
    # test_size = 30019

    ########################so so so so#################
    # output_unit=23
    # batch_size = 128
    # learning_rate = 0.0014
    # maxlen = 3  # max length of Pad Sequence
    # FILE_PATH = 'data/so_rmtpp/'
    # epochs = 10
    # # train_size = 332683
    # # 367260
    # # 300182
    # train_size = 367260
    # # test_size = 121199
    # # 93255
    # # 107269
    # test_size = 93255

    ########################finance finance finance finance#################
    # output_unit=2
    # batch_size = 16
    # learning_rate = 0.001
    # maxlen = 3  # max length of Pad Sequence
    # FILE_PATH = 'data/finance/'
    # epochs = 20
    # train_size = 278592
    # test_size = 119424


    ########################nsh nsh nsh nsh#################
    # output_unit=38
    # batch_size = 128
    # learning_rate = 0.002
    # maxlen = 10  # max length of Pad Sequence
    # FILE_PATH = 'data/'
    # epochs = 30
    # train_size = 72450
    # test_size = 30084


    X_file_train = FILE_PATH + 'all_datatrain_seq' + str(maxlen + 1) + '.pkl'
    Time_file_train = FILE_PATH + 'all_timetrain_seq' + str(maxlen + 1) + '.pkl'
    Time_gap_file_train = FILE_PATH + 'all_timetrain_gap_seq' + str(maxlen + 1) + '.pkl'
    Time_file_train_label = FILE_PATH + 'all_timetrain_label_seq' + str(maxlen + 1) + '.pkl'
    Time_file_train_raw = FILE_PATH + 'all_timetrain_raw_seq' + str(maxlen + 1) + '.pkl'
    y_file_train = FILE_PATH + 'all_labeltrain_seq' + str(maxlen + 1) + '.pkl'
    X_file_test = FILE_PATH + 'all_datatest_seq' + str(maxlen + 1) + '.pkl'
    Time_file_test = FILE_PATH + 'all_timetest_seq' + str(maxlen + 1) + '.pkl'
    Time_gap_file_test = FILE_PATH + 'all_timetest_gap_seq' + str(maxlen + 1) + '.pkl'
    Time_file_test_label = FILE_PATH + 'all_timetest_label_seq' + str(maxlen + 1) + '.pkl'
    Time_file_test_raw = FILE_PATH + 'all_timetest_raw_seq' + str(maxlen + 1) + '.pkl'
    y_file_test = FILE_PATH + 'all_labeltest_seq' + str(maxlen + 1) + '.pkl'


