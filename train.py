import tensorflow as tf
import os
from os.path import join, exists

import keras.backend as K
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy
from keras.layers import Input

from tfrecord_server import decode_tfrecord
from model import Arch,model_Haoliang_Li

from glob import glob
import numpy as np
import pandas as pd
import math
import argparse
import configparser

from metrics_tools import get_eer, get_hter, get_accuracy, get_loss, get_auc, group_chunk_pred_to_ori_video


def train(**kwargs):

    interval = kwargs['interval']
    model_depth = int(kwargs['model_depth']) #模型深度【4,5,6,7】
    batch_size = int(kwargs['batch_size']) 
    epoch = int(kwargs['epoch'])
    gpuId = str(kwargs['gpuid'])
    framenb = kwargs['framenb'] #模型输入的深度
    classnb = int(kwargs['classnb']) #输出类别数，【1,2】
    dense_or_gap = kwargs['dense_or_gap'] #使用Dense层还是GAP层
    do_validation = bool(int(kwargs['do_validation'])) #CASIA数据集不采用validation，Replayattack采用validation
    dataset = str(kwargs['dataset'])
    train_sample_nb = int(kwargs['train_sample_nb'])
    train_sample_nb_one_epoch = int(kwargs['train_sample_nb_one_epoch'])
    test_sample_nb = int(kwargs['test_sample_nb'])
    
    train_tfrecord = './tfrecord/{}/frames-{}/frames-{}-{}-interval-{}-{}.tfrecord'.format(dataset,framenb,framenb,'train',interval,train_sample_nb)
    
    test_tfrecord = train_tfrecord.replace('train','test').replace(str(train_sample_nb),str(test_sample_nb))
    
    tfrecord_sample_csv = './tfrecord/{}/frames-{}/frames-{}-interval-{}.csv'.format(dataset,framenb,framenb,interval)
    sample_df = pd.read_csv(tfrecord_sample_csv)
    
    if do_validation:
        devel_sample_nb = int(kwargs['devel_sample_nb'])
        devel_tfrecord = train_tfrecord.replace('train','devel').replace(str(train_sample_nb),str(devel_sample_nb))
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuId
    
    savedir = 'train-results/{}/depth-{}-{}/frames-{}/'.format(dataset,model_depth,dense_or_gap,framenb)
    if not os.path.exists(savedir): os.makedirs(savedir)
    
    
    if do_validation: 
        monitor = 'devel_eer'
    else:
        monitor = 'test_eer'
    
    
    #Callbacks
    callback_csv_logger = CSVLogger(savedir+'/training.log')
    callback_ckp_by_perform = ModelCheckpoint(savedir+'/{epoch:03d}-{'+monitor+':.3f}.hdf5',
                                       verbose=1,
                                       monitor=monitor,
                                       save_best_only=True,
                                       mode='min')

    callback_lr_decay = ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=7, verbose=0, mode='min', cooldown=0, min_lr=1e-4)
    callback_early_stopping = EarlyStopping(patience=15,monitor=monitor)
    
    callback_list = [callback_csv_logger,callback_ckp_by_perform,callback_lr_decay,callback_early_stopping]
    
    class EvaluateInputTensor(Callback):
        def __init__(self,model,steps,df,metrics_prefix='devel',verbose=1):
            super(EvaluateInputTensor, self).__init__()
            self.devel_model = model
            self.verbose = verbose
            self.num_steps = steps
            self.df = df
            self.metrics_prefix = metrics_prefix

        def on_epoch_end(self, epoch, logs={}):
            self.devel_model.set_weights(self.model.get_weights())
            y_pred = self.devel_model.predict(None,None,steps=int(self.num_steps),verbose=self.verbose)
            y_true, y_pred = group_chunk_pred_to_ori_video(self.df,y_pred,'devel')
            acc = get_accuracy(self.label,y_pred)
            pred_loss = get_loss(self.label,y_pred)
            equal_error_rate, thd = get_eer(self.label[:len(y_pred)],y_pred)
            auc = get_auc(self.label,y_pred)
            lr = K.eval(self.model.optimizer.lr)
            
            metrics_names = ['acc','loss','eer','auc','lr','thd']
            results = [acc,pred_loss,equal_error_rate,auc,lr,thd]
            
            metrics_str = ' '
            for result, name in zip(results, metrics_names):
                metric_name = self.metrics_prefix + '_' + name
                logs[metric_name] = result
                if self.verbose > 0:
                    metrics_str = metrics_str + metric_name + ': ' + str(result) + ' '
            if self.verbose > 0:
                print(metrics_str) 
                
    class TestInputTensor(Callback):
        def __init__(self, test_model,test_steps,df,metrics_prefix='test',verbose=1):
            super(TestInputTensor, self).__init__()
            self.test_model = test_model
            self.verbose = verbose
            self.test_steps = test_steps
            self.df = df
            self.metrics_prefix = metrics_prefix

        def on_epoch_end(self, epoch, logs={}):
            metrics_names = ['acc','loss','hter','auc']
            if int(epoch)%3!=0:
                self.verbose=0
                results = np.asarray((np.inf,np.inf,np.inf,np.inf))
                self.verbose=1
            else:
                eer_thd = float(logs['devel_thd'])
                print('eer_thd is {}'.format(eer_thd))
                self.test_model.set_weights(self.model.get_weights())
                y_pred = self.test_model.predict(None, None, steps=int(self.test_steps),verbose=self.verbose)
                y_true, y_pred = group_chunk_pred_to_ori_video(self.df,y_pred,'devel')
                
                acc = get_accuracy(y_true,y_pred)
                pred_loss = get_loss(y_true,y_pred)
                hter = get_hter(y_true,y_pred,eer_thd)
                auc = get_auc(y_true,y_pred,eer_thd)
                results = [acc,pred_loss,hter,auc]
                
            metrics_str = ' '
            for result, name in zip(results, metrics_names):
                metric_name = self.metrics_prefix + '_' + name
                logs[metric_name] = result
                if self.verbose > 0:
                    metrics_str = metric_name + ': ' + str(result) + ' ' +metrics_str
            if self.verbose > 0:
                print(metrics_str)
    
    
    loss_str = 'categorical_crossentropy' if classnb==2 else 'binary_crossentropy'
    
    loss_monitor = {'predict':loss_str}
    metrics_monitor = {'predict':['acc']}
    
    one_hot = True if classnb ==2 else False
    
    train_data, train_label = decode_tfrecord(filenames=train_tfrecord,normalize_method=0,batch_size=batch_size,one_hot=one_hot,perform_shuffle=True)
    train_model = model_Haoliang_Li(model_input=Input(tensor=train_data),classnb=classnb,conv_block_nb=model_depth,dense_or_gap=dense_or_gap)
    train_target_tensor = {'predict':train_label}
    train_model.compile(optimizer=Adam(lr=1e-3,decay=1e-7),loss=loss_monitor,metrics=metrics_monitor,target_tensors=train_target_tensor)
    # 固定每次训练的样本数量，为了加速收敛，我们规定每次训练样本点为6000（使用ReplayAttack时原训练样本点数量为12000，减为一半）
    train_steps_per_epoch = train_sample_nb_one_epoch//batch_size
    
    
    test_data, test_label = decode_tfrecord(filenames=test_tfrecord,normalize_method=0,batch_size=batch_size,one_hot=one_hot) 
    test_model = model_Haoliang_Li(model_input=Input(tensor=test_data),classnb=classnb,conv_block_nb=model_depth,dense_or_gap=dense_or_gap)
    test_target_tensor = {'predict':test_label}
    test_model.compile(optimizer=Adam(lr=1e-3),loss=loss_monitor,metrics=metrics_monitor,target_tensors=test_target_tensor)
    test_steps_per_epoch = test_sample_nb//batch_size
    
    if do_validation:
        devel_data, devel_label = decode_tfrecord(filenames=devel_tfrecord,normalize_method=0,batch_size=batch_size,one_hot=one_hot) 
        devel_model = model_Haoliang_Li(model_input=Input(tensor=devel_data),classnb=classnb,conv_block_nb=model_depth,dense_or_gap=dense_or_gap)
        devel_steps_per_epoch = devel_sample_nb//batch_size
        devel_target_tensor = {'predict':devel_label}
        devel_model.compile(optimizer=Adam(lr=1e-3),loss=loss_monitor,metrics=metrics_monitor,target_tensors=devel_target_tensor)

        callback_devel = EvaluateInputTensor(model=devel_model,
                                             steps=devel_steps_per_epoch,
                                             df=sample_df)

        callback_test = TestInputTensor(test_model=test_model,
                                        test_steps=test_steps_per_epoch,
                                        df=sample_df,
                                        metrics_prefix='test')
                                        
        callback_list.insert(0,callback_devel)
        callback_list.insert(1,callback_test)
    else:
        callback_test = EvaluateInputTensor(model=test_model,
                                            steps=test_steps_per_epoch,
                                            metrics_prefix='test',
                                            df=sample_df)
        callback_list.insert(0,callback_test)
    
    train_model.summary()
    
    print('start training...')
    history = train_model.fit(steps_per_epoch=train_steps_per_epoch,epochs=epoch,verbose=1,callbacks=callback_list)

    train_model.save_weights(savedir+'/final_model.hdf5')
    pd.to_pickle(history,savedir+'/history.pkl')
    
    return history


if __name__ == '__main__':
    ## 管理参数，创建实例
    parser = argparse.ArgumentParser(description='configuration file')
    #  '-conf': 配置文件
    parser.add_argument('-conf',help='configure file')
    arg = parser.parse_args()
    config_filename = arg.conf
    
    #生成config对象
    config = configparser.ConfigParser()
    #用config对象读取配置文件  
    config.read(config_filename)
    
    input_dict = {}
    
    # print(config.sections())
    section = config[config.sections()[0]]

    for key in section:
        input_dict[key] = section[key]
    print(input_dict)
    hisory = train(**input_dict)