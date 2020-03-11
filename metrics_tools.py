import numpy as np
from os.path import basename
from sklearn.metrics import accuracy_score, log_loss, roc_curve, roc_auc_score


def group_chunk_pred_to_ori_video(df,y_pred,sub_folder,seed=1994):
    sub_df = df[df.sub_folder==sub_folder].sample(frac=1,random_state=seed)
    if len(y_pred.shape) > 1: y_pred = y_pred[:,1]
    pred_df = pd.DataFrame(y_pred,columns=['y_pred'])
    df_concat = pd.concat([sub_df.reset_index(drop=True),pred_df.reset_index(drop=True)],axis=1)
    df_mean = df_concat[['video_index','real_spoof','y_pred']].groupby('video_index').mean()
    y_true = df_mean[['real_spoof']]
    y_pred = df_mean[['y_pred']]
    return y_true, y_pred
    
    
def get_tfrecord_sample_nb(tfrecord):
    return int(basename(tfrecord).split('.')[0].split('-')[-1])

def get_eer(y_true,y_pred):
    '''https://github.com/maciej3031/whale-recognition-challenge/blob/30a9ef774ef344916bbc8dce4229fa005d346a64/utils/performance_utils.py#L310'''
    if len(y_pred.shape) >1 : y_pred = y_pred[:,-1]
    fpr, tpr, thrd = roc_curve(y_true,y_pred,pos_label=1)
    frr = 1 - tpr
    index_eer = np.argmin(abs(fpr - frr))
    eer = (fpr[index_eer] + frr[index_eer])/2
    eer_thrd = thrd[index_eer]
    return eer, eer_thrd

def get_hter(y_true,y_pred,eer_thd=None):
    '''please input eer_thd, else eer of test set will be calculated'''
    if len(y_pred.shape) >1 : y_pred = y_pred[:,-1]
    fpr, tpr, thrd = roc_curve(y_true,y_pred,pos_label=1)
    frr = 1 - tpr
    if eer_thd is None:
        index = np.argmin(abs(fpr - frr))
        hter = min((fpr[index]+frr[index])/2)
    else:
        hter_thrd_idx = np.argmin(abs(thrd-eer_thd))
        hter = (fpr[hter_thrd_idx]+frr[hter_thrd_idx])/2
    return hter

def apcer_bpcer_acer(y_true,y_pred,PAI_type_1_idxs,PAI_type_2_idxs):
    '''
    在计算APCER，BPCER和ACER时需要指出属于纸张攻击和视频攻击的indexs
    因为我们这里attack只有两种：print_attack and replay_attack(video_attack)
    故只有两种PAI_types
    
    return apcer, bpcer and acer
    '''

    bona = 0  # 活脸标签为0
    attack = 1  # 非活脸标签为1
    
    '''BPCER'''
    # step-1 找出y_true中的真脸对应的index
    true_bona_idxs = np.where(y_true==bona)[0]
    nb_true_bona = len(true_bona_idxs)
    # step-2 找出y_pred中属于step-1的index的预测标签
    pred_values_belong_to_bona = y_pred[true_bona_idxs]
    # step-3 从属于Bona的预测值里去数哪些数据是真正分类准确的
    nb_pred_bona_correctly_classified = np.where(pred_values_belong_to_bona==bona)[0]
    # step-4 BPCER
    bpcer = nb_pred_bona_correctly_classified/nb_true_bona
    
    '''APCERs'''
    # 预测标签队列里属于attack_type_1的标签
    pred_attack_type_1 = y_pred[PAI_type_1_idxs]
    # 真实标签队列中attack_type_1的样本数
    nb_true_attack_type_1 = len(PAI_type_1_idxs)
    # 预测标签队列中属于attack_type_1的标签里预测也为attack_type_1的样本数量
    nb_pred_attack_type_1_correctly_classified = len(np.where(pred_attack_type_1==attack)[0])
    # 算出属于attack_type_1的APCER
    apcer_type_1 = nb_pred_attack_type_1_correctly_classified/nb_true_attack_type_1
    
    
    # 预测attack_type_2的结果
    pred_attack_type_2 = y_pred[PAI_type_2_idxs]
    nb_true_attack_type_2 = len(PAI_type_2_idxs)
    nb_pred_attack_type_2_correctly_classified = len(np.where(pred_attack_type_2==attack)[0])
    apcer_type_2 = nb_pred_attack_type_2_correctly_classified/nb_true_attack_type_2
    
    # APCER是纸张攻击和视频攻击中值较高的那个
    apcer = max(apcer_type_1,apcer_type_2)
    
    acer = (apcer+bpcer)/2
    
    
    return apcer, bpcer, acer
    

def get_accuracy(y_true,y_pred):
    if len(y_pred.shape) >1 : y_pred = y_pred[:,-1]
    y_pred = np.round(y_pred)
    accuracy = accuracy_score(y_true,y_pred)
    return accuracy

def get_loss(y_true,y_pred):
    if len(y_pred.shape) >1 : y_pred = y_pred[:,-1]
    return log_loss(y_true,y_pred,eps=1e-7)

def get_auc(y_true,y_pred):
    if len(y_pred.shape) >1 : y_pred = y_pred[:,-1]
    return roc_auc_score(y_true,y_pred)