import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import os
# import imageio
import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from os.path import splitext, basename
import matplotlib.pyplot as pl

'========================tfrecord encoder================================='

def data_to_tfrecord(total_video_crop_csv,to_file=None):
    """for each csv file contains train,test,devel information we write tfrecord of each subset, 
    first shuffle each subset, then in each subset write tfrecord line by line"""

    """
    extract key information, 
    frame_nb represents how many frames will be extracted in in each data point
    interval represents the gap between two sampled frames
    """
    frame_nb = int(basename(total_video_crop_csv).split('_')[1])
    interval = int(basename(total_video_crop_csv).split('_')[-1].split('.')[0])
    
    df = pd.read_csv(total_video_crop_csv)
    
    
    for subset_flag in ['train','devel','test']:
        """shuffle"""
        shuffled_subset = _pd_shuffle(df,subset_flag)
        """write a tfrecord of each subset"""
        subset_to_tfrecord(subset=shuffled_subset,
                           subset_flag=subset_flag,
                           frame_nb=frame_nb,
                           interval=interval,
                           to_file=to_file)

def subset_to_tfrecord(subset,
                       subset_flag,
                       frame_nb,
                       interval,
                       total_vdname_csv=None,
                       to_file=None,
                       cvt_format=None,
                       temp_length=None):
    """iterate over one of train test and devel set to get tfrecord,
        subset shold be shuffled already with a type of pd.Dataframe"""
    
    if total_vdname_csv==None: 
        total_vdname_csv = 'E:/Xuhang/code/Graduation/replayattack_video_names_NO_EDITTING.csv'
        
    if to_file==None: to_file = 'E:/Xuhang/code/Graduation/replayattack-baseline/'
    
    if cvt_format==None: cvt_frmat='.jpg'
        
    video_list = list(pd.read_csv(total_vdname_csv).iloc[:,1].to_numpy())
    
    sample_nb = subset.shape[0] if temp_length is None else temp_length
    
    save_file = to_file+'frames-{}-{}-interval-{}-{}.tfrecord'.format(frame_nb,subset_flag,interval,sample_nb)
    
    with tf.python_io.TFRecordWriter(save_file) as writer: 
        for i in tqdm(range(sample_nb)):
            line = subset.iloc[i]
            video_name = video_list[int(line.video_index)]
            line_example = line_to_example(line=line,
                                          video_name=video_name,
                                          frame_nb=frame_nb,
                                          interval=interval,
                                          cvt_fmt=cvt_format)
            writer.write(line_example)
        writer.close()

def line_to_example(line,video_name,frame_nb,cvt_fmt,interval):
    """cvt one row of Dataframe to tfrecord serialized example"""
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def read_by_opencv(video_name,start_idx,frame_nb,interval):
        cap = cv2.VideoCapture(video_name)
        cap.set(cv2.CAP_PROP_POS_FRAMES,start_idx)
        i,frames = 0,[]
        while i < frame_nb:
            ret, frame = cap.read()
            i += 1
            if i%interval==0:frames.append(frame[:,:,::-1])
        cap.release()
        return np.asarray(frames)
    
    def read_by_imageio(video_name,start_idx,frame_nb,interval):
        reader = imageio.get_reader(video_name)
        frames = [reader.get_data(i) for i in range(start_idx,start_idx+frame_nb*interval,interval)]
        reader.close()
        return np.asarray(frames)
    
    """read video by start frame and extract frames by interval"""
    def read_video_by_start_idx(video_name,start_idx,frame_nb,interval):
        if '.mov' in video_name:
            return read_by_imageio(video_name,start_idx,frame_nb,interval)
        else:
            try:
                # cv2 read video faster than imageio but cv2 is not that stable
                frames = read_by_opencv(video_name,start_idx,frame_nb,interval)
            except:
                frames = read_by_imageio(video_name,start_idx,frame_nb,interval)
        return np.asarray(frames)
    
    """convert information of each row from csv_file as tfrecord sample"""
    frames = read_video_by_start_idx(video_name,line.start_idx,frame_nb,interval)
    top,left,bottom,right = line.top,line.left,line.bottom,line.right
    chunk_faces = frames[:,top:bottom,left:right]
    
    line_feature = {}
    line_feature['height'] = _int64_feature(chunk_faces.shape[1])
    line_feature['width'] = _int64_feature(chunk_faces.shape[2])
    line_feature['depth'] = _int64_feature(chunk_faces.shape[3])
    line_feature['video_index'] = _int64_feature(int(line.video_index))
    line_feature['start_idx'] = _int64_feature(line.start_idx)
    line_feature['label'] = _int64_feature(line.real_spoof)
    for count, face in enumerate(chunk_faces):
        face_raw = face.tostring()
        line_feature['frame/{:04d}'.format(count)] = _bytes_feature(face_raw)
    
    example = tf.train.Example(features=tf.train.Features(feature=line_feature))
    return example.SerializeToString()
    
'==========================tfrecord decode.....=================================='
def decode_tfrecord(filenames,
                    batch_size=32,
                    normalize_method=2,
                    perform_shuffle=False,
                    repeat_count=None,
                    one_hot=False):
    '''
    input:  filenames--tfrecord file name
            perform_shuffle--True means shuffle, False means no shuffle
            repeat_count--interger, repeat times in an epoch through the whole dataset, if None, forever loop through dataset
            batch_size--no need to explain
            fp_name--face points, only accept parameters from list of ['left_eye','right_eye','nose','left_mouth_corner','right_mouth_corner']
    output: an tuple, has three values which are labels, facial points and video sequences
            shape is (labels, facial pints, video sequences)
            for each output, shape is (batch_size, XX.shape), XX means one of labels, fpoints and video sequences
    '''
    
    frame_nb = int(basename(filenames).split('-')[1])
    
    def normalize_tensor(tensor,method=2):
        '''three normalize methods will be implement here
            tensor shape is [frame_nb, height, width, channels]
            1. use tensorflow API tf.image.per_image_standardization for each image of input tensor
            2. normalize all frames using mean and stddev of first frame of input tensor
            3. normalize all frames using mean and stddev of all frames of input tensor
        '''
        if not tensor.dtype==tf.float32:  tensor = tf.cast(tensor,tf.float32)

        def norm_func(frame):
            v_max, v_min = tf.reduce_max(frame), tf.reduce_min(frame)
            frame = (frame-v_min)/(v_max-v_min)
            return frame
        
        if method==1: 
            normalized = tf.image.per_image_standardization(tensor)
            
        elif method==2: 
            mean, variance = tf.nn.moments(tensor,axes=[0,1,2,3])
            stddev = tf.sqrt(variance)
            normalized = (tensor-mean)/stddev
            
        else: raise ValueError('the method you are using is not considered, please choose from 0-2 !!!')

        normalized = tf.map_fn(fn=lambda frame:norm_func(frame), elems=normalized,dtype=tf.float32)

        return tf.squeeze(normalized)
    
    
    def decode_fn(serialized_example):
        line_feature = {}
        line_feature['label'] = tf.FixedLenFeature([], tf.int64)
        line_feature['height'] = tf.FixedLenFeature([], tf.int64)
        line_feature['width'] = tf.FixedLenFeature([], tf.int64)
        line_feature['depth'] = tf.FixedLenFeature([], tf.int64)
        for count in range(frame_nb):
            line_feature['frame/{:04d}'.format(count)] = tf.FixedLenFeature([],tf.string)

        parsed_features = tf.parse_single_example(serialized_example, line_feature)

        label = parsed_features['label']
        height = parsed_features['height']
        width = parsed_features['width']
        depth = parsed_features['depth']
        
        face_images = [tf.decode_raw(parsed_features['frame/{:04d}'.format(count)],tf.uint8) for count in range(frame_nb)]
        face_images = [tf.reshape(face,[height,width,depth]) for face in face_images]
        face_images = [tf.image.resize_nearest_neighbor(tf.expand_dims(face,axis=0),[128,128]) for face in face_images]
        face_images = tf.squeeze(face_images)
        if normalize_method == 0:
            face_images = tf.image.convert_image_dtype(face_images,tf.float32)
        else:
            face_images = normalize_tensor(face_images,method=normalize_method)
        face_images = tf.reshape(face_images,[frame_nb,128,128,3])
        

        if one_hot: 
            label = tf.one_hot(label,2)
        else:
            label = tf.expand_dims(label,axis=-1)
            label = tf.cast(label, tf.float32)
        return face_images, label

    dataset = tf.data.TFRecordDataset(filenames=filenames,num_parallel_reads=10)
    
    dataset = dataset.map(decode_fn)
    if perform_shuffle: dataset = dataset.shuffle(buffer_size=500)
    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(batch_size)  # Batch size to use
    dataset = dataset.prefetch(8)
    iterator = dataset.make_one_shot_iterator()
    
    return iterator.get_next()