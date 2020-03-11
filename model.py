import tensorflow as tf
from keras.layers import Conv3D,BatchNormalization,MaxPooling3D,Flatten,Dense,Dropout,Input,concatenate
from keras.layers import GlobalAveragePooling3D
from keras.layers import Activation, LeakyReLU,ReLU
from keras.regularizers import l2
from keras.initializers import he_normal, lecun_normal
from keras import Model
import os
from keras import regularizers

def _conv_block(x,filters,idx,k_size=(3,3,3),
             c_strd=(3,3,3),p_strd=(1,2,2),p_size=(1,2,2),
             initial='he_normal',
             pad='same'):
    c = Conv3D(filters,k_size,padding=pad,kernel_initializer=initial,name='c_b_{}'.format(idx))(x)
    b = BatchNormalization(axis=-1,name='b_b_{}'.format(idx))(c)
    a = ReLU()(b)
    p = MaxPooling3D(p_strd,p_strd,padding=pad,name='m_b_{}'.format(idx))(a)
    
    return p

def Arch(model_input,
         block_nb=6,
         include_top=True,
         drop_rate=0.5,
         classes=2,
         theta=0.001):
    '''theta represents L2 nrom number'''
    
    if isinstance(model_input,tuple): model_input = Input(shape=model_input)
    
    if block_nb==6:
        x = _conv_block(model_input,filters=64,idx='1_1')
        x = _conv_block(x,filters=64,p_strd=(1,2,2),idx='1_2')
    else:
        x = _conv_block(model_input,filters=64,p_strd=(1,2,2),idx='1_2')
    
    if block_nb >= 5:
        x = _conv_block(x,filters=128,idx='2_1')
    x = _conv_block(x,filters=128,p_strd=(2,2,2),idx='2_2')
    
    
    if block_nb>=4:
        x = _conv_block(x,filters=256,idx='3_1')
    x = _conv_block(x,filters=256,p_strd=(2,2,2),idx='3_2')
    
    x = GlobalAveragePooling3D()(x)

    
    if include_top:
        x = Dropout(drop_rate)(x)
        x = Dense(4096,kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(theta))(x)
        x = BatchNormalization()(x)
        x = Dropout(drop_rate)(x)
        
    out_activation = 'softmax' if classes==2 else 'sigmoid'
    x = Dense(classes,activation=out_activation,name='predict')(x)
    
    model = Model(inputs=model_input,outputs=x)
    
    return model



def model_Haoliang_Li(model_input,classnb,conv_block_nb=5,dense_or_gap='gap'):
    '''
    default conv_block_nb as 5
    default dense_or_gap as gap
    '''
        
    def block(x,nbfilter,conv_size=(3,3,3),conv_strides=(1,1,1),p_size=(1,2,2),p_strides=(1,2,2)):
        x = Conv3D(nbfilter,conv_size,strides=conv_strides,padding='same',kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling3D(p_size,strides=p_strides,padding='same')(x)
        return x
    if isinstance(model_input,tuple):
        input_layer = Input(shape=model_input)
    else:
        input_layer = Input(tensor=model_input)
    # 1st block
    x = block(input_layer,128)
    # 2nb block
    x = block(x,128)
    # 3rd block
    x = block(x,128)
    
    for i in range(3,conv_block_nb):
        x = block(x,128,p_size=(2,2,2),p_strides=(2,2,2))

    if dense_or_gap == 'gap':
        x = GlobalAveragePooling3D(name='feature_layer')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    elif dense_or_gap == 'dense':
        x = Flatten()(x)    
        x = Dense(1024,activation='linear',kernel_initializer='he_normal',kernel_regularizer=l2(1e-4),name='feature_layer')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    else:
        raise ValueError('unsupported architecture, only accepted dense layer or GAP layer!!!')
    
    x = Dropout(rate=0.5)(x)
    
    out_activation = 'sigmoid' if classnb == 1 else 'softmax'
    x = Dense(classnb,activation=out_activation,name='predict')(x)
    
    model = Model(inputs=input_layer,outputs=x)
    return model