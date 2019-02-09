import keras
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.engine.topology import Layer
from keras import backend as K
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from utils import LRN2D
import utils
from functools import partial, update_wrapper

def get_base_model(is_training=True):
    myInput = Input(shape=(96, 96, 3))

    x = ZeroPadding2D(padding=(3, 3), input_shape=(96, 96, 3))(myInput)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    # x = tf.layers.batch_normalization(x, training=is_training, renorm=True, name='bn1', axis=3)
    x = BatchNormalization(axis=3, epsilon=0.001, name='bn1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)
    x = Lambda(LRN2D, name='lrn_1')(x)
    x = Conv2D(64, (1, 1), name='conv2')(x)
    # x = tf.layers.batch_normalization(x, training=is_training, renorm=True, name='bn2', axis=3)
    x = BatchNormalization(axis=3, epsilon=0.001, name='bn2')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(192, (3, 3), name='conv3')(x)
    # x = tf.layers.batch_normalization(x, training=is_training, renorm=True, name='bn3', axis=3)
    x = BatchNormalization(axis=3, epsilon=0.001, name='bn3')(x)
    x = Activation('relu')(x)
    x = Lambda(LRN2D, name='lrn_2')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)

    # Inception3a
    inception_3a_3x3 = Conv2D(96, (1, 1), name='inception_3a_3x3_conv1')(x)
    # inception_3a_3x3 = tf.layers.batch_normalization(inception_3a_3x3, training=is_training, renorm=True, name='inception_3a_3x3_bn1', axis=3)
    inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.001, name='inception_3a_3x3_bn1')(inception_3a_3x3)
    inception_3a_3x3 = Activation('relu')(inception_3a_3x3)
    inception_3a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)
    inception_3a_3x3 = Conv2D(128, (3, 3), name='inception_3a_3x3_conv2')(inception_3a_3x3)
    # inception_3a_3x3 = tf.layers.batch_normalization(inception_3a_3x3, training=is_training, renorm=True, name='inception_3a_3x3_bn2', axis=3)    
    inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.001, name='inception_3a_3x3_bn2')(inception_3a_3x3)
    inception_3a_3x3 = Activation('relu')(inception_3a_3x3)

    inception_3a_5x5 = Conv2D(16, (1, 1), name='inception_3a_5x5_conv1')(x)
    # inception_3a_5x5 = tf.layers.batch_normalization(inception_3a_5x5, training=is_training, renorm=True, name='inception_3a_5x5_bn1', axis=3)
    inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.001, name='inception_3a_5x5_bn1')(inception_3a_5x5)
    inception_3a_5x5 = Activation('relu')(inception_3a_5x5)
    inception_3a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)
    inception_3a_5x5 = Conv2D(32, (5, 5), name='inception_3a_5x5_conv2')(inception_3a_5x5)
    # inception_3a_5x5 = tf.layers.batch_normalization(inception_3a_5x5, training=is_training, renorm=True, name='inception_3a_5x5_bn2', axis=3)    
    inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.001, name='inception_3a_5x5_bn2')(inception_3a_5x5)
    inception_3a_5x5 = Activation('relu')(inception_3a_5x5)

    inception_3a_pool = MaxPooling2D(pool_size=3, strides=2)(x)
    inception_3a_pool = Conv2D(32, (1, 1), name='inception_3a_pool_conv')(inception_3a_pool)
    # inception_3a_pool = tf.layers.batch_normalization(inception_3a_pool, training=is_training, renorm=True, name='inception_3a_pool_bn', axis=3)
    inception_3a_pool = BatchNormalization(axis=3, epsilon=0.001, name='inception_3a_pool_bn')(inception_3a_pool)
    inception_3a_pool = Activation('relu')(inception_3a_pool)
    inception_3a_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(inception_3a_pool)

    inception_3a_1x1 = Conv2D(64, (1, 1), name='inception_3a_1x1_conv')(x)
    # inception_3a_1x1 = tf.layers.batch_normalization(inception_3a_1x1, training=is_training, renorm=True, name='inception_3a_1x1_bn', axis=3)    
    inception_3a_1x1 = BatchNormalization(axis=3, epsilon=0.001, name='inception_3a_1x1_bn')(inception_3a_1x1)
    inception_3a_1x1 = Activation('relu')(inception_3a_1x1)

    inception_3a = concatenate([inception_3a_3x3, inception_3a_5x5, inception_3a_pool, inception_3a_1x1], axis=3)

    # Inception3b
    inception_3b_3x3 = Conv2D(96, (1, 1), name='inception_3b_3x3_conv1')(inception_3a)
    # inception_3b_3x3 = tf.layers.batch_normalization(inception_3b_3x3, training=is_training, renorm=True, name='inception_3b_3x3_bn1', axis=3)    
    inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.001, name='inception_3b_3x3_bn1')(inception_3b_3x3)
    inception_3b_3x3 = Activation('relu')(inception_3b_3x3)
    inception_3b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)
    inception_3b_3x3 = Conv2D(128, (3, 3), name='inception_3b_3x3_conv2')(inception_3b_3x3)
    # inception_3b_3x3 = tf.layers.batch_normalization(inception_3b_3x3, training=is_training, renorm=True, name='inception_3b_3x3_bn2', axis=3)    
    inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.001, name='inception_3b_3x3_bn2')(inception_3b_3x3)
    inception_3b_3x3 = Activation('relu')(inception_3b_3x3)

    inception_3b_5x5 = Conv2D(32, (1, 1), name='inception_3b_5x5_conv1')(inception_3a)
    # inception_3b_5x5 = tf.layers.batch_normalization(inception_3b_5x5, training=is_training, renorm=True, name='inception_3b_5x5_bn1', axis=3)    
    inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.001, name='inception_3b_5x5_bn1')(inception_3b_5x5)
    inception_3b_5x5 = Activation('relu')(inception_3b_5x5)
    inception_3b_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)
    inception_3b_5x5 = Conv2D(64, (5, 5), name='inception_3b_5x5_conv2')(inception_3b_5x5)
    # inception_3b_5x5 = tf.layers.batch_normalization(inception_3b_5x5, training=is_training, renorm=True, name='inception_3b_5x5_bn2', axis=3)    
    inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.001, name='inception_3b_5x5_bn2')(inception_3b_5x5)
    inception_3b_5x5 = Activation('relu')(inception_3b_5x5)

    inception_3b_pool = Lambda(lambda x: x**2, name='power2_3b')(inception_3a)
    inception_3b_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3b_pool)
    inception_3b_pool = Lambda(lambda x: x*9, name='mult9_3b')(inception_3b_pool)
    inception_3b_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_3b')(inception_3b_pool)
    inception_3b_pool = Conv2D(64, (1, 1), name='inception_3b_pool_conv')(inception_3b_pool)
    # inception_3b_pool = tf.layers.batch_normalization(inception_3b_pool, training=is_training, renorm=True, name='inception_3b_pool_bn', axis=3)        
    inception_3b_pool = BatchNormalization(axis=3, epsilon=0.001, name='inception_3b_pool_bn')(inception_3b_pool)
    inception_3b_pool = Activation('relu')(inception_3b_pool)
    inception_3b_pool = ZeroPadding2D(padding=(4, 4))(inception_3b_pool)

    inception_3b_1x1 = Conv2D(64, (1, 1), name='inception_3b_1x1_conv')(inception_3a)
    inception_3b_1x1 = BatchNormalization(axis=3, epsilon=0.001, name='inception_3b_1x1_bn')(inception_3b_1x1)
    inception_3b_1x1 = Activation('relu')(inception_3b_1x1)

    inception_3b = concatenate([inception_3b_3x3, inception_3b_5x5, inception_3b_pool, inception_3b_1x1], axis=3)

    # Inception3c
    inception_3c_3x3 = utils.conv2d_bn(inception_3b,
                                    layer='inception_3c_3x3',
                                    cv1_out=128,
                                    cv1_filter=(1, 1),
                                    cv2_out=256,
                                    cv2_filter=(3, 3),
                                    cv2_strides=(2, 2),
                                    padding=(1, 1))

    inception_3c_5x5 = utils.conv2d_bn(inception_3b,
                                    layer='inception_3c_5x5',
                                    cv1_out=32,
                                    cv1_filter=(1, 1),
                                    cv2_out=64,
                                    cv2_filter=(5, 5),
                                    cv2_strides=(2, 2),
                                    padding=(2, 2))

    inception_3c_pool = MaxPooling2D(pool_size=3, strides=2)(inception_3b)
    inception_3c_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_3c_pool)

    inception_3c = concatenate([inception_3c_3x3, inception_3c_5x5, inception_3c_pool], axis=3)

    #inception 4a
    inception_4a_3x3 = utils.conv2d_bn(inception_3c,
                                    layer='inception_4a_3x3',
                                    cv1_out=96,
                                    cv1_filter=(1, 1),
                                    cv2_out=192,
                                    cv2_filter=(3, 3),
                                    cv2_strides=(1, 1),
                                    padding=(1, 1))
    inception_4a_5x5 = utils.conv2d_bn(inception_3c,
                                    layer='inception_4a_5x5',
                                    cv1_out=32,
                                    cv1_filter=(1, 1),
                                    cv2_out=64,
                                    cv2_filter=(5, 5),
                                    cv2_strides=(1, 1),
                                    padding=(2, 2))

    inception_4a_pool = Lambda(lambda x: x**2, name='power2_4a')(inception_3c)
    inception_4a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_4a_pool)
    inception_4a_pool = Lambda(lambda x: x*9, name='mult9_4a')(inception_4a_pool)
    inception_4a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_4a')(inception_4a_pool)
    inception_4a_pool = utils.conv2d_bn(inception_4a_pool,
                                    layer='inception_4a_pool',
                                    cv1_out=128,
                                    cv1_filter=(1, 1),
                                    padding=(2, 2))
    inception_4a_1x1 = utils.conv2d_bn(inception_3c,
                                    layer='inception_4a_1x1',
                                    cv1_out=256,
                                    cv1_filter=(1, 1))
    inception_4a = concatenate([inception_4a_3x3, inception_4a_5x5, inception_4a_pool, inception_4a_1x1], axis=3)

    #inception4e
    inception_4e_3x3 = utils.conv2d_bn(inception_4a,
                                    layer='inception_4e_3x3',
                                    cv1_out=160,
                                    cv1_filter=(1, 1),
                                    cv2_out=256,
                                    cv2_filter=(3, 3),
                                    cv2_strides=(2, 2),
                                    padding=(1, 1))
    inception_4e_5x5 = utils.conv2d_bn(inception_4a,
                                    layer='inception_4e_5x5',
                                    cv1_out=64,
                                    cv1_filter=(1, 1),
                                    cv2_out=128,
                                    cv2_filter=(5, 5),
                                    cv2_strides=(2, 2),
                                    padding=(2, 2))
    inception_4e_pool = MaxPooling2D(pool_size=3, strides=2)(inception_4a)
    inception_4e_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_4e_pool)

    inception_4e = concatenate([inception_4e_3x3, inception_4e_5x5, inception_4e_pool], axis=3)

    #inception5a
    inception_5a_3x3 = utils.conv2d_bn(inception_4e,
                                    layer='inception_5a_3x3',
                                    cv1_out=96,
                                    cv1_filter=(1, 1),
                                    cv2_out=384,
                                    cv2_filter=(3, 3),
                                    cv2_strides=(1, 1),
                                    padding=(1, 1))

    inception_5a_pool = Lambda(lambda x: x**2, name='power2_5a')(inception_4e)
    inception_5a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_5a_pool)
    inception_5a_pool = Lambda(lambda x: x*9, name='mult9_5a')(inception_5a_pool)
    inception_5a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_5a')(inception_5a_pool)
    inception_5a_pool = utils.conv2d_bn(inception_5a_pool,
                                    layer='inception_5a_pool',
                                    cv1_out=96,
                                    cv1_filter=(1, 1),
                                    padding=(1, 1))
    inception_5a_1x1 = utils.conv2d_bn(inception_4e,
                                    layer='inception_5a_1x1',
                                    cv1_out=256,
                                    cv1_filter=(1, 1))

    inception_5a = concatenate([inception_5a_3x3, inception_5a_pool, inception_5a_1x1], axis=3)

    #inception_5b
    inception_5b_3x3 = utils.conv2d_bn(inception_5a,
                                    layer='inception_5b_3x3',
                                    cv1_out=96,
                                    cv1_filter=(1, 1),
                                    cv2_out=384,
                                    cv2_filter=(3, 3),
                                    cv2_strides=(1, 1),
                                    padding=(1, 1))
    inception_5b_pool = MaxPooling2D(pool_size=3, strides=2)(inception_5a)
    inception_5b_pool = utils.conv2d_bn(inception_5b_pool,
                                    layer='inception_5b_pool',
                                    cv1_out=96,
                                    cv1_filter=(1, 1))
    inception_5b_pool = ZeroPadding2D(padding=(1, 1))(inception_5b_pool)

    inception_5b_1x1 = utils.conv2d_bn(inception_5a,
                                    layer='inception_5b_1x1',
                                    cv1_out=256,
                                    cv1_filter=(1, 1))
    inception_5b = concatenate([inception_5b_3x3, inception_5b_pool, inception_5b_1x1], axis=3)

    av_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(inception_5b)
    reshape_layer = Flatten()(av_pool)
    dense_layer = Dense(128, name='dense_layer')(reshape_layer)
    norm_layer = Lambda(lambda  x: K.l2_normalize(x, axis=1), name='norm_layer')(dense_layer)


    # Final Model
    model = Model(inputs=[myInput], outputs=norm_layer)
    return model

def get_model(target_model, img_size, num_classes):
    if target_model == 'triplet':
        model, base_model = Triplet().get_model(img_size, num_classes)
    else:
        raise Exception('Unknown model: {}'.format(target_model))
    return model, base_model

def euclidean_distance(vects):
    x, y = vects
    dist = K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
    return dist

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

class Triplet:
    def __init__(self):
        self.alpha = 1 #opt.alpha
        self.embd_dim = 128 #opt.embd_dim

    def triplet_loss(self, y_true, y_pred, embd_dim=128, alpha=0.2):
        anchor = y_pred[:,0:embd_dim]
        positive = y_pred[:,embd_dim:embd_dim*2]
        negative = y_pred[:,embd_dim*2:embd_dim*3]

        # distance between the anchor and the positive
        pos_dist = K.sum(K.square(anchor - positive), axis=1)

        # distance between the anchor and the negative
        neg_dist = K.sum(K.square(anchor - negative), axis=1)

        # compute loss
        basic_loss = pos_dist - neg_dist + alpha
        loss = K.maximum(basic_loss, 0.0)
        return K.mean(loss)

    def count_nonzero(self, y_true, y_pred):
        """
        Custom metric
        Returns count of nonzero embeddings
        """
        return(tf.count_nonzero(y_pred))

    def check_nonzero(self, y_true, y_pred):
        """
        Custom metric
        Returns sum of all embeddings
        """
        return(K.sum(y_pred))
    
    def cos_sim(self, y_true, y_pred):
        """
        Custom metric
        Returns cosine similarity
        """
        y_pred = y_pred[:,128:256] # pos
        y_true = y_pred[:,:128]    # anc
        x = K.l2_normalize(y_true, axis=-1)
        y = K.l2_normalize(y_pred, axis=-1)
        return K.mean(K.sum(x * y, axis=-1), axis=-1)
    
    def cos_sim_neg(self, y_true, y_pred):
        """
        Custom metric
        Returns cosine similarity
        """
        y_pred = y_pred[:,256:] # neg
        y_true = y_pred[:,:128]    # anc
        x = K.l2_normalize(y_true, axis=-1)
        y = K.l2_normalize(y_pred, axis=-1)
        return K.sum(x * y, axis=-1)

    def pos_dist(self, y_true, y_pred):
        y_true = y_pred[:,128:256] # pos
        y_pred = y_pred[:,:128]    # anc
        return K.mean(K.sum(K.square(y_pred - y_true), axis=-1))

    def neg_dist(self, y_true, y_pred):
        y_true = y_pred[:,256:] # neg
        y_pred = y_pred[:,:128]    # anc
        return K.mean(K.sum(K.square(y_pred - y_true), axis=-1))

    def get_model(self, img_size, num_classes):
        shape = (img_size, img_size, 3)
        a = Input(shape=shape)
        p = Input(shape=shape)
        n = Input(shape=shape)

        base_model = get_base_model()
        base_model.load_weights('./checkpoint/nn4.small2.lrn_py3_noBN.h5')

        net = base_model.output
        net = Dense(self.embd_dim, activation='linear')(net)
        net = Lambda(lambda x: K.l2_normalize(x, axis=-1))(net)
        base_model = Model(base_model.input, net, name='facenet')

        a_emb = base_model(a)
        p_emb = base_model(p)
        n_emb = base_model(n)

        merge_vec = concatenate([a_emb, p_emb, n_emb], axis=-1)

        model = Model(inputs=[a, p, n], outputs=merge_vec)
        optm = keras.optimizers.Adam(1e-5, decay=1e-6)
        model.compile(loss=#stacked_dists,
                        wrapped_partial(self.triplet_loss,
                                    embd_dim=self.embd_dim,
                                    alpha=self.alpha),
                      optimizer=optm,
                      metrics=[self.pos_dist, self.neg_dist, self.cos_sim_neg])
        # model.summary()
        return model, base_model


if __name__ == '__main__':
    model = get_base_model()
    model.load_weights('./checkpoint/nn4.small2.lrn_py3.h5')
    
    for i, layer in enumerate(model.layers):
        layer_name = layer.get_config()['name']
        if 'bn' in layer_name:
            print ('[%d]'%i, layer_name)
            bn_gamma, bn_beta, bn_mean, bn_var = model.get_layer(layer_name).get_weights()
            print('gamma:', bn_gamma.shape, bn_gamma.mean(), '\n'+
                  'beta: ', bn_beta.shape, bn_beta.mean(), '\n'+
                  'mean: ', bn_mean.shape, bn_mean.mean(), '\n'+
                  'var:  ', bn_var.shape, bn_var.mean())

    bn_gamma, bn_beta, bn_mean, bn_var = model.layers[2].get_weights()
    print (bn_gamma, bn_beta, bn_mean, bn_var)