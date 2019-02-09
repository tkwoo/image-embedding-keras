# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import join
import cv2
import argparse
import pickle
import math

import numpy as np

import keras
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
from keras import backend as K
from network import get_model
from preprocessor import TripletGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # or any {'0', '1', '2', '3'}
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def l2_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def train(config):
    # training parameters
    batch_size = config.batch_size
    num_classes = 1000
    input_shape = (96, 96, 3)  # input image shape

    """ Model """
    model, base_model = get_model('triplet', 96, num_classes)
    if config.pretrained_weight_path is not None:
        model.load_weights(config.pretrained_weight_path)

    """ Load data """
    DATASET_PATH = './data'
    train_dataset_path = os.path.join(DATASET_PATH, 'train')

    triplet_datagen = TripletGenerator(dim=(96,96), n_workers=12, flg_caching_all=True)
    triplet_gen = triplet_datagen.flow_from_directory(train_dataset_path, batch_size=batch_size)

    def lr_step_decay(epoch):
        init_lr = config.initial_learning_rate
        lr_decay = config.learning_rate_decay_factor
        epoch_per_decay = config.epoch_per_decay
        lrate = init_lr * math.pow(lr_decay, math.floor((1+epoch)/epoch_per_decay))
        # print lrate
        return lrate

    """ Callback """
    # usr_callback = callbacks.trainCheck(config, base_model)
    learning_rate = LearningRateScheduler(lr_step_decay)
    if not os.path.exists(join(config.ckpt_dir, config.ckpt_name)):
        os.makedirs(join(config.ckpt_dir, config.ckpt_name), exist_ok=True)
    model_checkpoint = ModelCheckpoint(
                os.path.join(join(config.ckpt_dir, config.ckpt_name, 'weights.{epoch:04d}.h5')), 
                period=5,
                save_weights_only=True)
    callbacks = [learning_rate, model_checkpoint] 

    """ Training loop """
    res = model.fit_generator(generator=triplet_gen, #train_gen, #triplet_gen,
                                steps_per_epoch=triplet_datagen.n//batch_size, #steps_per_epoch, #triplet_datagen.n//batch_size,
                                epochs=config.epochs,
                            #   validation_data=dev_gen if dev_size > 0 else None,
                            #   validation_steps=validation_steps if dev_size > 0 else None,
                                shuffle=True,
                            #   initial_epoch=epoch,
                                callbacks=callbacks)

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument("--initial_learning_rate", help="init lr", default=1e-5, type=float)
    args.add_argument("--learning_rate_decay_factor", help="learning rate decay", default=0.5, type=float)
    args.add_argument("--epoch_per_decay", help="lr decay period", default=20, type=int)
    args.add_argument("--pretrained_weight_path", help="weight.h5 path", default=None)
    args.add_argument("--ckpt_dir", help="checkpoint root directory", default='./checkpoint')
    args.add_argument("--ckpt_name", help="[.../ckpt_dir/ckpt_name/weights.h5]", default='ir')

    config = args.parse_args()

    train(config)