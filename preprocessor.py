import numpy as np
import random
import os
from glob import glob
from os.path import join
import cv2
import keras
from sklearn.externals.joblib import Parallel, delayed
# from augmentator import ImageAugment

class TripletGenerator(keras.utils.Sequence):
    def __init__(self, dim=(96,96), n_channels=3, n_classes=1, flg_shuffle=True, seed=0, n=0, n_workers=1, flg_caching_all=True):
        self.file_list = None
        self.batch_size = 8
        self.batch_idx = 0
        self.epoch_idx = 0
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.flg_shuffle = flg_shuffle
        self.dim = dim
        self.n = n
        self.seed = seed
        self.R = random.Random(self.seed)
        self.indexes = None
        self.n_workers = n_workers
        self.dict_imgs_per_id = {}
        self.id_list = []
        self.flg_caching_all = flg_caching_all
        self.get_data_once = self.random_triplet_selection
        self.cache_size = 0
        self.time = 0

        aug_params = dict(rotation_range = 10.0,
                  width_shift_range = 0.1,
                  height_shift_range = 0.1,
                  zoom_range = 0.1,
                  brightness_range = [0.6, 1.1],
                  gamma_adjustment = True
                 )
        # self.ImgAug = ImageAugment(**aug_params)

        super(TripletGenerator, self).__init__()
    
    def __len__(self):
        self.n = len(self.id_list)
        return self.n

    def __file_list(self, path):
        id_path_list = sorted(glob(join(path, '*')))        
        self.file_list = sorted(glob(join(path, "*/*.jpg")))
        self.dict_imgs_per_id = {}
        self.id_list = []

        for id_path in id_path_list:
            id = os.path.basename(id_path)
            jpg_list = sorted(glob(join(id_path, '*.jpg'))) 
            if len(jpg_list) > 1:
                self.dict_imgs_per_id[id] = jpg_list
                self.id_list.append(id)
        
        self.n = len(self.id_list) * 3 # magic number
        self.indexes = list(range(self.n))

    def on_epoch_begin(self):
        print ('[*] -----  epoch start  -----')
        print ('[*] triplet data on load time: %.3f s'%self.time)
        print ('[*] triplet data size: %d x %d'%(self.cache_size, 3))
        print ('[*] num of batches per epoch: %d'%(self.n//self.batch_size))
        print ('[*] -------------------------')

    def on_epoch_end(self):
        'Update indexes after each epoch'
        if self.flg_shuffle == True:
            self.R.shuffle(self.indexes)

    def load_all_data(self, cache_size, *args):
        # print (cache_size, len(args[0].keys()), len(args[1]), args[2])
        np_ancs = np.zeros((cache_size, self.dim[0], self.dim[1], 3), dtype=np.uint8)
        np_poss = np.zeros((cache_size, self.dim[0], self.dim[1], 3), dtype=np.uint8)
        np_negs = np.zeros((cache_size, self.dim[0], self.dim[1], 3), dtype=np.uint8)

        data = Parallel(n_jobs=self.n_workers, verbose=0, backend='threading')(
            delayed(self.get_data_once)(*args) for _ in range(cache_size)
        )
        
        for i, (np_anc, np_pos, np_neg) in enumerate(data):
                np_ancs[i] = np_anc
                np_poss[i] = np_pos
                np_negs[i] = np_neg
        
        return np_ancs, np_poss, np_negs

    def random_triplet_selection(self, dict_imgs, list_ids, img_size):
        anchor_id, neg_id = random.sample(list_ids, 2)
        anchor, positive = random.sample(dict_imgs[anchor_id], 2)
        [negative] = random.sample(dict_imgs[neg_id], 1)

        img_anc = cv2.resize(cv2.imread(anchor, 1), img_size)
        img_pos = cv2.resize(cv2.imread(positive, 1), img_size)
        img_neg = cv2.resize(cv2.imread(negative, 1), img_size)

        return img_anc, img_pos, img_neg

    def __getitem__(self, x=None):
        while True:
            img_size = self.dim
            list_param = [self.dict_imgs_per_id, self.id_list, img_size]
            self.cache_size = self.n
            start = cv2.getTickCount()
            cached_data = self.load_all_data(self.cache_size, *list_param)
            self.time = (cv2.getTickCount() - start) / cv2.getTickFrequency()
            
            ### batching data
            for self.batch_idx in range(self.n//self.batch_size):
                # print (self.batch_idx)
                if self.batch_size * (self.batch_idx + 1) >= self.n:
                    list_batch = [np_data[self.batch_size*self.batch_idx:] for np_data in cached_data]
                else:
                    list_batch = [np_data[self.batch_size*self.batch_idx:self.batch_size*(self.batch_idx+1)] for np_data in cached_data]
                
                np_anc_batch = list_batch[0].astype(np.float32) 
                np_pos_batch = list_batch[1].astype(np.float32) 
                np_neg_batch = list_batch[2].astype(np.float32) 

                ### with augmentation
                # aug_anc_batch = self.ImgAug.augment(np_anc_batch)
                # aug_pos_batch = self.ImgAug.augment(np_pos_batch)
                # aug_neg_batch = self.ImgAug.augment(np_neg_batch)

                X = [np_anc_batch / 255] + [np_pos_batch / 255] + [np_neg_batch / 255]

                ### with augmentation
                # X = [aug_anc_batch / 255] + [aug_pos_batch / 255] + [aug_neg_batch / 255]
                Y = np.zeros((len(list_batch[0]), 128*3))
                yield X, Y

    def flow_from_directory(self, path, batch_size=8, seed=0):
        ### read file path list
        self.__file_list(path)
        self.batch_size = batch_size
        self.R = random.Random(seed)
        
        self.indexes = list(range(self.n))
        print ('[*] found %d images, %d classes'%(len(self.file_list), self.n//3))
        return self.__getitem__()
        
                    
if __name__ == '__main__':
    triplet_datagen = TripletGenerator(dim=(224,224), n_workers=12, flg_caching_all=True)
    batch_size = 64
    triplet_gen = triplet_datagen.flow_from_directory('../../opendata/inshop/Img/train', batch_size=batch_size)

    cnt=0
    while True:
        triplet_set,Y = next(triplet_gen)
        # print (triplet_datagen.batch_idx)
        anc = triplet_set[0]
        pos = triplet_set[1]
        neg = triplet_set[2]
        print (cnt, triplet_datagen.batch_idx)
        cnt+=1
        # for idx in range(batch_size):
        #     img_anc = anc[idx]#.astype(np.uint8)
        #     img_pos = pos[idx]#.astype(np.uint8)
        #     img_neg = neg[idx]#.astype(np.uint8)
            # print ('----batch %d----'%idx)
            # print (img_anc.shape, img_anc.mean(), img_anc.dtype)
            # print (img_pos.shape, img_pos.mean(), img_pos.dtype)
            # print (img_neg.shape, img_neg.mean(), img_neg.dtype)
            # cv2.imshow('anc', img_anc)
            # cv2.imshow('pos', img_pos)
            # cv2.imshow('neg', img_neg)
            # key = cv2.waitKey(0)
            # if key == 27:
            #     exit()

    
