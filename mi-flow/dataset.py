from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
# if sys.version_info[0] == 2:
#     import cPickle as pickle
# else:
#     import pickle


def get_img(img_dir, seq, imsize, bbox=None, transform=None):
    imgs = []
    for i in seq:
        img_path = img_dir + i +".jpg"
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)
            img = img.crop([x1, y1, x2, y2])

        if transform is not None:
            img = transform(img)
        imgs.append(img)
    imgs = torch.stack(imgs, 0)

    return imgs


class FaceDataset(data.Dataset):
    def __init__(self, data_dir, base_size=64,
                 transform=None, target_transform=None):
        self.transform = transform
        self.imsize = base_size
        self.word, self.filenames, self.img_names = self.load_filenames(data_dir)

    def load_filenames(self, data_dir):
        self.dir = data_dir
        word = os.listdir(data_dir)

        filenames = []
        img_names = {}
        self.word_idx = {}
        n = 0
        for w in word:
            self.word_idx[w] = n
            n += 1
            img_dir = os.listdir(data_dir+w)
            img_dir.sort()
            wlen = len(w)
            for i in img_dir:
                ID = w + "/" + i[:wlen+7]
                seq = i[wlen+7:wlen+9]
                try:
                    img_names[ID].append(seq)
                except:
                    filenames.append(ID)
                    img_names[ID] = [w, seq]
        print("load %s ID file from %s" %(len(filenames), data_dir))


        return word, filenames, img_names

    def __getitem__(self, index):
        #
        file = self.filenames[index]
        ID = file[-6:-1]
        word = self.img_names[file][0]
        seq = self.img_names[file][1:]
        seq_len = len(seq)

        imgs = get_img(self.dir+file, seq, self.imsize, 
                        transform=self.transform)

        return imgs, word, ID, seq


    def __len__(self):
        return len(self.filenames)
