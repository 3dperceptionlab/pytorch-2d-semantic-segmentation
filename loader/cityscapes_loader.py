import collections
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import torch
import torch.utils.data
import torchvision

import loader.utils

class CityscapesLoader(torch.utils.data.Dataset):

    def __init__(self, root, split, imgWidth, imgHeight, imgNorm=True, isTransform=False):

        self.root = root
        self.split = split
        self.img_size = (imgWidth, imgHeight)
        self.img_norm = imgNorm
        self.is_transform = isTransform

        self.num_classes = 19
        self.void_classes = [0, 1, 2, 3, 4, 6, 7, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))
        self.colors = [[128, 64, 128],
                        [244, 35, 232],
                        [70, 70, 70],
                        [102, 102, 156],
                        [190, 153, 153],
                        [153, 153, 153],
                        [250, 170, 30],
                        [220, 220, 0],
                        [107, 142, 35],
                        [152, 251, 152],
                        [0, 130, 180],
                        [220, 20, 60],
                        [255, 0, 0],
                        [0, 0, 142],
                        [0, 0, 70],
                        [0, 60, 100],
                        [0, 80, 100],
                        [0, 0, 230],
                        [119, 11, 32]]
        self.label_colors = dict(zip(range(19), self.colors))

        self.images_base_path = os.path.join(self.root, 'leftImg8bit', self.split)
        self.labels_base_path = os.path.join(self.root, 'gtFine', self.split)
        self.files = loader.utils.recursive_glob(root=self.images_base_path, suffix='.png')

        if not self.files:
            raise Exception("No files for the requested split {0} were found in {1}".format(
                                self.split,
                                self.images_base_path))

        print("Found {0} images in split {1} on {2}".format(
                len(self.files),
                self.split,
                self.images_base_path))

    def __len__(self):

        return len(self.files)

    def __getitem__(self, index):

        img_path_ = self.files[index].rstrip()
        lbl_path_ = os.path.join(self.labels_base_path,
                                    img_path_.split(os.sep)[-2],
                                    os.path.basename(img_path_)[:-15] + 'gtFine_labelIds.png')

        print("Loading image {0} with labels {1}".format(
                img_path_,
                lbl_path_))

        img_ = scipy.misc.imread(img_path_)
        img_ = np.array(img_, dtype=np.uint8)

        lbl_ = scipy.misc.imread(lbl_path_)
        lbl_ = self.encode_labels(np.array(lbl_, dtype=np.uint8))

        if self.is_transform:
            img_, lbl_ = self.transform(img_, lbl_)

        return img_, lbl_
    
    def __repr__(self):

        return "Dataset loader for Cityscapes {0} split with {1} images in {2}...".format(
                    self.split,
                    self.__len__(),
                    self.images_base_path)

    def transform(self, img, lbl):

        img_ = scipy.misc.imresize(img, self.img_size)
        img_ = img_[:,:,::-1]
        img_ = img_.astype(np.float64)
        if self.img_norm:
            img_ = img_.astype(float) / 255.0
        img_ = img_.transpose(2, 0, 1)

        lbl_ = lbl.astype(float)
        lbl_ = scipy.misc.imresize(lbl_, self.img_size, 'nearest', mode='F')

        img_ = torch.from_numpy(img_).float()
        lbl_ = torch.from_numpy(lbl_).long()

        return img_, lbl_

    def encode_labels(self, labels):

        lbls_ = np.copy(labels)

        for vc in self.void_classes:
            lbls_[labels == vc] = self.ignore_index
        for vc in self.valid_classes:
            lbls_[labels == vc] = self.class_map[vc]

        return lbls_

    def decode_labels(self, labels):

        r_ = labels.copy()
        g_ = labels.copy()
        b_ = labels.copy()

        for l in range(0, self.num_classes):

            r_[labels == l] = self.label_colors[l][0]
            g_[labels == l] = self.label_colors[l][1]
            b_[labels == l] = self.label_colors[l][2]

        rgb_ = np.zeros((labels.shape[0], labels.shape[1], 3))
        rgb_[:,:,0] = r_ / 255.0
        rgb_[:,:,1] = g_ / 255.0
        rgb_[:,:,2] = b_ / 255.0
        
        return rgb_

if __name__ == '__main__':

    batch_ = 0
    batch_size_ = 4

    path_ = '../datasets/cityscapes/'
    dataset_loader_ = CityscapesLoader(path_, 'train', 512, 512, isTransform=True)
    train_loader_ = torch.utils.data.DataLoader(dataset_loader_, batch_size=batch_size_)

    for i, data in enumerate(train_loader_):

        imgs_, lbls_ = data

        imgs_ = imgs_.numpy()[:,::-1,:,:]
        imgs_ = np.transpose(imgs_, [0, 2, 3, 1])
        f_, ax_ = plt.subplots(batch_size_, 2)

        for j in range(batch_size_):
            ax_[j][0].imshow(imgs_[j])
            ax_[j][1].imshow(dataset_loader_.decode_labels(lbls_.numpy()[j]))

        plt.show()
