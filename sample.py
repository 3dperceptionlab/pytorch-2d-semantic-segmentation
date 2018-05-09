import logging
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils

import loader.utils

log = logging.getLogger(__name__)

def sample(args):

    batch_ = 0
    batch_size_ = 4

    loader_ = loader.utils.get_loader(args.dataset)
    path_ = loader.utils.get_path(args.dataset)

    data_loader_ = loader_(path_, args.split, args.img_width, args.img_height, isTransform=True)
    log.info(data_loader_)
    torch_data_loader_ = torch.utils.data.DataLoader(data_loader_, batch_size=batch_size_, num_workers=1, shuffle=False)

    for i, data in enumerate(torch_data_loader_):

        imgs_, lbls_, lbls_rgb_ = data

        imgs_ = imgs_.numpy()[:,::-1,:,:]
        imgs_ = np.transpose(imgs_, [0, 2, 3, 1])
        f_, ax_ = plt.subplots(batch_size_, 3)

        for j in range(batch_size_):
            ax_[j][0].imshow(imgs_[j])
            ax_[j][1].imshow(data_loader_.decode_labels(lbls_.numpy()[j]))
            ax_[j][2].imshow(lbls_rgb_[j])

        plt.show()

if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser_ = argparse.ArgumentParser(description='Parameters')

    parser_.add_argument('--dataset', nargs='?', type=str, default='cityscapes',
                            help='The dataset to be used for training.')
    parser_.add_argument('--split', nargs='?', type=str, default='train',
                            help='The split to stat')
    parser_.add_argument('--img_width', nargs='?', type=int, default=256,
                            help='Image width')
    parser_.add_argument('--img_height', nargs='?', type=int, default=256,
                            help='Image height')

    args_ = parser_.parse_args()

    sample(args_)
