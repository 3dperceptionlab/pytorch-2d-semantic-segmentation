import argparse
import datetime
import logging
import os
import sys

import numpy as np
import scipy
import tqdm
import torch.autograd
import torch.nn
import torch.utils
import torchvision

import loader.utils
import network.utils
import loss.utils

log = logging.getLogger(__name__)

def infer(args):

    # Data loader
    data_loader_ = loader.utils.get_loader(args.dataset)
    data_path_ = loader.utils.get_path(args.dataset)
    loader_ = data_loader_(data_path_, args.split, args.img_width, args.img_height, isTransform=True)
    log.info(loader_)

    num_classes_ = loader_.num_classes

    # Load image
    img_ = None
    if os.path.isfile(args.img_path):

        log.info('Reading input image from {0}...'.format(args.img_path))
        img_ = scipy.misc.imread(args.img_path)
        img_ = scipy.misc.imresize(img_, (args.img_width, args.img_height), interp='bicubic')
        img_ = img_[:, :, ::-1]
        img_ = img_.astype(float) / 255.0
        img_ = img_.transpose(2, 0, 1)
        img_ = np.expand_dims(img_, 0)
        img_ = torch.from_numpy(img_).float()

    else:

        log.info('Image not found at {0}...'.format(args.img_path))
        raise FileNotFoundError

    # Network loader
    network_ = network.utils.get_network(args.network, num_classes_)
    log.info(network_)
    network_ = torch.nn.DataParallel(network_, device_ids=range(torch.cuda.device_count()))
    network_.cuda()

    # Load checkpoint if specified
    if args.checkpoint is not None:

        if os.path.isfile(args.checkpoint):

            log.info('Loading checkpoint {}'.format(args.checkpoint))
            checkpoint_ = torch.load(args.checkpoint)
            network_.load_state_dict(checkpoint_['model_state'])
            log.info('Loaded network...')

        else:
            log.info('The checkpoint file at {} was not found'.format(args.checkpoint))

    # Infer image
    log.info('Inference...')

    network_.eval()
    with torch.no_grad():

        img_ = torch.autograd.Variable(img_.cuda())
        output_ = network_(img_)
        preds_ = np.squeeze(output_.data.max(1)[1].cpu().numpy(), axis=0)
        decoded_ = loader_.decode_labels(preds_)

        scipy.misc.imsave(args.output_path, decoded_)
        log.info('Output saved to {0}'.format(args.output_path))


if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser_ = argparse.ArgumentParser(description='Parameters')
    parser_.add_argument('--dataset', nargs='?', type=str, default='cityscapes',
                            help='The dataset to be used for training.')
    parser_.add_argument('--split', nargs='?', type=str, default='val',
                            help='The evaluation split name')
    parser_.add_argument('--network', nargs='?', type=str, default='unet',
                            help='The network architecture to be used for training.')
    parser_.add_argument('--img_width', nargs='?', type=int, default=256,
                            help='Image width')
    parser_.add_argument('--img_height', nargs='?', type=int, default=256,
                            help='Image height')
    parser_.add_argument('--checkpoint', nargs='?', type=str, default=None,
                            help='Checkpoint file to resume training')
    parser_.add_argument('--img_path', nargs='?', type=str, default='in.png',
                            help='Path to the image to be inferred')
    parser_.add_argument('--output_path', nargs='?', type=str, default='out.png',
                            help='Path to the output image')
    args_ = parser_.parse_args()
    infer(args_)
