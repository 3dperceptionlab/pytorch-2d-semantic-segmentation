import argparse
import datetime
import logging
import os
import sys

import tqdm
import torch.autograd
import torch.nn
import torch.utils
import torchvision

import loader.utils
import network.utils
import loss.utils
import metrics.average_metrics

log = logging.getLogger(__name__)

def evaluate(args):

    # Data loader
    data_loader_ = loader.utils.get_loader(args.dataset)
    data_path_ = loader.utils.get_path(args.dataset)
    evaluation_loader_ = data_loader_(data_path_, args.split, args.img_width, args.img_height, isTransform=True)
    log.info(evaluation_loader_)
    evaluation_data_loader_ = torch.utils.data.DataLoader(evaluation_loader_, batch_size=args.batch_size, num_workers=8)

    num_classes_ = evaluation_loader_.num_classes

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

    average_metrics_ = metrics.average_metrics.AverageMetrics(num_classes_)
    
    network_.eval()

    for i, (imgs, lbls, lbls_rgb) in tqdm.tqdm(enumerate(evaluation_data_loader_), total=len(evaluation_loader_) / args.batch_size):

        imgs_ = torch.autograd.Variable(imgs.cuda(), volatile=True)
        lbls_ = torch.autograd.Variable(lbls.cuda(), volatile=True)

        outputs_ = network_(imgs_)
        preds_ = outputs_.data.max(1)[1].cpu().numpy()
        gt_ = lbls_.data.cpu().numpy()

        average_metrics_.update(preds_, gt_)

    scores_ = average_metrics_.evaluate()

    for score, value in scores_.items():
        score_str_ = "{0}: {1}".format(score, value)
        log.info(score_str_)

    average_metrics_.reset()

if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser_ = argparse.ArgumentParser(description='Parameters')
    parser_.add_argument('--dataset', nargs='?', type=str, default='cityscapes',
                            help='The dataset to be used for training.')
    parser_.add_argument('--split', nargs='?', type=str, default='val',
                            help='The evaluation split name')
    parser_.add_argument('--network', nargs='?', type=str, default='unet',
                            help='The network architecture to be used for training.')
    parser_.add_argument('--loss', nargs='?', type=str, default='crossentropy2d',
                            help='Loss function')
    parser_.add_argument('--batch_size', nargs='?', type=int, default=32,
                            help='Batch size for training')
    parser_.add_argument('--img_width', nargs='?', type=int, default=256,
                            help='Image width')
    parser_.add_argument('--img_height', nargs='?', type=int, default=256,
                            help='Image height')
    parser_.add_argument('--checkpoint', nargs='?', type=str, default=None,
                            help='Checkpoint file to resume training')
    args_ = parser_.parse_args()
    evaluate(args_)
