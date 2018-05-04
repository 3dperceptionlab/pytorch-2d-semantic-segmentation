import argparse
import logging
import sys

import tqdm
import torch.autograd
import torch.nn
import torch.utils

import loader.utils
import network.utils
import loss.utils
import metrics.average_metrics

log = logging.getLogger(__name__)

def train(args):

    # Data loader
    data_loader_ = loader.utils.get_loader(args.dataset)
    data_path_ = loader.utils.get_path(args.dataset)

    train_loader_ = data_loader_(data_path_, 'train', 512, 512, isTransform=True)
    log.info(train_loader_)
    test_loader_ = data_loader_(data_path_, 'test', 512, 512, isTransform=True)
    log.info(test_loader_)
    val_loader_ = data_loader_(data_path_, 'val', 512, 512, isTransform=True)
    log.info(val_loader_)

    train_data_loader_ = torch.utils.data.DataLoader(train_loader_, batch_size=args.batch_size, num_workers=8, shuffle=True)
    test_data_loader_ = torch.utils.data.DataLoader(test_loader_, batch_size=args.batch_size, num_workers=8)

    num_classes_ = train_loader_.num_classes

    # Network loader
    network_ = network.utils.get_network(args.network, num_classes_)
    log.info(network_)
    network_ = torch.nn.DataParallel(network_, device_ids=range(torch.cuda.device_count()))
    network_.cuda()

    # Set up optimizer
    optimizer_ = torch.optim.SGD(network_.parameters(),
                                    lr=args.learning_rate,
                                    momentum=0.99,
                                    weight_decay=5e-4)

    # Set up loss
    loss_function_ = loss.utils.get_loss(args.loss)
    criterion_ = loss_function_(sizeAverage=False, ignoreIndex=train_loader_.ignore_index)
    criterion_.cuda()
    log.info(criterion_)

    # Set up metrics
    average_metrics_ = metrics.average_metrics.AverageMetrics(num_classes_)

    # Training loop
    for epoch in range(args.epochs):

        log.info('*** Epoch {0} ***'.format(epoch))

        network_.train()

        log.info('Training...')

        for i, (imgs, lbls) in tqdm.tqdm(enumerate(train_data_loader_), total=len(train_loader_) / args.batch_size):

            N_ = imgs.size(0)

            imgs_ = torch.autograd.Variable(imgs.cuda())
            lbls_ = torch.autograd.Variable(lbls.cuda())

            optimizer_.zero_grad()
            outputs_ = network_(imgs_)

            loss_ = criterion_(outputs_, lbls_) / N_

            loss_.backward()
            optimizer_.step()

        network_.eval()

        log.info('Evaluating on training set...')

        for i, (imgs, lbls) in tqdm.tqdm(enumerate(train_data_loader_), total=len(train_loader_) / args.batch_size):

            imgs_ = torch.autograd.Variable(imgs_.cuda(), volatile=True)
            lbls_ = torch.autograd.Variable(lbls_.cuda(), volatile=True)

            ouputs_ = network_(imgs_)
            preds_ = outputs_.data.max(1)[1].cpu().numpy()
            gt_ = lbls_.data.cpu().numpy()

            average_metrics_.update(preds_, gt_)

        scores_ = average_metrics_.evaluate()

        for score, value in scores_.items():
            log.info(score, value)

        average_metrics_.reset()

        log.info('Evaluating on testing set...')

        for i, (imgs, lbls) in tqdm.tqdm(enumerate(test_data_loader_), total=len(test_loader_) / args.batch_size):

            imgs_ = torch.autograd.Variable(imgs_.cuda(), volatile=True)
            lbls_ = torch.autograd.Variable(lbls_.cuda(), volatile=True)

            ouputs_ = network_(imgs_)
            preds_ = outputs_.data.max(1)[1].cpu().numpy()
            gt_ = lbls_.data.cpu().numpy()

            average_metrics_.update(preds_, gt_)

        scores_ = average_metrics_.evaluate()

        for score, value in scores_.items():
            log.info(score, value)

        average_metrics_.reset()


if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser_ = argparse.ArgumentParser(description='Parameters')
    parser_.add_argument('--dataset', nargs='?', type=str, default='cityscapes',
                            help='The dataset to be used for training.')
    parser_.add_argument('--network', nargs='?', type=str, default='unet',
                            help='The network architecture to be used for training.')
    parser_.add_argument('--learning_rate', nargs='?', type=float, default=1e-5,
                            help='Starting learning rate for the optimizer')
    parser_.add_argument('--momentum', nargs='?', type=float, default=0.99,
                            help='Momentum for the optimizer')
    parser_.add_argument('--weight_decay', nargs='?', type=float, default=5e-4,
                            help='Weight decay for the optimizer')
    parser_.add_argument('--loss', nargs='?', type=str, default='crossentropy2d',
                            help='Loss function')
    parser_.add_argument('--epochs', nargs='?', type=int, default=10,
                            help='Number of training epochs')
    parser_.add_argument('--batch_size', nargs='?', type=int, default=4,
                            help='Batch size for training')
    args_ = parser_.parse_args()
    train(args_)
