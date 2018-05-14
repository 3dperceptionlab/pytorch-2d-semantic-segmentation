import argparse
import datetime
import logging
import os
import sys

import tensorboardX
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

def evaluate(network, loader, numImages, batchSize, numClasses):

    average_metrics_ = metrics.average_metrics.AverageMetrics(numClasses)
    
    network.eval()

    for i, (imgs, lbls, lbls_rgb) in tqdm.tqdm(enumerate(loader), total=numImages / batchSize):

        imgs_ = torch.autograd.Variable(imgs.cuda(), volatile=True)
        lbls_ = torch.autograd.Variable(lbls.cuda(), volatile=True)

        outputs_ = network(imgs_)
        preds_ = outputs_.data.max(1)[1].cpu().numpy()
        gt_ = lbls_.data.cpu().numpy()

        average_metrics_.update(preds_, gt_)

    scores_ = average_metrics_.evaluate()

    for score, value in scores_.items():
        score_str_ = "{0}: {1}".format(score, value)
        log.info(score_str_)

    average_metrics_.reset()


def train(args):

    experiment_str_ = '{0}-{1}-{2}'.format(
                        args.network,
                        args.dataset,
                        datetime.datetime.now().strftime('%b%d_%H-%M-%S'))

    writer_ = tensorboardX.SummaryWriter('experiments/logs/' + experiment_str_)

    # Data loader
    data_loader_ = loader.utils.get_loader(args.dataset)
    data_path_ = loader.utils.get_path(args.dataset)

    train_loader_ = data_loader_(data_path_, 'train', args.img_width, args.img_height, isTransform=True)
    log.info(train_loader_)
    test_loader_ = data_loader_(data_path_, 'val', args.img_width, args.img_height, isTransform=True)
    log.info(test_loader_)

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

    # Load checkpoint if specified
    if args.checkpoint is not None:

        if os.path.isfile(args.checkpoint):

            log.info('Loading checkpoint {}'.format(args.checkpoint))
            checkpoint_ = torch.load(args.checkpoint)
            network_.load_state_dict(checkpoint_['model_state'])
            log.info('Loaded network...')

            if args.finetune:
                log.info('Finetuning enabled, optimizer not loaded...')
            else:
                optimizer_.load_state_dict(checkpoint_['optimizer_state'])
                log.info('Loaded optimizer...')
        
        else:
            log.info('The checkpoint file at {} was not found'.format(args.checkpoint))


    # Training loop
    for epoch in range(args.epochs):

        log.info('*** Epoch {0} ***'.format(epoch))

        network_.train()

        log.info('Training...')

        for i, (imgs, lbls, lbls_rgb) in tqdm.tqdm(enumerate(train_data_loader_), total=len(train_loader_) / args.batch_size):

            N_ = imgs.size(0)

            imgs_ = torch.autograd.Variable(imgs.cuda())
            lbls_ = torch.autograd.Variable(lbls.cuda())

            optimizer_.zero_grad()
            outputs_ = network_(imgs_)

            loss_ = criterion_(outputs_, lbls_) / N_

            # Visualization
            n_iter_ = epoch * len(train_loader_) / args.batch_size + i + 1
            writer_.add_scalar('Loss', loss_.item(), n_iter_)

            lr_i_ = 0
            for g in optimizer_.param_groups:
                writer_.add_scalar('Learning_rate{0}'.format(lr_i_), g['lr'], n_iter_)
                lr_i_ += 1
 
            vis_imgs_ = torchvision.utils.make_grid(imgs_, normalize=False, scale_each=False)
            writer_.add_image('Input', vis_imgs_, n_iter_)

            lbls_rgb_ = lbls_rgb.permute(0, 3, 1, 2)
            vis_lbls_ = torchvision.utils.make_grid(lbls_rgb_, normalize=False, scale_each=False)
            writer_.add_image('Labels', vis_lbls_, n_iter_)

            outputs_lbls_ = train_loader_.decode_labels_batch(outputs_.data.max(1)[1].cpu().numpy())
            outputs_lbls_ = outputs_lbls_.permute(0, 3, 1, 2)
            vis_outputs_lbls_ = torchvision.utils.make_grid(outputs_lbls_, normalize=False, scale_each=False)
            writer_.add_image('Outputs', vis_outputs_lbls_, n_iter_)

            loss_.backward()
            optimizer_.step()

        if epoch != 0 and epoch % args.evaluate == 0:

            log.info('=== Evaluating on training set ===')
            evaluate(network_, train_data_loader_, len(train_loader_), args.batch_size, num_classes_)

            log.info('=== Evaluating on testing set ===')
            evaluate(network_, test_data_loader_, len(test_loader_), args.batch_size, num_classes_)

            log.info('=== Checkpoint ===')
            state_ = {'epoch': epoch+1,
                      'model_state': network_.state_dict(),
                      'optimizer_state': optimizer_.state_dict(),}
            torch.save(state_, ('experiments/checkpoints/' + experiment_str_ + '_{0}.pkl').format(epoch))

if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser_ = argparse.ArgumentParser(description='Parameters')
    parser_.add_argument('--dataset', nargs='?', type=str, default='cityscapes',
                            help='The dataset to be used for training.')
    parser_.add_argument('--network', nargs='?', type=str, default='unet',
                            help='The network architecture to be used for training.')
    parser_.add_argument('--learning_rate', nargs='?', type=float, default=1e-3,
                            help='Starting learning rate for the optimizer')
    parser_.add_argument('--momentum', nargs='?', type=float, default=0.99,
                            help='Momentum for the optimizer')
    parser_.add_argument('--weight_decay', nargs='?', type=float, default=5e-4,
                            help='Weight decay for the optimizer')
    parser_.add_argument('--loss', nargs='?', type=str, default='crossentropy2d',
                            help='Loss function')
    parser_.add_argument('--epochs', nargs='?', type=int, default=100,
                            help='Number of training epochs')
    parser_.add_argument('--batch_size', nargs='?', type=int, default=32,
                            help='Batch size for training')
    parser_.add_argument('--evaluate', nargs='?', type=int, default=20,
                            help='Evaluate each number of epochs')
    parser_.add_argument('--img_width', nargs='?', type=int, default=256,
                            help='Image width')
    parser_.add_argument('--img_height', nargs='?', type=int, default=256,
                            help='Image height')
    parser_.add_argument('--checkpoint', nargs='?', type=str, default=None,
                            help='Checkpoint file to resume training')
    parser_.add_argument('--finetune', nargs='?', type=bool, default=False,
                            help='Active finetuning so the optimizer is not loaded')
    args_ = parser_.parse_args()
    train(args_)
