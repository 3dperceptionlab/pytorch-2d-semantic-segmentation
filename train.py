import argparse

import torch.nn

import loader.utils
import network.utils
import loss.utils

def train(args):

    # Data loader
    data_loader_ = loader.utils.get_loader(args.dataset)
    data_path_ = loader.utils.get_path(args.dataset)

    train_loader_ = data_loader_(data_path_, 'train', 512, 512, isTransform=True)
    print(train_loader_)
    test_loader_ = data_loader_(data_path_, 'test', 512, 512, isTransform=True)
    print(test_loader_)
    val_loader_ = data_loader_(data_path_, 'val', 512, 512, isTransform=True)
    print(val_loader_)

    num_classes_ = train_loader_.num_classes

    # Network loader
    network_ = network.utils.get_network(args.network, num_classes_)
    network_ = torch.nn.DataParallel(network_, device_ids=range(torch.cuda.device_count()))
    network_.cuda()
    print(network_)

    # Set up optimizer
    optimizer_ = torch.optim.SGD(network_.parameters(),
                                    lr=args.learning_rate,
                                    momentum=0.99,
                                    weight_decay=5e-4)

    # Set up loss
    loss_function_ = loss.utils.get_loss(args.loss)
    criterion_ = loss_function_(sizeAverage=False, ignoreIndex=train_loader_.ignore_index)
    criterion_.cuda()
    print(criterion_)

if __name__ == '__main__':

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
    args_ = parser_.parse_args()
    train(args_)
