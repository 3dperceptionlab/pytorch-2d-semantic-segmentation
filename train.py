import argparse

import torch.nn

import loader.utils
import network.utils

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

if __name__ == '__main__':

    parser_ = argparse.ArgumentParser(description='Parameters')
    parser_.add_argument('--dataset', nargs='?', type=str, default='cityscapes',
                            help='The dataset to be used for training.')
    parser_.add_argument('--network', nargs='?', type=str, default='unet',
                            help='The network architecture to be used for training.')

    args_ = parser_.parse_args()
    train(args_)
