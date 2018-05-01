import argparse

import loader.utils

def train(args):

    # Dataloader
    data_loader_ = loader.utils.get_loader(args.dataset)
    data_path_ = loader.utils.get_path(args.dataset)

    train_loader_ = data_loader_(data_path_, 'train', 512, 512, isTransform=True)
    test_loader_ = data_loader_(data_path_, 'test', 512, 512, isTransform=True)
    val_loader_ = data_loader_(data_path_, 'val', 512, 512, isTransform=True)

if __name__ == '__main__':

    parser_ = argparse.ArgumentParser(description='Parameters')
    parser_.add_argument('--dataset', nargs='?', type=str, default='cityscapes',
                            help='The dataset to be used for training.')

    args_ = parser_.parse_args()
    train(args_)
