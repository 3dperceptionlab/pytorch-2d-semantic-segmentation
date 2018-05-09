import argparse
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
import torch.utils

import loader.utils

log = logging.getLogger(__name__)

def stats(args):

    loader_ = loader.utils.get_loader(args.dataset)
    path_ = loader.utils.get_path(args.dataset)

    data_loader_ = loader_(path_, args.split, args.img_width, args.img_height, isTransform=True)
    log.info(data_loader_)

    torch_data_loader_ = torch.utils.data.DataLoader(data_loader_, batch_size=1, num_workers=1, shuffle=False)

    num_classes_ = data_loader_.num_classes
    class_map_ = data_loader_.class_map
    classes_ = data_loader_.valid_classes
    class_names_ = data_loader_.class_names
    num_images_ = len(data_loader_)

    class_image_count_ = np.zeros(num_classes_)
    class_image_percentage_ = np.zeros(num_classes_)
    class_pixel_count_ = np.zeros(num_classes_)
    class_pixel_percentage_ = np.zeros(num_classes_)

    for c in range(num_classes_):
        class_image_count_[c] = 0
        class_image_percentage_[c] = 0
        class_pixel_count_[c] = 0
        class_pixel_percentage_[c] = 0

    log.info('Number of images: {0}'.format(num_images_))
    log.info('Number of classes: {0}'.format(num_classes_))
    log.info('Class map:')

    print(class_map_)

    for k, v in class_map_.items():
        log.info('Class {0} with name {1} : {2}'.format(k, class_names_[v], v))

    for i, (img, lbl, lbl_rgb) in tqdm.tqdm(enumerate(torch_data_loader_), total=len(data_loader_)):

        labels_ = np.squeeze(lbl.numpy())
        pixel_count_ = np.bincount(labels_.flatten())

        for c in range(num_classes_):

            if pixel_count_[c] > 0:

                class_image_count_[c] += 1.0

            class_pixel_count_[c] += pixel_count_[c]

        log.debug(class_pixel_count_)
        log.debug(class_image_count_)

        if i > 10:
            break

    for c in range(num_classes_):

        class_image_percentage_[c] = class_image_count_[c] * 100.0 / np.sum(class_image_count_)
        class_pixel_percentage_[c] = class_pixel_count_[c] * 100.0 / np.sum(class_pixel_count_)

    log.debug('Sum of percentages of images: {0}'.format(np.sum(class_image_percentage_)))
    log.debug('Sum of percentages of pixels: {0}'.format(np.sum(class_pixel_percentage_)))

    log.info("Summary:")
    
    for c in range(num_classes_):

        log.info('Class {} appears in {:d} images ({:06.2f}%), totalling {:d} pixels ({:06.2f}%)'.format(
                    class_names_[c],
                    int(class_image_count_[c]),
                    class_image_percentage_[c],
                    int(class_pixel_count_[c]),
                    class_pixel_percentage_[c]))

    # Plot per-class image percentage
    x_labels_ = class_names_
    x_values_ = np.arange(len(x_labels_)-1)
    y_values_ = class_image_percentage_
    y_label_ = 'Percentage of Images (%)'

    fig_, ax_ = plt.subplots()
    barlist_ = ax_.barh(x_values_, y_values_)
    ax_.set_yticks(x_values_)
    ax_.set_yticklabels(x_labels_)
    ax_.invert_yaxis()
    ax_.set_xlim(0, 100)
    ax_.set_xlabel(y_label_)
    ax_.set_title('Per-class Image Percentage')
    plt.show()

    # Plot per-class pixel percentage
    x_labels_ = class_names_
    x_values_ = np.arange(len(x_labels_)-1)
    y_values_ = class_pixel_percentage_
    y_label_ = 'Percentage of Images (%)'

    fig_, ax_ = plt.subplots()
    barlist_ = ax_.barh(x_values_, y_values_)
    ax_.set_yticks(x_values_)
    ax_.set_yticklabels(x_labels_)
    ax_.invert_yaxis()
    ax_.set_xlim(0, 100)
    ax_.set_xlabel(y_label_)
    ax_.set_title('Per-class Pixel Percentage')
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

    stats(args_)
