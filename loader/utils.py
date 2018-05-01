import os

import loader.ade20k_loader
import loader.cityscapes_loader

loaders_ = { 'ade20k' : loader.ade20k_loader.ADE20KLoader,
             'cityscapes': loader.cityscapes_loader.CityscapesLoader }

paths_ = { 'ade20k' : 'datasets/ade20k',
            'cityscapes' : 'datasets/cityscapes' }

def get_loader(name):
    return loaders_[name]

def get_path(name):
    return paths_[name]

def recursive_glob(root='.', suffix=''):
    files_ = [os.path.join(dirpath, filename)
                for dirpath, _, filenames in os.walk(root)
                for filename in filenames if filename.endswith(suffix)]
    return files_
