import os

import loader.cityscapes_loader

loaders_ = { 'cityscapes': loader.cityscapes_loader.CityscapesLoader }

paths_ = { 'cityscapes' : '/src/datasets_asimov/cityscapes' }

def get_loader(name):
    return loaders_[name]

def get_path(name):
    return paths_[name]

def recursive_glob(root='.', suffix=''):
    files_ = [os.path.join(dirpath, filename)
                for dirpath, _, filenames in os.walk(root)
                for filename in filenames if filename.endswith(suffix)]
    return files_
