import network.unet_network
import network.pspnet

networks = { 'UNET' : network.unet_network.UNetNetwork, 
			 'PSPNET' : lambda numClasses, sizes, psp_size, deep_features_size, backend: network.pspnet.PSPNet(n_classes= numClasses, 
			 	sizes= sizes, psp_size= psp_size, deep_features_size= deep_features_size, backend= backend)
}

def get_network(config):
    net_ = None
    net_config = config["NETWORK"]
    dataset_config = config["DATASET"]

    if net_config['NAME'] == 'UNET':
    	net_ = networks[net_config['NAME']](dataset_config['NUM_CLASSES'])
    elif net_config['NAME'] == 'PSPNET':
    	net_ = networks[net_config['NAME']](dataset_config['NUM_CLASSES'], net_config['SIZES'], net_config['PSP_SIZE'], net_config['DEEP_FEATURES_SIZE'], net_config['BACKEND'])

    return net_
