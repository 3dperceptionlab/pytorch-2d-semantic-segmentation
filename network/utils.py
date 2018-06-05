import network.unet_network
import network.pspnet

networks = { 'unet' : network.unet_network.UNetNetwork, 
'pspnet_resnet18' : lambda numClasses: network.pspnet.PSPNet(n_classes= numClasses, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
'pspnet_resnet34' : lambda numClasses: network.pspnet.PSPNet(n_classes= numClasses, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
'pspnet_resnet50' : lambda numClasses: network.pspnet.PSPNet(n_classes= numClasses, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
'pspnet_resnet101': lambda numClasses: network.pspnet.PSPNet(n_classes= numClasses, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
'pspnet_resnet152': lambda numClasses: network.pspnet.PSPNet(n_classes= numClasses, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')}

def get_network(name, numClasses):
    
    return networks[name](numClasses)
