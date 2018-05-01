import network.unet_network

networks = { 'unet' : network.unet_network.UNetNetwork }

def get_network(name, numClasses):
    
    return networks[name](numClasses)
