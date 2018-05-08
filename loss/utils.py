import loss.crossentropy2d_loss_new

losses = { 'crossentropy2d' : loss.crossentropy2d_loss_new.CrossEntropy2DLoss }

def get_loss(name):

    return losses[name]
