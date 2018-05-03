import loss.crossentropy2d_loss

losses = { 'crossentropy2d' : loss.crossentropy2d_loss.CrossEntropy2DLoss }

def get_loss(name):

    return losses[name]
