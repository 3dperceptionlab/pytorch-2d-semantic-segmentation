import loss.crossentropy2d_loss_new
import loss.crossentropy2d

losses = { 'crossentropy2d' : loss.crossentropy2d_loss_new.CrossEntropy2DLoss,
		   'crossentropy' : loss.crossentropy2d.CrossEntropy2D }

def get_loss(name):
    return losses[name]
