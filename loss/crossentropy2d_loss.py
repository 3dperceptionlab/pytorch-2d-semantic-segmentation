import torch.nn as nn
import torch.nn.functional as F

class CrossEntropy2DLoss(nn.Module):

    def __init__(self, weight=None, sizeAverage=True, ignoreIndex=255):

        super().__init__()

        self.weight = weight
        self.size_average = sizeAverage
        self.ignore_index = ignoreIndex

        # CHECK: NLLLoss2d deprecated??
        self.nll_loss = nn.NLLLoss(weight, sizeAverage, ignoreIndex)

    def __repr__(self):

        return "Cross-entropy 2D loss with weight {0} and ignoring index {1}".format(
                    self.weight,
                    self.ignore_index)

    def forward(self, inputs, targets):

        return self.nll_loss(F.log_softmax(inputs), targets)
