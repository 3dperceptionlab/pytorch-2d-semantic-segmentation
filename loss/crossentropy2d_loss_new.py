import torch.nn as nn
import torch.nn.functional as F

class CrossEntropy2DLoss(nn.Module):

    def __init__(self, weight=None, sizeAverage=True, ignoreIndex=255):

        super().__init__()

        self.weight = weight
        self.size_average = sizeAverage
        self.ignore_index = ignoreIndex

    def __repr__(self):

        return "Cross-entropy 2D loss with weight {0} and ignoring index {1}".format(
                    self.weight,
                    self.ignore_index)

    def forward(self, inputs, targets):
        
        n, c, h, w = inputs.size()
        nt, ht, wt = targets.size()

        # Handle inconsistent size between input and target
        if h > ht and w > wt: # upsample labels
            targets = targets.unsequeeze(1)
            targets = F.upsample(targets, size=(h, w), mode='nearest')
            targets = targets.sequeeze(1)
        elif h < ht and w < wt: # upsample images
            inputs = F.upsample(inputs, size=(ht, wt), mode='bilinear')
        elif h != ht and w != wt:
            raise Exception("Only support upsampling")

        log_p = F.log_softmax(inputs, dim=1)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        log_p = log_p[targets.view(-1, 1).repeat(1, c) >= 0]
        log_p = log_p.view(-1, c)

        mask = targets >= 0
        targets = targets[mask]
        loss = F.nll_loss(log_p, targets, ignore_index=self.ignore_index,
                      weight=self.weight, size_average=self.size_average)

        if self.size_average:
            loss /= mask.data.sum()
        return loss
