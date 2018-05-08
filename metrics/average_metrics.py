import numpy as np

class AverageMetrics(object):

    def __init__(self, numClasses):

        self.num_classes = numClasses
        self.hist_ = np.zeros((numClasses, numClasses))

    def _fast_hist(self, predictions, groundTruth, numClasses):
        
        mask_ = (groundTruth >= 0) & (groundTruth < numClasses)
        hist_ = np.bincount(
                numClasses * groundTruth[mask_].astype(int) +
                predictions[mask_], minlength=numClasses ** 2).reshape(numClasses, numClasses)

        return hist_

    def update(self, predictions, groundTruth):

        for lp, lt in zip(predictions, groundTruth):

            self.hist_ += self._fast_hist(lp.flatten(), lt.flatten(), self.num_classes)

    def evaluate(self):

        acc_ = np.diag(self.hist_).sum() / self.hist_.sum()

        acc_cls_ = np.diag(self.hist_) / self.hist_.sum(axis=1)
        acc_cls_ = np.nanmean(acc_cls_)

        iou_ = np.diag(self.hist_) / (self.hist_.sum(axis=1) + self.hist_.sum(axis=0) - np.diag(self.hist_))

        mean_iou_ = np.nanmean(iou_)

        freq_ = self.hist_.sum(axis=1) / self.hist_.sum()
        fwacc_ = (freq_[freq_ > 0] * iou_[freq_ > 0]).sum()

        iou_cls_ = dict(zip(range(self.num_classes), iou_))

        return {'acc' : acc_cls_,
                'fwacc' : fwacc_,
                'iou' : mean_iou_}

    def reset(self):

        self.hist_ = np.zeros((self.num_classes, self.num_classes))
