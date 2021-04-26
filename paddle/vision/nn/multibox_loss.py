import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..utils import box_utils


# class MultiboxLoss(object):
class MultiboxLoss(nn.Layer):
    def __init__(self, priors, neg_pos_ratio, center_variance, size_variance):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.shape[2]
        with paddle.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, 2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = paddle.concat([confidence[:, :, 0].masked_select(mask).reshape([-1, 1]),
                                    confidence[:, :,1].masked_select(mask).reshape([-1, 1])], axis=1)
        classification_loss = F.cross_entropy(confidence.reshape([-1, num_classes]), labels.masked_select(mask), reduction='sum')
        pos_mask = labels > 0
        predicted_locations = predicted_locations.masked_select(paddle.concat([pos_mask.reshape(pos_mask.shape+[1]), pos_mask.reshape(pos_mask.shape+[1]), pos_mask.reshape(pos_mask.shape+[1]), pos_mask.reshape(pos_mask.shape+[1])], axis=2)).reshape([-1, 4])
        gt_locations = gt_locations.masked_select(paddle.concat([pos_mask.reshape(pos_mask.shape+[1]), pos_mask.reshape(pos_mask.shape+[1]), pos_mask.reshape(pos_mask.shape+[1]), pos_mask.reshape(pos_mask.shape+[1])], axis=2)).reshape([-1, 4])
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations.cast('float32'), reduction='sum')  # smooth_l1_loss
        # smooth_l1_loss = F.mse_loss(predicted_locations, gt_locations, reduction='sum')  #l2 loss
        num_pos = gt_locations.shape[0]
        return smooth_l1_loss / num_pos, classification_loss / num_pos
