import torch

from torch import nn


class FocalLoss(nn.Module):
    __name__ = 'focal'

    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = torch.nn.functional.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size()) == 3:
            loss = loss.sum(dim=1)
        return loss.mean()

class CrossEntropyLoss(nn.Module):
    __name__ = 'CrossEntropyLoss'

    def __init__(self):
        super().__init__()
        self.cross = nn.CrossEntropyLoss()

    def forward(self, logit, target):
        target = torch.argmax(target, dim=1)
        return self.cross(logit, target)

