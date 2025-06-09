import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


### Regularization Terms
def stable_rank(x):
    x = F.normalize(x, dim=-1)
    return -(
        (torch.linalg.matrix_norm(x, ord="fro").pow(2))
        / (torch.linalg.matrix_norm(x, ord=2).pow(2))
        / x.shape[1]
    )

def uniformity(x):
    x = F.normalize(x, dim=-1)
    return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()


# Losses
class AlignmentLoss(_Loss):
    def __init__(self):
        super().__init__()

    @staticmethod
    def alignment(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def forward(
        self,
        paired_user,
        paired_item,
        embed_user=None,
        embed_item=None,
        regs=None,
        gammas=None,
    ):
        # Assume that paired_user and paired_item are set up such that index i in each corresponds to the i-th user-item interaction
        loss = self.alignment(paired_user, paired_item)
        for r, reg in enumerate(regs):
            loss += gammas[r] * (eval(reg)(embed_user) + eval(reg)(embed_item)) / 2

        return loss


class BPRLoss(_Loss):
    def __init__(self):
        super().__init__(None, None, "sum")

    def forward(
        self,
        preds,
        embed_user=None,
        embed_item=None,
        neg_ratio=1,
        regs=None,
        gammas=None,
    ):

        # will assume that preds/labels are ordered such that the ith entry of pos and neg entry are same source
        pred_sets = preds.chunk(neg_ratio + 1)
        # split the pred sets, the first chunk is positives, and then we have self.neg_ratio chunks of negative samples
        pos = pred_sets[0]
        # restack the negative samples
        neg = torch.vstack(pred_sets[1:])

        # note that if there are more than 1 neg samples, this will broadcast pos to neg
        loss = -F.logsigmoid(pos - neg).mean()
        for r, reg in enumerate(regs):
            loss += gammas[r] * (eval(reg)(embed_user) + eval(reg)(embed_item)) / 2

        return loss

class SSMLoss(_Loss):
    def __init__(self):
        super().__init__(None, None, "sum")

    def forward(self,  
                preds,
                embed_user=None,
                embed_item=None,
                neg_ratio=1,
                regs=None,
                gammas=None): 
        
        # will assume that preds/labels are ordered such that the ith entry of pos and neg entry are same source
        pred_sets = preds.chunk(neg_ratio + 1)
        # split the pred sets, the first chunk is positives, and then we have self.neg_ratio chunks of negative samples
        pos = pred_sets[0]
        # restack the negative samples
        neg = torch.stack(pred_sets[1:], dim=1)
     
        scores = torch.cat((pos.unsqueeze(1), neg), dim=1)
        probs = F.softmax(scores, dim=1)
        hit_probs = probs[:, 0]
        loss = -torch.log(hit_probs).mean()
   
        for r, reg in enumerate(regs):
            loss += gammas[r] * (eval(reg)(embed_user) + eval(reg)(embed_item)) / 2

        return loss
