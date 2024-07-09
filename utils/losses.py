import torch
import torch.nn as nn


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits_img, logits_text):
        batch_size = logits_img.shape[0]
        labels = torch.arange(batch_size).to(logits_img.device)
        criterion = nn.CrossEntropyLoss()
        loss_img = criterion(logits_img, labels)
        loss_text = criterion(logits_text, labels)

        loss = (loss_img + loss_text) / 2.0

        return loss


class AsymmetricLoss(nn.Module):
    """
    modified from Asymmetric Loss For Multi-Label Classification https://github.com/Alibaba-MIIL/ASL
    """

    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-6,
        disable_torch_grad_focal_loss=True,
    ):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        if x.dim() == 3:
            x = self.softmax(x)
            xs_pos = x[:, 0, :]
            xs_neg = x[:, 1, :]
        else:
            x = self.sigmoid(x)
            xs_pos = x
            xs_neg = 1 - x
        y = y.reshape(-1)
        xs_pos = xs_pos.reshape(-1)
        xs_neg = xs_neg.reshape(-1)

        xs_pos = xs_pos[y != -1]
        xs_neg = xs_neg[y != -1]
        y = y[y != -1]

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class MultilabelPairLoss(nn.Module):

    def __init__(self, enable_maxpool=True):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.enable_pool = enable_maxpool

    def forward(self, x, y):
        if self.enable_pool:
            y[y == -1] = 0
            mask = torch.triu(torch.ones(y.shape[1], y.shape[1]), diagonal=1).flatten()
            y = torch.triu(y.t() @ y, diagonal=1)
            y = y.flatten()[mask == 1]

            y[y > 1] = 1
            x, _ = torch.max(x, dim=0)
        else:
            mask = []
            for i in range(y.shape[0]):
                combinations = torch.combinations(y[i, :], r=2).sum(dim=-1)  # [64, 91]
                mask.append((combinations == 2).float())  # [64, 91]
            y = torch.stack(mask, dim=0)
        loss = self.criterion(x, y)
        return loss


# ranking loss
class LossRanking(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, weight_a=0.1, weight_b=0.07):
        super().__init__()
        self.criterion1 = AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos)
        self.criterion2 = nn.MultiLabelMarginLoss()
        self.softmax = nn.Softmax(dim=1)
        self.weight_a = weight_a
        self.weight_b = weight_b

    def forward(self, x, y, z):
        loss1 = self.weight_a * self.criterion1(x, y)
        x_pos = self.softmax(x)[:, 0].reshape(-1)
        y = y.reshape(-1)
        x_pos = x_pos[y != -1]
        y = y[y != -1]
        indices = torch.nonzero(y).reshape(-1)
        target = torch.full_like(x_pos, -1)
        target[: len(indices)] = indices

        loss2 = self.weight_b * self.criterion2(x_pos, target.long())

        loss = loss1 + loss2
        return loss


class LossMixture(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, weight_a=0.1, weight_b=1):
        super().__init__()
        self.criterion1 = AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos)
        self.criterion2 = MultilabelPairLoss()
        self.softmax = nn.Softmax(dim=1)
        self.weight_a = weight_a
        self.weight_b = weight_b

    def forward(self, x, y, z):
        loss1 = self.weight_a * self.criterion1(x, y)
        loss2 = self.weight_b * self.criterion2(z, y)

        loss = loss1 + loss2
        return loss
