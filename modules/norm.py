import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np


def calc_ins_mean_std(x, eps=1e-5):
    """extract feature map statistics"""
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = x.size()
    N, C = x.shape[0], x.shape[2]
    var = x.contiguous().view(N,-1, C).var(dim=1) + eps
    std = var.sqrt().view(N,1,C)
    mean = x.contiguous().view(N, -1, C).mean(dim=1).view(N,1, C)
    return mean, std


def instance_norm_mix(content_feat, style_feat):
    """replace content statistics with style statistics"""
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_ins_mean_std(style_feat)
    content_mean, content_std = calc_ins_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def cn_rand_bbox(size, beta, bbx_thres):
    """sample a bounding box for cropping."""
    W = size[2]
    H = size[3]
    while True:
        ratio = np.random.beta(beta, beta)
        cut_rat = np.sqrt(ratio)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        ratio = float(bbx2 - bbx1) * (bby2 - bby1) / (W * H)
        if ratio > bbx_thres:
            break

    return bbx1, bby1, bbx2, bby2


def cn_op_2ins_space_chan(x, crop='neither', beta=1, bbx_thres=0.1, lam=None, chan=False):
    """2-instance crossnorm with cropping."""

    assert crop in ['neither', 'style', 'content', 'both']
    ins_idxs = torch.randperm(x.size()[0]).to(x.device)

    if crop in ['style', 'both']:
        bbx3, bby3, bbx4, bby4 = cn_rand_bbox(x.size(), beta=beta, bbx_thres=bbx_thres)
        x2 = x[ins_idxs, :, bbx3:bbx4, bby3:bby4]
    else:
        x2 = x[ins_idxs]

    if chan:
        chan_idxs = torch.randperm(x.size()[1]).to(x.device)
        x2 = x2[:, chan_idxs, :, :]

    if crop in ['content', 'both']:
        x_aug = torch.zeros_like(x)
        bbx1, bby1, bbx2, bby2 = cn_rand_bbox(x.size(), beta=beta, bbx_thres=bbx_thres)
        x_aug[:, :, bbx1:bbx2, bby1:bby2] = instance_norm_mix(content_feat=x[:, :, bbx1:bbx2, bby1:bby2],
                                                              style_feat=x2)

        mask = torch.ones_like(x, requires_grad=False)
        mask[:, :, bbx1:bbx2, bby1:bby2] = 0.
        x_aug = x * mask + x_aug
    else:
        x_aug = instance_norm_mix(content_feat=x, style_feat=x2)

    if lam is not None:
        x = x * lam + x_aug * (1-lam)
    else:
        x = x_aug

    return x


class CrossNorm(nn.Module):
    """CrossNorm module"""
    def __init__(self, crop=None, beta=None):
        super(CrossNorm, self).__init__()

        self.active = False
        self.cn_op = functools.partial(cn_op_2ins_space_chan,
                                       crop=crop, beta=beta)

    def forward(self, x):
        if self.training and self.active:

            x = self.cn_op(x)

        self.active = False

        return x


class SelfNorm(nn.Module):
    """SelfNorm module"""
    def __init__(self, chan_num, is_two=False):
        super(SelfNorm, self).__init__()

        # channel-wise fully connected layer
        self.g_fc = nn.Sequential(nn.Linear(chan_num*2, int(chan_num//2)), nn.GELU(), nn.Linear(int(chan_num//2), chan_num))
        self.g_bn = nn.BatchNorm1d(chan_num)

        if is_two is True:
            self.f_fc = nn.Sequential(nn.Linear(chan_num*2, int(chan_num//2)), nn.GELU(), nn.Linear(intq(chan_num//2), chan_num))
            self.f_bn = nn.BatchNorm1d(chan_num)
        else:
            self.f_fc = None

    def forward(self, x):
        b, _, c = x.size()
        #x = x.contiguous().permute(0,2,1)
        
        mean, std = calc_ins_mean_std(x, eps=1e-12)

        statistics = torch.cat([mean, std], -1)

        g_y = self.g_fc(statistics)
        #print(g_y.shape)
        g_y = self.g_bn(g_y.contiguous().permute(0,2,1)).contiguous().permute(0,2,1)
        g_y = torch.sigmoid(g_y)
        g_y = g_y.view(b,-1,c)

        if self.f_fc is not None:
            f_y = self.f_fc(statistics)
            f_y = self.f_bn(f_y.contiguous().permute(0,2,1)).contiguous().permute(0,2,1)
            f_y = torch.sigmoid(f_y)
            f_y = f_y.view(b, c, 1, 1)
            x = x * g_y.expand_as(x) + mean.expand_as(x) * (f_y.expand_as(x)-g_y.expand_as(x))
            return x.contiguous()#.permute(0,2,1)
        else:
            x = x * g_y.expand_as(x)
            return x.contiguous()#.permute(0,2,1)

class CNSN(nn.Module):
    """A module to combine CrossNorm and SelfNorm"""
    def __init__(self, crossnorm, selfnorm):
        super(CNSN, self).__init__()
        self.crossnorm = crossnorm
        self.selfnorm = selfnorm

    def forward(self, x):
        if self.crossnorm and self.crossnorm.active:
            x = self.crossnorm(x)
        if self.selfnorm:
            x = self.selfnorm(x)
        return x