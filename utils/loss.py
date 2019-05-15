import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

from models.iqanet import IQANet
from dataset.dataset import get_dataset
from constants import DATASET


# Losses
class IQALoss(nn.Module):
    def __init__(self, path_to_model_weight, patch_size, feat_names):
        super(IQALoss, self).__init__()

        self.iqa_model = IQANet(weighted=False)
        self.iqa_model.load_state_dict(torch.load(path_to_model_weight)['state_dict'])

        self.patch_size = patch_size
        self.feat_names = feat_names
        self._denorm = get_dataset(DATASET).denormalize

    def forward(self, output, target):
        self.iqa_model.eval()   # Switch to eval
        rets = self.iqa_forward(output, target)

        sel_feats = []
        # The features are fetched and stored in the order of feat_names
        for k in self.feat_names:
            sel_feats.append(rets.features.__getitem__(k))

        for i, f in enumerate(sel_feats):
            if isinstance(f, tuple):
                assert len(f) == 2
                # Looks like that F.mse_loss gives unexpected values when
                # the target requires grad
                sel_feats[i] = F.mse_loss(f[0], f[1].detach())
            else:
                sel_feats[i] = torch.mean(torch.abs(f))

        return torch.stack(sel_feats, dim=0)

    def iqa_forward(self, output, target):
        output = self.renormalize(output)
        target = self.renormalize(target)
        
        output_patches = self._extract_patches(output)
        target_patches = self._extract_patches(target)

        return self.iqa_model(output_patches, target_patches)
        
    def _extract_patches(self, img):
        h, w = img.shape[-2:]
        nh, nw = h//self.patch_size, w//self.patch_size
        ch, cw = nh*self.patch_size, nw*self.patch_size
        bh, bw = (h-ch)//2, (w-cw)//2

        vpatchs = torch.stack(torch.split(img[...,bh:bh+ch,:], self.patch_size, dim=-2), dim=1)
        patchs = torch.cat(torch.split(vpatchs[...,bw:bw+cw], self.patch_size, dim=-1), dim=1)

        return patchs

    def renormalize(self, img):
        # Clamp to [0, 255] before normalizing
        return torch.clamp(self._denorm(img, 'hr'), 0, 255)/255.0

    def freeze(self):
        # Freeze the parameters
        for p in self.iqa_model.parameters():
            p.requires_grad = False

    def unfreeze(self):
        # Freeze the parameters
        for p in self.iqa_model.parameters():
            p.requires_grad = True

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.iqa_model.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.iqa_model.load_state_dict(state_dict, strict)

    def parameters(self):
        return self.iqa_model.parameters()

    @contextmanager
    def learner(self):
        self.unfreeze()
        yield self.iqa_model
        self.freeze()


class ComLoss(nn.Module):
    def __init__(
        self, model_path, weights, feat_names,
        alpha=1.0, patch_size=32, criterion='MAE'
    ):
        super(ComLoss, self).__init__()

        if criterion == 'MAE':
            self.pixel_criterion = F.l1_loss
        elif criterion == 'MSE':
            self.pixel_criterion = F.mse_loss
        elif criterion == 'IQA':
            self.pixel_criterion = self._none
            assert weights is not None
        elif hasattr(criterion, '__call__'):
            self.pixel_criterion = criterion
        else:
            raise ValueError('invalid criterion')
            
        self.alpha = alpha
        self.weights = weights
        if self.weights is not None:
            assert len(weights) == len(feat_names)
            self.weights = torch.FloatTensor(weights)
            if torch.cuda.is_available(): self.weights = self.weights.cuda()
            self.iqa_loss = IQALoss(model_path, patch_size, feat_names)
            self.feat_criterion = self._calc_feat_loss
        else:
            self.feat_criterion = self._none

    def forward(self, output, target):
        pixel_loss = self.pixel_criterion(output, target)
        
        if not self.training:
            return pixel_loss

        feat_loss = self.feat_criterion(output, target)
        tv_loss = self._calc_tv_loss(output)

        total_loss = self.alpha*pixel_loss + feat_loss + 1e-3*tv_loss

        return total_loss, pixel_loss, feat_loss

    def _calc_feat_loss(self, output, target):
        return torch.sum(self.weights*self.iqa_loss(output, target))

    def _none(self, output, target):
        return torch.tensor(0.0).type_as(output)

    def _calc_tv_loss(self, x):
        # Copied from https://github.com/jxgu1016/Total_Variation_Loss.pytorch/blob/master/TVLoss.py
        def _tensor_size(t):
            return t.size()[1]*t.size()[2]*t.size()[3]
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = _tensor_size(x[:,:,1:,:])
        count_w = _tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return 2*(h_tv/count_h+w_tv/count_w)/batch_size
