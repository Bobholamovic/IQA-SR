import torch
import torch.nn as nn
import torch.nn.functional as F

from models.iqanet import IQANet
from dataset.dataset import get_dataset
from constants import DATASET


# Losses
class IQALoss(nn.Module):
    def __init__(self, path_to_model_weight, patch_size, feat_names):
        super(IQALoss, self).__init__()

        self.iqa_model = IQANet(weighted=False)
        self.iqa_model.load_state_dict(torch.load(path_to_model_weight)['state_dict'])

        # Freeze the parameters
        for p in self.iqa_model.parameters():
            p.requires_grad = False

        self.patch_size = patch_size
        self.feat_names = feat_names
        self._denorm = get_dataset(DATASET).denormalize

    def forward(self, output, target):
        output = self._renormalize(output)
        target = self._renormalize(target)
        output_patches = self._extract_patches(output)
        target_patches = self._extract_patches(target)

        self.iqa_model.eval()
        rets = self.iqa_model(output_patches, target_patches)
        
        sel_feats = [v for k,v in rets.features.items() if k in self.feat_names]

        for i, f in enumerate(sel_feats):
            if isinstance(f, tuple):
                assert len(f) == 2
                sel_feats[i] = F.mse_loss(*f)
            else:
                sel_feats[i] = torch.mean(torch.abs(f))

        return torch.stack(sel_feats, dim=0)
        
    def _extract_patches(self, img):
        h, w = img.shape[-2:]
        nh, nw = h//self.patch_size, w//self.patch_size
        ch, cw = nh*self.patch_size, nw*self.patch_size
        bh, bw = (h-ch)//2, (w-cw)//2

        vpatchs = torch.stack(torch.split(img[...,bh:bh+ch,:], self.patch_size, dim=-2), dim=1)
        patchs = torch.cat(torch.split(vpatchs[...,bw:bw+cw], self.patch_size, dim=-1), dim=1)

        return patchs

    def _renormalize(self, img):
        return self._denorm(img, 'hr')/255.0


class ComLoss(nn.Module):
    def __init__(self, model_path, weights, feat_names, patch_size=32, criterion='MAE'):
        super(ComLoss, self).__init__()

        if criterion == 'MAE':
            self._pixel_criterion = F.l1_loss
        elif criterion == 'MSE':
            self._pixel_criterion = F.mse_loss
        elif criterion == 'IQA':
            self._pixel_criterion = self._none
            assert weights is not None
        else:
            if hasattr(criterion, '__call__'):
                self.criterion = criterion
            raise ValueError('invalid criterion')
            
        self.weights = weights
        if self.weights is not None:
            assert len(weights) == len(feat_names)
            self.weights = torch.FloatTensor(weights)
            if torch.cuda.is_available(): self.weights = self.weights.cuda()
            self.iqa_loss = IQALoss(model_path, patch_size, feat_names)
            self._feat_criterion = self._calc_feat_loss
        else:
            self._feat_criterion = self._none

    def forward(self, output, target):
        pixel_loss = self._pixel_criterion(output, target)
        feat_loss = self._feat_criterion(output, target)

        total_loss = pixel_loss + feat_loss

        if self.training:
            return total_loss, pixel_loss, feat_loss
        else:
            return total_loss

    def _calc_feat_loss(self, output, target):
        if self.training:
            return torch.sum(self.weights*self.iqa_loss(output, target))
        else:
            return self._none(output, target)

    def _none(self, output, target):
        return torch.tensor(0.0).type_as(output)
