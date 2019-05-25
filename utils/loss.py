import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from pdb import set_trace as db
from contextlib import contextmanager
from collections import OrderedDict

from models.iqanet import IQANet
from dataset.dataset import get_dataset
from constants import DATASET


# Losses
class IQALoss(nn.Module):
    def __init__(self, path_to_model_weight, patch_size, feat_names):
        super(IQALoss, self).__init__()

        self.iqa_model = IQANet(weighted=False)
        if os.path.exists(path_to_model_weight):
            self.iqa_model.load_state_dict(
                torch.load(path_to_model_weight)['state_dict']
            )

        self.patch_size = patch_size
        # Strip nr and invalid names
        self.feat_names = [
            n 
            for n in feat_names
            # not nr 
            if n != 'nr' 
            and 
            # exists in the model
            hasattr(self.iqa_model, n)
        ]
        self.regular = 'nr' in feat_names
        self._denorm = get_dataset(DATASET).denormalize

    def forward(self, output, target):
        self.iqa_model.eval()   # Switch to eval

        feat_t = self.prepare(self.new_feature_dict())
        self.iqa_forward(target.data)
        self.done()

        feat_o = self.prepare(self.new_feature_dict())
        score_o = self.iqa_forward(output)
        self.done()

        # losses = [F.mse_loss(feat_o[n], feat_t[n]) for n in self.feat_names]

        losses = [
            F.mse_loss(fo, ft)
            for fo, ft 
            in zip(feat_o, feat_t)
        ]

        if self.regular:
            # Put nr loss on the last
            losses.append(torch.pow(score_o/100.0, 2).mean())

        return torch.stack(losses, dim=0)

    def iqa_forward(self, x):
        x = self.renormalize(x)
        
        patches = self._extract_patches(x)

        return self.iqa_model(patches)
        
    def _extract_patches(self, img):
        h, w = img.shape[-2:]
        nh, nw = h//self.patch_size, w//self.patch_size
        ch, cw = nh*self.patch_size, nw*self.patch_size
        bh, bw = (h-ch)//2, (w-cw)//2

        vpatchs = torch.stack(torch.split(img[...,bh:bh+ch,:], self.patch_size, dim=-2), dim=1)
        patchs = torch.cat(torch.split(vpatchs[...,bw:bw+cw], self.patch_size, dim=-1), dim=1)

        # # Random selection to introduce noise
        # n = patchs.size(1)  # The number of patches
        # return torch.index_select(
        #     patchs, 
        #     1, 
        #     torch.randperm(n)[:n//2].to(patchs.device)
        # )
        return patchs

    def prepare(self, features):
        from functools import partial

        def _hook(m, i, o, n=''):
            # To retain gradients, store the identical
            features[n] = o
        
        # Register new
        self._handles = [
            l.register_forward_hook(partial(_hook, n=n))
            for n, l in self.iqa_model.named_children()
            if n in self.feat_names
        ]

        return features.values()

    def new_feature_dict(self):
        # Keep in order
        return OrderedDict(
            zip(self.feat_names,
            (None,)*len(self.feat_names))
        )

    def done(self):
        # Remove old hooks
        # This is important in that the references to the feature dict
        # in the hook functions are removed such that it would be recycled
        for h in self._handles:
            h.remove()

    def renormalize(self, img):
        # # Clamp to [0, 255] before normalizing
        # return torch.clamp(self._denorm(img, 'hr'), 0, 255)/255.0
        return img

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
        is_training = self.iqa_model.training
        param_is_frozen = (
            (n, p.requires_grad)
            for n, p in self.iqa_model.named_parameters()
        )
        self.unfreeze() # Unfreeze all parameters
        self.iqa_model.train()

        yield self.iqa_model

        # Revert to the original
        for p, s in param_is_frozen: p.requires_grad = s

        if not is_training:
            self.iqa_model.eval()


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
            try:
                # Move nr to the tail
                nr_idx = feat_names.index('nr')
                weights = weights[:nr_idx]+weights[nr_idx+1:]+[weights[nr_idx]]
            except ValueError:
                pass
            finally:
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
        #tv_loss = self._calc_tv_loss(output)

        total_loss = self.alpha*pixel_loss + feat_loss# + 1e-3*tv_loss

        return total_loss, pixel_loss, feat_loss

    def _calc_feat_loss(self, output, target):
        loss = self.iqa_loss(output, target)
        return torch.sum(self.weights*loss)

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
