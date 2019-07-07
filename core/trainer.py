import shutil
import os

import torch
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from skimage import io
from dataset.dataset import get_dataset
from dataset.augmentation import Compose, MSCrop, Flip
from utils.metric import ShavedPSNR, ShavedSSIM, Metric
from utils.loss import ComLoss
from utils.misc import Logger
from utils.MDSI import MDSI
from models.sr_models import build_model

from constants import ARCH, DATASET
from constants import (
    CKP_BEST, CKP_LATEST, CKP_COUNTED, 
    CKP_DISCR_BEST, CKP_DISCR_LATEST, CKP_DISCR_COUNTED
)


class Trainer:
    def __init__(self, settings):
        super(Trainer, self).__init__()
        self.settings = settings
        self.phase = settings.cmd
        self.batch_size = settings.batch_size
        self.data_dir = settings.data_dir
        self.list_dir = settings.list_dir
        self.checkpoint = settings.resume
        self.load_checkpoint = (len(self.checkpoint)>0)
        self.num_epochs = settings.num_epochs
        self.lr = float(settings.lr)
        self.save = settings.save_on or settings.out_dir
        self.path_ctrl = settings.global_path
        self.path = self.path_ctrl.get_path

        log_dir = '' if settings.log_off else self.path_ctrl.get_dir('log')
        self.logger = Logger(
            scrn=True, 
            log_dir=log_dir, 
            phase=self.phase
        )

        for k,v in sorted(settings.__dict__.items()): 
            self.logger.show("{}: {}".format(k,v))

        self.start_epoch = 0
        self._init_max_acc = 0.0

        self.model = None
        self.criterion = None

    def train_epoch(self):
        raise NotImplementedError

    def validate_epoch(self, epoch, store):
        raise NotImplementedError

    def train(self):
        cudnn.benchmark = True
        
        if self.load_checkpoint:
            self._resume_from_checkpoint()
        max_acc = self._init_max_acc
        best_epoch = self.get_ckp_epoch()

        self.model.cuda()
        self.criterion.cuda()
        
        for epoch in range(self.start_epoch, self.num_epochs):
            lr = self._adjust_learning_rate(epoch)
            
            self.logger.show_nl("Epoch: [{0}]\tlr {1:.06f}".format(epoch, lr))
            # Train for one epoch
            self.train_epoch()

            # Evaluate the model on validation set
            self.logger.show_nl("Validate")
            acc = self.validate_epoch(epoch=epoch, store=self.save)
            
            is_best = acc > max_acc
            if is_best:
                max_acc = acc
                best_epoch = epoch
            self.logger.show_nl("Current: {:.6f} ({:03d})\tBest: {:.6f} ({:03d})\t".format(
                                acc, epoch, max_acc, best_epoch))

            # The checkpoint saves next epoch
            self._save_checkpoint(self.model.state_dict(), max_acc, epoch+1, is_best)
    
    def validate(self):
        if self.checkpoint: 
            if self._resume_from_checkpoint():
                self.model.cuda()
                self.criterion.cuda()
                self.validate_epoch(self.get_ckp_epoch(), self.save)
        else:
            self.logger.warning("no checkpoint assigned!") 

    def _load_pretrained(self):
        raise NotImplementedError
        
    def _adjust_learning_rate(self, epoch):
        # Note that this does not take effect for separate learning rates
        if self.settings.lr_mode == 'step':
            lr = self.lr * (0.5 ** (epoch // self.settings.step))
        elif self.settings.lr_mode == 'poly':
            lr = self.lr * (1 - epoch / self.num_epochs) ** 1.1
        elif self.settings.lr_mode == 'const':
            lr = self.lr
        else:
            raise ValueError('unknown lr mode {}'.format(self.settings.lr_mode))

        if lr == self.lr:
            return self.lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def _resume_from_checkpoint(self):
        if not os.path.isfile(self.checkpoint):
            self.logger.error("=> no checkpoint found at '{}'".format(self.checkpoint))
            return False

        self.logger.show("=> loading checkpoint '{}'".format(
                        self.checkpoint))
        checkpoint = torch.load(self.checkpoint)

        state_dict = self.model.state_dict()
        ckp_dict = checkpoint.get('state_dict', checkpoint)
        update_dict = {k:v for k,v in ckp_dict.items() 
            if k in state_dict and state_dict[k].shape == v.shape}
        
        num_to_update = len(update_dict)
        if (num_to_update < len(state_dict)) or (len(state_dict) < len(ckp_dict)):
            if self.phase == 'val' and (num_to_update < len(state_dict)):
                self.logger.error("=> mismatched checkpoint for validation")
                return False
            self.logger.warning("warning: trying to load an mismatched checkpoint")
            if num_to_update == 0:
                self.logger.error("=> no parameter is to be loaded")
                return False
            else:
                self.logger.warning("=> {} params are to be loaded".format(num_to_update))
        elif (not self.settings.anew) or (self.phase != 'train'):
            # Note in the non-anew mode, it is not guaranteed that the contained field 
            # max_acc be the corresponding one of the loaded checkpoint.
            self.start_epoch = checkpoint.get('epoch', self.start_epoch)
            self._init_max_acc = checkpoint.get('max_acc', self._init_max_acc)

        state_dict.update(update_dict)
        self.model.load_state_dict(state_dict)

        self.logger.show("=> loaded checkpoint '{}' (epoch {}, max_acc {:.4f})".format(
            self.checkpoint, self.get_ckp_epoch(), self._init_max_acc
            ))
        return True
        
    def _save_checkpoint(self, state_dict, max_acc, epoch, is_best):
        state = {
            'epoch': epoch,
            'state_dict': state_dict,
            'max_acc': max_acc
        } 
        # Save history
        history_path = self.path('weight', CKP_COUNTED.format(
                                e=epoch, s=self.scale
                                ), underline=True)
        if epoch % self.settings.trace_freq == 0:
            torch.save(state, history_path)
        # Save latest
        latest_path = self.path(
            'weight', CKP_LATEST.format(s=self.scale), 
            underline=True
        )
        torch.save(state, latest_path)
        if is_best:
            shutil.copyfile(
                latest_path, self.path(
                    'weight', CKP_BEST.format(s=self.scale), 
                    underline=True
                )
            )
    
    def get_ckp_epoch(self):
        # Get current epoch of the checkpoint
        # For dismatched ckp or no ckp, set to 0
        return max(self.start_epoch-1, 0)
        
    
class SRTrainer(Trainer):
    def __init__(self, settings):
        super(SRTrainer, self).__init__(settings)
        self.scale = settings.scale
        self.criterion = ComLoss(
            settings.iqa_model_path, 
            settings.__dict__.get('weights'), 
            settings.__dict__.get('feat_names'), 
            settings.alpha, 
            settings.iqa_patch_size, 
            settings.criterion
        )
        if hasattr(self.criterion, 'iqa_loss'):
            # For saving cost
            self.criterion.iqa_loss.freeze()

        self.model = build_model(ARCH, scale=self.scale)
        self.dataset = get_dataset(DATASET)

        if self.phase == 'train':
            self.train_loader = torch.utils.data.DataLoader(
                self.dataset(
                    self.data_dir, 'train', self.scale, 
                    list_dir=self.list_dir, 
                    transform=Compose(
                        MSCrop(self.scale, settings.patch_size), 
                        Flip()
                    ), 
                    repeats=settings.reproduce), 
                batch_size=self.batch_size, #max(self.batch_size//settings.reproduce, 1),
                shuffle=True, 
                num_workers=settings.num_workers, 
                pin_memory=True, drop_last=True
                )

        self.val_loader = self.dataset(
            self.data_dir, 'val', 
            self.scale, 
            subset=settings.subset, 
            list_dir=self.list_dir
        )
            
        if not self.val_loader.lr_avai:
            self.logger.warning("warning: the low-resolution sources are not available")

        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          betas=(0.9,0.999), 
                                          lr=self.lr, 
                                          weight_decay=settings.weight_decay
                                         )
        # self.optimizer = torch.optim.RMSprop(
        #     self.model.parameters(),
        #     lr=self.lr,
        #     alpha=0.9,
        #     weight_decay=settings.weight_decay
        # )

        self.logger.dump(self.model)    # Log the architecture

    def train_epoch(self):
        losses = Metric()
        pixel_loss = Metric()
        feat_loss = Metric()
        len_train = len(self.train_loader)
        pb = tqdm(self.train_loader)
        
        self.model.train()
        # Make sure the criterion is also set to the correct state
        self.criterion.train()
        self.criterion.iqa_loss.eval()

        for i, (lr, hr) in enumerate(pb):
            # Note that the lr here means low-resolution (images)
            # rather than learning rate
            lr, hr = lr.cuda(), hr.cuda()
            sr = self.model(lr)
            
            loss, pl, fl = self.criterion(sr, hr)

            losses.update(loss.data, n=self.batch_size)
            pixel_loss.update(pl.data, n=self.batch_size)
            feat_loss.update(fl.data, n=self.batch_size)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            desc = "[{}/{}] Loss {loss.val:.4f} ({loss.avg:.4f}) " \
                    "PL {pixel.val:.4f} ({pixel.avg:.4f}) " \
                    "FL {feat.val:.6f} ({feat.avg:.6f})"\
                .format(i+1, len_train, loss=losses, 
                    pixel=pixel_loss, feat=feat_loss)
            pb.set_description(desc)
            self.logger.dump(desc)

    def validate_epoch(self, epoch=0, store=False):
        self.logger.show_nl("Epoch: [{0}]".format(epoch))
        losses = Metric(self.criterion)
        ssim = ShavedSSIM(self.scale)
        psnr = ShavedPSNR(self.scale)
        len_val = len(self.val_loader)
        pb = tqdm(self.val_loader)
        to_image = self.dataset.tensor_to_image

        self.model.eval()
        self.criterion.eval()

        with torch.no_grad():
            for i, (name, lr, hr) in enumerate(pb):
                if self.phase == 'train' and i >= 16: 
                    # Do not validate all images on training phase
                    pb.close()
                    self.logger.warning("validation ends early")
                    break
                    
                lr, hr = lr.unsqueeze(0).cuda(), hr.unsqueeze(0).cuda()

                sr = self.model(lr)

                losses.update(sr, hr)

                lr = to_image(lr.squeeze(0).cpu(), 'lr')
                sr = to_image(sr.squeeze(0).cpu())
                hr = to_image(hr.squeeze(0).cpu())

                psnr.update(sr, hr)
                ssim.update(sr, hr)

                pb.set_description("[{}/{}]"
                        "Loss {loss.val:.4f} ({loss.avg:.4f}) "
                        "PSNR {psnr.val:.4f} ({psnr.avg:.4f}) "
                        "SSIM {ssim.val:.4f} ({ssim.avg:.4f})"
                        .format(i+1, len_val, loss=losses,
                                    psnr=psnr, ssim=ssim))

                self.logger.dump("[{}/{}]"
                            "{} "
                            "Loss {loss.val:.4f} ({loss.avg:.4f}) "
                            "PSNR {psnr.val:.4f} ({psnr.avg:.4f}) "
                            "SSIM {ssim.val:.4f} ({ssim.avg:.4f})"
                            .format(
                                i+1, len_val, name,
                                loss=losses,
                                psnr=psnr, ssim=ssim)
                            )
                
                if store:
                    # lr_name = self.path_ctrl.add_suffix(name, suffix='lr', underline=True)
                    # hr_name = self.path_ctrl.add_suffix(name, suffix='hr', underline=True)
                    sr_name = self.path_ctrl.add_suffix(name, suffix='sr', underline=True)

                    # self.save_image(lr_name, lr, epoch)
                    # self.save_image(hr_name, hr, epoch)
                    self.save_image(sr_name, sr, epoch)

        return psnr.avg
        
    def save_image(self, file_name, image, epoch):
        file_path = os.path.join(
            'x{}/epoch_{}/'.format(self.scale, epoch),
            self.settings.out_dir,
            file_name
        )
        out_path = self.path(
            'out', file_path,
            suffix=not self.settings.suffix_off,
            auto_make=True,
            underline=True
        )
        return io.imsave(out_path, image)


class JointTrainer(SRTrainer):
    def __init__(self, settings):
        assert hasattr(settings, 'weights')
        super().__init__(settings)
        self.assessor = self.criterion.iqa_loss
        self.iqa_optim = torch.optim.Adam(
            self.assessor.parameters(), 
            lr=1e-4, 
            weight_decay=0.0
        )
        self.assessor.freeze()

    def train_epoch(self):
        losses = Metric()
        pixel_loss = Metric()
        feat_loss = Metric()
        iqa_loss = Metric()
        len_train = len(self.train_loader)
        pb = tqdm(self.train_loader)
        
        self.model.train()
        # Make sure the criterion is also set to the correct state
        self.criterion.train()

        for i, (lr, hr) in enumerate(pb):
            # Note that the lr here means low-resolution (images)
            # rather than learning rate
            lr, hr = lr.cuda(), hr.cuda()
            sr = self.model(lr)
                
            sr_norm = self.assessor.renormalize(sr.detach())
            hr_norm = self.assessor.renormalize(hr)

            if not hasattr(self, '_margin'):
                self._margin = MDSI(sr_norm*255.0, hr_norm*255.0)
            else:
                beta = 0.99
                m = MDSI(sr_norm*255.0, hr_norm*255.0)
                self._margin = beta*self._margin + (1-beta)*m

            if i % 200 < 100:
                # Train the IQA model
                with self.assessor.learner():
                    out_lq = self.assessor.iqa_model(sr_norm)
                    out_hq = self.assessor.iqa_model(hr_norm)
                    ql = torch.nn.functional.relu(out_lq - out_hq + self._margin).mean()
                    iqa_loss.update(ql.data, n=self.batch_size)

                    self.iqa_optim.zero_grad()
                    ql.backward()
                    self.iqa_optim.step()
            else:
                del sr_norm, hr_norm
                # Train the SR model
                loss, pl, fl = self.criterion(sr, hr)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update data
                losses.update(loss.data, n=self.batch_size)
                pixel_loss.update(pl.data, n=self.batch_size)
                feat_loss.update(fl.data, n=self.batch_size)

            # Log for this mini-batch
            desc = "[{}/{}] Loss {loss.val:.4f} ({loss.avg:.4f}) " \
                    "QL {iqa.val:.6f} ({iqa.avg:.6f}) " \
                    "PL {pixel.val:.4f} ({pixel.avg:.4f}) " \
                    "FL {feat.val:.4f} ({feat.avg:.4f})"\
                .format(i+1, len_train, loss=losses, 
                        iqa=iqa_loss, 
                        pixel=pixel_loss, feat=feat_loss)
            pb.set_description(desc)
            self.logger.dump(desc)

    def rank_learn(self, im_sr, im_hr):
        out_lq = self.assessor.iqa_forward(im_sr).mean(-1)
        out_hq = self.assessor.iqa_forward(im_hr).mean(-1)

        # Ranking hinge loss
        margin = 0.5
        loss = torch.nn.functional.relu(out_lq - out_hq + margin).mean()

        self.iqa_optim.zero_grad()
        loss.backward()
        self.iqa_optim.step()

        return loss.data    # Since the gradients are no longer needed

    def _save_checkpoint(self, state_dict, max_acc, epoch, is_best):
        # Save the generator checkpoint first
        super()._save_checkpoint(state_dict, max_acc, epoch, is_best)
        # Save latest quality assessor
        state = {
            'epoch': epoch,
            'state_dict': self.assessor.state_dict()
        } 
        # Save history
        history_path = self.path('weight', CKP_DISCR_COUNTED.format(
                                e=epoch, s=self.scale
                                ), underline=True)
        if epoch % self.settings.trace_freq == 0:
            torch.save(state, history_path)
        # Save latest
        latest_path = self.path(
            'weight', CKP_DISCR_LATEST.format(s=self.scale), 
            underline=True
        )
        torch.save(state, latest_path)
        if is_best:
            shutil.copyfile(
                latest_path, self.path(
                    'weight', CKP_DISCR_BEST.format(s=self.scale), 
                    underline=True
                )
            )
