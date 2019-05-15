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
from utils.ms_ssim import MS_SSIM
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
        self.lr = settings.lr
        self.save = settings.save_on or settings.out_dir
        self.from_pause = self.settings.continu
        self.path_ctrl = settings.global_path
        self.path = self.path_ctrl.get_path

        log_dir = '' if settings.log_off else self.path_ctrl.get_dir('log')
        self.logger = Logger(
            scrn=True, 
            log_dir=log_dir, 
            phase=self.phase
        )

        for k,v in settings.__dict__.items(): 
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
        
        end_epoch = self.num_epochs if self.from_pause else self.start_epoch+self.num_epochs
        for epoch in range(self.start_epoch, end_epoch):
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
        start_epoch = 0 if self.from_pause else self.start_epoch
        if self.settings.lr_mode == 'step':
            lr = self.lr * (0.5 ** ((epoch-start_epoch) // self.settings.step))
        elif self.settings.lr_mode == 'poly':
            lr = self.lr * (1 - (epoch-start_epoch) / (self.num_epochs-start_epoch)) ** 1.1
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
        if len(state_dict) != num_to_update:
            if self.phase == 'val':
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
        if (epoch-self.start_epoch) % self.settings.trace_freq == 0:
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

                lr = to_image(lr.squeeze(0), 'lr')
                sr = to_image(sr.squeeze(0))
                hr = to_image(hr.squeeze(0))

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


class GANTrainer(SRTrainer):
    def __init__(self, settings):
        assert hasattr(settings, 'weights')
        super().__init__(settings)
        self.discriminator = self.criterion.iqa_loss
        ## Hard coding here
        self.discr_optim = torch.optim.Adam(
            self.discriminator.parameters(), 
            betas=(0.9, 0.999), 
            lr=1e-4, 
            weight_decay=1e-4
        )
        self.discr_critn = MS_SSIM(max_val=1.0)
        # The discriminator criterion does not require gradients
        for p in self.discr_critn.parameters():
            p.requires_grad = False

    def train_epoch(self):
        losses = Metric()
        pixel_loss = Metric()
        feat_loss = Metric()
        discr_loss = Metric()
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
            
            if i % 1 == 0:
                with self.criterion.iqa_loss.learner():
                    # Train the IQA model
                    dl = self.discr_learn(hr, hr, 0.0)   # Good-quality images
                    dl += self.discr_learn(sr.detach(), hr) # Bad-quality images
                    dl /= 2
                    discr_loss.update(dl, n=self.batch_size)

            # Train the SR model
            loss, pl, fl = self.criterion(sr, hr)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()

            # Update data
            losses.update(loss.data, n=self.batch_size)
            pixel_loss.update(pl.data, n=self.batch_size)
            feat_loss.update(fl.data, n=self.batch_size)

            # Log for this mini-batch
            desc = "[{}/{}] Loss {loss.val:.4f} ({loss.avg:.4f}) " \
                    "DL {discr.val:.4f} ({discr.avg:.4f}) " \
                    "PL {pixel.val:.4f} ({pixel.avg:.4f}) " \
                    "FL {feat.val:.6f} ({feat.avg:.6f})"\
                .format(i+1, len_train, loss=losses, 
                        discr=discr_loss, 
                        pixel=pixel_loss, feat=feat_loss)
            pb.set_description(desc)
            self.logger.dump(desc)

    def discr_learn(self, output, target, score=None):
        score_o = self.discriminator.iqa_forward(output, target).score
        bs = target.size(0)  # Batch size
        if score is not None:
            score_t = torch.FloatTensor(score) if isinstance(score, tuple) \
                else torch.FloatTensor([score]*bs)
            score_t = score_t.type_as(target)
        else:
            output = self.discriminator.renormalize(output.data)
            target = self.discriminator.renormalize(target.data)
            # # Ensure that the criterion is on the eval mode
            # self.discr_critn.eval()

            # This looks weird and I wonder if there's something like 
            # apply_along_axis in pytorch to handle this loop
            chunks = zip(
                torch.chunk(output, bs, dim=0), 
                torch.chunk(target, bs, dim=0)
            )
            score_t = torch.stack([
                (1.0-self.discr_critn(o, t))*100
                for o, t 
                in chunks
            ])

        assert score_o.shape == score_t.shape
        loss = torch.nn.functional.l1_loss(score_o, score_t)

        self.discr_optim.zero_grad()
        loss.backward()
        self.discr_optim.step()

        return loss.data    # Since the gradients are no longer needed

    def _save_checkpoint(self, state_dict, max_acc, epoch, is_best):
        # Save the generator checkpoint first
        super()._save_checkpoint(state_dict, max_acc, epoch, is_best)
        # Save latest discriminator checkpoint
        state = {
            'epoch': epoch,
            'state_dict': self.discriminator.state_dict()
        } 
        # Save history
        history_path = self.path('weight', CKP_DISCR_COUNTED.format(
                                e=epoch, s=self.scale
                                ), underline=True)
        if (epoch-self.start_epoch) % self.settings.trace_freq == 0:
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

class MTTrainer(Trainer):
    # Mult-task trainer
    modes = ('IQA', 'SR', 'MT')
    def __init__(self, settings, mode='MT'):
        super(MTTrainer, self).__init__(settings)
        assert mode in MTTrainer.modes
        self.mode = mode
        self.scale = settings.scale
        self.sr_critn = torch.nn.L1Loss()
        self.iqa_critn = torch.nn.L1Loss()

        self.model = build_model('MT', scale=self.scale)
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
                batch_size=self.batch_size,
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

        self.sr_optim = torch.optim.Adam(self.model.sr_branch.parameters(), 
                                          betas=(0.9,0.999), 
                                          lr=self.lr, 
                                          weight_decay=settings.weight_decay
                                         )
        self.iqa_optim = torch.optim.Adam(self.model.iqa_branch.parameters(), 
                                          betas=(0.9,0.999), 
                                          lr=self.lr, 
                                          weight_decay=settings.weight_decay
                                         )
        self.iqa_metric = MS_SSIM(max_val=1.0)

        self.logger.dump(self.model)    # Log the architecture

        # If training a single branch, freeze the other
        if self.mode == 'IQA':
            for p in self.model.sr_branch.parameters():
                p.requires_grad = False
        elif self.mode == 'SR':
            for p in self.model.iqa_branch.parameters():
                p.requires_grad = False

    def train_epoch(self):
        # losses = Metric()
        sr_losses = Metric()
        iqa_losses = Metric()
        len_train = len(self.train_loader)
        pb = tqdm(self.train_loader)
        
        self.model.train()

        for i, (lr, hr) in enumerate(pb):
            lr, hr = lr.cuda(), hr.cuda()
            sr, score = self.model(lr)
            
            score_gt = self.gauge_quality_map(sr.detach(), hr)

            sr_loss = self.sr_critn(sr, hr)
            iqa_loss = self.iqa_critn(score, score_gt)

            # losses.update(loss.data, n=self.batch_size)
            sr_losses.update(sr_loss.data, n=self.batch_size)
            iqa_losses.update(iqa_loss.data, n=self.batch_size)
            

            if self.mode != 'IQA':
                self.sr_optim.zero_grad()
                sr_loss.backward()
                self.sr_optim.step()

            if self.mode != 'SR':
                self.iqa_optim.zero_grad()
                iqa_loss.backward()
                self.iqa_optim.step()

            desc = # "[{}/{}] Loss {loss.val:.4f} ({loss.avg:.4f}) " \
                    "SR Loss {sr_loss.val:.4f} ({sr_loss.avg:.4f}) " \
                    "IQA Loss {iqa_loss.val:.4f} ({iqa_loss.avg:.4f})"\
                .format(i+1, len_train, # loss=losses, 
                    sr_loss=sr_loss, iqa_loss=iqa_loss)
            pb.set_description(desc)
            self.logger.dump(desc)

    def validate_epoch(self, epoch=0, store=False):
        self.logger.show_nl("Epoch: [{0}]".format(epoch))
        # losses = Metric()
        sr_losses = Metric()
        iqa_losses = Metric()
        ssim = ShavedSSIM(self.scale)
        psnr = ShavedPSNR(self.scale)
        len_val = len(self.val_loader)
        pb = tqdm(self.val_loader)
        to_image = self.dataset.tensor_to_image

        self.model.eval()

        with torch.no_grad():
            for i, (name, lr, hr) in enumerate(pb):
                if self.phase == 'train' and i >= 16: 
                    # Do not validate all images on training phase
                    pb.close()
                    self.logger.warning("validation ends early")
                    break
                    
                lr, hr = lr.unsqueeze(0).cuda(), hr.unsqueeze(0).cuda()

                sr, score = self.model(lr)
                
                score_gt = self.gauge_iqa_score(sr, hr)

                sr_loss = self.sr_critn(sr, hr)
                iqa_loss = self.iqa_critn(score, score_gt)

                # losses.update(loss)
                sr_losses.update(sr_loss)
                iqa_losses.update(iqa_loss)

                lr = to_image(lr.squeeze(0), 'lr')
                sr = to_image(sr.squeeze(0))
                hr = to_image(hr.squeeze(0))

                psnr.update(sr, hr)
                ssim.update(sr, hr)

                desc = # "[{}/{}] Loss {loss.val:.4f} ({loss.avg:.4f}) " \
                        "SR Loss {sr_loss.val:.4f} ({sr_loss.avg:.4f}) " \
                        "IQA Loss {iqa_loss.val:.4f} ({iqa_loss.avg:.4f})"\
                        "PSNR {psnr.val:.4f} ({psnr.avg:.4f}) " \
                        "SSIM {ssim.val:.4f} ({ssim.avg:.4f})" \
                        .format(i+1, len_val, # loss=losses,
                                sr_loss=sr_loss, 
                                iqa_loss=iqa_loss,
                                psnr=psnr, ssim=ssim)

                pb.set_description(desc)

                self.logger.dump(desc)
                
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

    def gauge_quality_map(self, x1, x2):
        return self.iqa_metric._ssim(x1, x2, size_average=False)