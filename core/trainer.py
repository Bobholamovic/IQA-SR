import shutil
import os

import torch
import torch.backends.cudnn as cudnn

from pdb import set_trace as db
from tqdm import tqdm
from skimage import io
from dataset.dataset import WaterlooDataset
from dataset.augmentation import Compose, Crop, Flip
from dataset.common import resize
from utils.metric import ShavedPSNR, ShavedSSIM, Metric
from utils.loss import ComLoss
from utils.misc import Logger
from models.sr_models.factory import build_model

from constants import ARCH


class Trainer:
    def __init__(self, settings):
        super(Trainer, self).__init__()
        self.settings = settings
        self.phase = settings.cmd
        self.batch_size = settings.batch_size
        self.num_workers = settings.num_workers
        self.data_dir = settings.data_dir
        self.list_dir = settings.list_dir
        self.checkpoint = settings.resume
        self.load_checkpoint = (len(self.checkpoint)>0)
        self.patch_size = settings.patch_size
        self.num_epochs = settings.num_epochs
        self.lr = settings.lr
        self.save = not settings.save_off
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
    def validate_epoch(self):
        raise NotImplementedError

    def train(self):
        cudnn.benchmark = True
        
        if self.load_checkpoint: self._resume_from_checkpoint()
        max_acc = self._init_max_acc
        best_epoch = self.start_epoch-1

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
            self.logger.show_nl("Current: {:.6f}({:03d})\tBest: {:.6f}({:03d})\t".format(
                                acc, epoch, max_acc, best_epoch))

            self._save_checkpoint(self.model.state_dict(), max_acc, epoch, is_best)
    
    def validate(self):
        if self.checkpoint: 
            if self._resume_from_checkpoint():
                self.model.cuda()
                self.criterion.cuda()
                self.validate_epoch(self.start_epoch-1, self.save)
        else:
            self.logger.warning("no checkpoint assigned!") 

    def _load_pretrained(self):
        raise NotImplementedError
        
    def _adjust_learning_rate(self, epoch):
        # Note that this does not take effect for separate learning rates
        if self.settings.lr_mode == 'step':
            lr = self.lr * (0.5 ** ((epoch-self.start_epoch) // self.settings.step))
        elif self.settings.lr_mode == 'poly':
            lr = self.lr * (1 - (epoch-self.start_epoch) / (self.num_epochs-self.start_epoch)) ** 1.1
        elif self.settings.lr_mode == 'const':
            return self.lr
        else:
            raise ValueError('unknown lr mode {}'.format(self.settings.lr_mode))

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
        ckp_dict = checkpoint['state_dict']
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
        else:
            self.start_epoch = checkpoint['epoch']
            self._init_max_acc = checkpoint['max_acc']
        
        state_dict.update(update_dict)
        self.model.load_state_dict(state_dict)

        self.logger.show("=> loaded checkpoint '{}' (epoch {})"
                .format(self.checkpoint, checkpoint['epoch']-1))
        return True
        
    def _save_checkpoint(self, state_dict, max_acc, epoch, is_best):
        state = {
            'epoch': epoch+1, 
            'state_dict': state_dict, 
            'max_acc': max_acc
        } 
        # Save history
        history_path = self.path('weight', 'checkpoint_{:03d}.pkl'.format(
                                epoch+1
                                ), underline=True)
        if (epoch-self.start_epoch) % self.settings.trace_freq == 0:
            torch.save(state, history_path) 
        # Save latest
        latest_path = self.path('weight', 'checkpoint_latest.pkl', underline=True)
        torch.save(state, latest_path)
        if is_best:
            shutil.copyfile(latest_path, self.path('weight', 'model_best.pkl', underline=True))
        
    
class SRTrainer(Trainer):
    def __init__(self, settings):
        super(SRTrainer, self).__init__(settings)
        self.scale = settings.scale
        self.criterion = ComLoss(
            settings.iqa_model_path, 
            settings.__dict__.get('weights'), 
            settings.__dict__.get('feat_names'), 
            settings.iqa_patch_size, 
            settings.criterion
        )

        self.model = build_model(ARCH, scale=self.scale)

        if self.phase == 'train':
            self.train_loader = torch.utils.data.DataLoader(
                WaterlooDataset(
                    self.data_dir, 'train', self.scale, 
                    list_dir=self.list_dir, 
                    transform=Compose(Crop(settings.patch_size), Flip())), 
                batch_size=self.batch_size, shuffle=True, 
                num_workers=self.num_workers, 
                pin_memory=True, drop_last=True
                )

        self.val_loader = WaterlooDataset(
            self.data_dir, 'val', 
            self.scale, 
            subset=settings.subset, list_dir=self.list_dir)
       
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          betas=(0.9,0.999), 
                                          lr=self.lr, 
                                          weight_decay=settings.weight_decay
                                         )

    def train_epoch(self):
        losses = Metric()
        pixel_loss = Metric()
        feat_loss = Metric()
        len_train = len(self.train_loader)
        pb = tqdm(enumerate(self.train_loader))
        
        self.model.train()
        # Make sure the criterion is also set to the correct state
        self.criterion.train()

        for i, (lr, hr) in pb:
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
        losses = Metric(self.criterion)
        ssim = ShavedSSIM(self.scale)
        psnr = ShavedPSNR(self.scale)
        interp = ShavedPSNR(self.scale) # For simple upsampling
        len_val = len(self.val_loader)
        to_image = self.val_loader.tensor_to_image
        pb = tqdm(enumerate(self.val_loader))

        self.model.eval()
        self.criterion.eval()

        with torch.no_grad():
            for i, (name, lr, hr) in pb:
                if self.phase == 'train' and i >= 32: 
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
                lr_int = resize(lr, (lr.shape[0]*self.scale, lr.shape[1]*self.scale))
                interp.update(lr_int, hr)

                pb.set_description("[{}/{}]"
                        "Loss {loss.val:.4f} ({loss.avg:.4f}) "
                        "PSNR {psnr.val:.4f} ({psnr.avg:.4f}) "
                        "SSIM {ssim.val:.4f} ({ssim.avg:.4f})"
                        .format(i+1, len_val, loss=losses,
                                    psnr=psnr, ssim=ssim))

                self.logger.dump("[{}/{}]"
                            "{} "
                            "Loss {loss.val:.4f} ({loss.avg:.4f}) "
                            "Interp {interp.val:.4f} ({interp.avg:.4f}) "
                            "PSNR {psnr.val:.4f} ({psnr.avg:.4f}) "
                            "SSIM {ssim.val:.4f} ({ssim.avg:.4f})"
                            .format(
                                i+1, len_val, name, 
                                loss=losses, interp=interp, 
                                psnr=psnr, ssim=ssim)
                            )
                
                if store:
                    lr_name = self.path_ctrl.add_suffix(name, suffix='lr', underline=True)
                    # int_name = self.path_ctrl.add_suffix(name, suffix='int', underline=True)
                    hr_name = self.path_ctrl.add_suffix(name, suffix='hr', underline=True)
                    sr_name = self.path_ctrl.add_suffix(name, suffix='sr', underline=True)

                    self.save_image(lr_name, lr, epoch)
                    # self.save_image(int_name, lr_int, epoch)
                    self.save_image(hr_name, hr, epoch)
                    self.save_image(sr_name, sr, epoch)

        return psnr.avg
        
    def save_image(self, file_name, image, epoch):
        file_path = os.path.join(
            'epoch_{}/'.format(epoch), 
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
