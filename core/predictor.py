import os

import torch

from tqdm import tqdm
from skimage import io
from dataset.dataset import get_dataset
from utils.misc import Logger
from models.sr_models import build_model

import constants


class Predictor:
    def __init__(
        self, data_dir, ckp_path, 
        save_lr=False, list_dir='', 
        out_dir='./', log_dir=''
    ):
        super(Predictor, self).__init__()
        self.data_dir = data_dir
        self.list_dir = list_dir
        self.out_dir = out_dir
        self.checkpoint = ckp_path
        self.save_lr = save_lr

        self.logger = Logger(
            scrn=True,
            log_dir=log_dir,
            phase='test'
        )

        self.model = None

    def test_epoch(self):
        raise NotImplementedError

    def test(self):
        if self.checkpoint:
            if self._resume_from_checkpoint():
                self.model.cuda()
                self.model.eval()
                self.test_epoch()
        else:
            self.logger.warning("no checkpoint assigned!")

    def _resume_from_checkpoint(self):
        if not os.path.isfile(self.checkpoint):
            self.logger.error("=> no checkpoint found at '{}'".format(self.checkpoint))
            return False

        self.logger.show("=> loading checkpoint '{}'".format(
                        self.checkpoint))
        checkpoint = torch.load(self.checkpoint)

        state_dict = self.model.state_dict()
        ckp_dict = checkpoint.get('state_dict', checkpoint)

        try:
            state_dict.update(ckp_dict)
            self.model.load_state_dict(ckp_dict)
        except KeyError as e:
            self.logger.error("=> mismatched checkpoint for test")
            self.logger.error(e)
            return False
        else:
            self.epoch = checkpoint.get('epoch', 0)

        self.logger.show("=> loaded checkpoint '{}'".format(self.checkpoint))
        return True
        
    
class SRPredictor(Predictor):
    def __init__(
        self, scale, data_dir, ckp_path, save_lr=False,
        list_dir='', out_dir='./', log_dir=''
    ):
        super(SRPredictor, self).__init__(
            data_dir=data_dir,
            ckp_path=ckp_path,
            save_lr=save_lr,
            list_dir=list_dir,
            out_dir=out_dir,
            log_dir=log_dir
        )
        self.scale = scale

        self.model = build_model(constants.ARCH, scale=self.scale)
        self.dataset = get_dataset(constants.DATASET)

        self.test_loader = self.dataset(
            self.data_dir, 'test', 
            self.scale, 
            list_dir=self.list_dir
        )

    def test_epoch(self):
        len_test = len(self.test_loader)
        pb = tqdm(self.test_loader)
        to_image = self.dataset.tensor_to_image

        with torch.no_grad():
            for i, (name, lr) in enumerate(pb):
                lr = lr.unsqueeze(0).cuda()

                sr = self.model(lr)

                lr = to_image(lr.squeeze(0), 'lr')
                sr = to_image(sr.squeeze(0))

                desc = "[Epoch {}] [{}/{}] {}".format(
                    self.epoch, i+1, len_test, name
                )
                pb.set_description(desc)

                self.logger.dump(desc)
                
                if self.save_lr:
                    self.save_image(name, lr)

                hr_name = self._hr_name_from_lr_name(name)
                self.save_image(hr_name, sr)
        
    def save_image(self, file_name, image):
        return io.imsave(os.path.join(self.out_dir, file_name), image)

    def _hr_name_from_lr_name(self, lr_name):
        name_, ext = os.path.splitext(lr_name)
        return '{:s}_x{:d}{:s}'.format(name_, self.scale, ext)
