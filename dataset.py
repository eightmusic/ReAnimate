import json
import cv2
import numpy as np
import os
import random
from pathlib import Path
from einops import rearrange

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, path, tokenizer, img_size=256, mode='train'):
        super().__init__()
        self.mode = mode
        self.tokenizer = tokenizer

        self.hint_file = []
        self.source_file = []
        self.target_file = []

        self.img_size = img_size
        for root, _, _ in os.walk(path):
            if root.endswith('target'):
                for img in Path(root).glob('*.png'):
                    self.target_file.append(str(img))

    def __len__(self):
        return len(self.target_file)

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.getitem_train(idx)
        else:
            return self.getitem_val(idx)

    def padimg(self, img):
        img = img[:, :, ::-1]
        h, w = img.shape[:2]
        if h > w:
            left = (h - w) // 2
            right = h - w - left
            img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        else:
            top = (w - h) // 2
            bottom = (w - h) - top
            img = cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        return img

    def getitem_train(self, idx):
        item = self.target_file[idx]

        dir_file = os.path.dirname(os.path.dirname(item))
        source_file = os.path.join(dir_file, 'source')
        hint_file = os.path.join(dir_file, 'hint')

        source_img = []
        for img_path in os.scandir(source_file):
            # temp_ = cv2.imread(img_path.path)
            # temp_ = self.padimg(temp_)
            # temp = cv2.resize(temp_, (self.img_size, self.img_size))
            source_img.append(img_path.path)
        source_img = cv2.resize(self.padimg(cv2.imread(random.choice(source_img))), (self.img_size, self.img_size))

        target_img_ = cv2.imread(item)
        target_img_ = self.padimg(target_img_)
        target_img = cv2.resize(target_img_, (self.img_size, self.img_size))

        hint_name = os.path.join(hint_file, os.path.basename(item))
        if not os.path.exists(hint_name):
            hint_name = hint_name.replace('.png', '_pose.png')
        hint_img_ = cv2.imread(hint_name)
        hint_img_ = self.padimg(hint_img_)
        hint_img = cv2.resize(hint_img_, (self.img_size, self.img_size))

        # Do not forget that OpenCV read images in BGR order.
        hint_img = cv2.cvtColor(hint_img, cv2.COLOR_BGR2RGB)

        # random rgb
        rgb = [0, 1, 2]
        random.shuffle(rgb)
        target_img = target_img[:, :, rgb]
        source_img = source_img[:, :, rgb]
        # source_img = [i[:, :, rgb] for i in source_img]
        # source_img = np.stack(source_img, 0)

        # Normalize source images to [0, 1].
        hint_img = hint_img.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target_img = (target_img.astype(np.float32) / 127.5) - 1.0

        source_img = (source_img.astype(np.float32) / 127.5) - 1.0
        target_img = rearrange(target_img, 'h w c->c h w')
        hint_img = rearrange(hint_img, 'h w c->c h w')
        source_img = rearrange(source_img, 'h w c->c h w')
        # source_img = rearrange(source_img, 'n h w c->n c h w')

        # txt_id = self.tokenizer("", return_tensors="pt").input_ids
        txt_id = self.tokenizer("", return_tensors="pt").input_ids[0]
        return dict(jpg=target_img, hint=hint_img, source=source_img,
                    txt=txt_id)

    def getitem_val(self, idx):
        item = self.target_file[idx]

        dir_file = os.path.dirname(os.path.dirname(item))
        source_file = os.path.join(dir_file, 'source')
        hint_file = os.path.join(dir_file, 'hint')

        source_img = []
        for img_path in os.scandir(source_file):
            # temp_ = cv2.imread(img_path.path)
            # temp_ = self.padimg(temp_)
            # temp = cv2.resize(temp_, (self.img_size, self.img_size))
            source_img.append(img_path.path)
        source_img = cv2.resize(self.padimg(cv2.imread(random.choice(source_img))), (self.img_size, self.img_size))

        target_img_ = cv2.imread(item)
        target_img_ = self.padimg(target_img_)
        target_img = cv2.resize(target_img_, (self.img_size, self.img_size))

        hint_name = os.path.join(hint_file, os.path.basename(item))
        if not os.path.exists(hint_name):
            hint_name = hint_name.replace('.png', '_pose.png')
        hint_img_ = cv2.imread(hint_name)
        hint_img_ = self.padimg(hint_img_)
        hint_img = cv2.resize(hint_img_, (self.img_size, self.img_size))

        # Do not forget that OpenCV read images in BGR order.
        hint_img = cv2.cvtColor(hint_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        # source_img = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in source_img]
        # source_img = np.stack(source_img, 0)

        # Normalize source images to [0, 1].
        hint_img = hint_img.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target_img = (target_img.astype(np.float32) / 127.5) - 1.0

        source_img = (source_img.astype(np.float32) / 127.5) - 1.0
        target_img = rearrange(target_img, 'h w c->c h w')
        hint_img = rearrange(hint_img, 'h w c->c h w')
        source_img = rearrange(source_img, 'h w c->c h w')
        # source_img = rearrange(source_img, 'n h w c->n c h w')

        # txt_id = self.tokenizer("", return_tensors="pt")['input_ids'].long()
        txt_id = self.tokenizer("", return_tensors="pt").input_ids[0]
        return dict(jpg=target_img, hint=hint_img, source=source_img,
                    txt=txt_id)
