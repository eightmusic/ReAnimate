import os
import argparse
import random
import inspect
from tqdm import tqdm
import math
import shutil
import cv2
import torch
from collections import OrderedDict
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from einops import rearrange
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator
from accelerate.logging import get_logger
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, PeftModel

from models.reference_model import AppearanceEncoderModel, Unet
from models.atten import ReferenceAttentionControl
from models.model import PoseFeat

from pipeline import AnimationPipeline
from train import save_val_image


def log_validation():
    fusion_blocks ='full'
    pretrained_model_name_or_path = '/home/zs/disk/my/magic-animate-main/pretrained_models/stable-diffusion-v1-5'
    weight_dtype = torch.float16
    device = 'cuda'
    seed = 0


    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    appearance_encoder = AppearanceEncoderModel().from_pretrained('/home/zs/disk/my/magic-animate-main/pretrained_models/MagicAnimate/appearance_encoder')
    unet = Unet()#.from_pretrained('/home/zs/disk/my/magic-animate-main/pretrained_models/stable-diffusion-v1-5/unet')
    vae = AutoencoderKL.from_pretrained('/home/zs/disk/my/magic-animate-main/pretrained_models/sd-vae-ft-mse')
    controlnet = PoseFeat(3, 320)
    control_state_dict = torch.load('/home/zs/disk1/project/ani/weights/control.ckpt', map_location="cpu")
    missing, unexpected = controlnet.load_state_dict(control_state_dict, strict=False)
    print(len(missing),len(unexpected))

    appearance_encoder_dict = torch.load('/home/zs/disk1/project/ani/weights/appearance_encoder/diffusion_pytorch_model.ckpt', map_location="cpu")
    state_dict = OrderedDict()
    for name, p in appearance_encoder_dict.items():
        if name.startswith('base_model'):
            state_dict[".".join(name.split('.')[2:])] = p

    missing, unexpected = appearance_encoder.load_state_dict(state_dict, strict=False)
    print(len(missing), len(unexpected))
    import pdb
    pdb.set_trace()

    unet_dict = torch.load('/home/zs/disk1/project/ani/weights/unet/diffusion_pytorch_model.ckpt', map_location="cpu")
    missing, unexpected = unet.load_state_dict(unet_dict, strict=False)
    print(len(missing), len(unexpected))


    pipeline = AnimationPipeline.from_pretrained(
        pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )
    pipeline.unet.forward = Unet.forward.__get__(pipeline.unet, Unet)
    reference_control_writer = ReferenceAttentionControl(appearance_encoder, do_classifier_free_guidance=True,
                                                         mode='write', fusion_blocks=fusion_blocks)
    reference_control_reader = ReferenceAttentionControl(unet, do_classifier_free_guidance=True, mode='read',
                                                         fusion_blocks=fusion_blocks)

    # pipeline.unet.forward = Unet.forward.__get__(pipeline.unet, Unet)

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)

    pipeline.set_progress_bar_config(disable=True)


    if seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(seed)

    # if len(args.validation_image) == len(args.validation_prompt):
    #     validation_images = args.validation_image
    #     validation_prompts = args.validation_prompt
    # elif len(args.validation_image) == 1:
    #     validation_images = args.validation_image * len(args.validation_prompt)
    #     validation_prompts = args.validation_prompt
    # elif len(args.validation_prompt) == 1:
    #     validation_images = args.validation_image
    #     validation_prompts = args.validation_prompt * len(args.validation_image)
    # else:
    #     raise ValueError(
    #         "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
    #     )
    image_logs = []
    prompt = ''
    control_file = '/home/zs/disk1/project/cn/data/enddata/one/1'
    control_list = os.listdir(os.path.join(control_file, 'hint'))
    control_list = random.sample(control_list, 1)
    control_image = []
    source_image = []
    target_image = []
    img_shape = (128, 128)
    for i in control_list:
        img = cv2.imread(os.path.join(control_file, 'hint', i))
        img = img[:, :, ::-1]
        img = cv2.resize(img, img_shape)
        control_image.append(img)
        img_t = cv2.imread(os.path.join(control_file, 'target', i))
        img_t = img_t[:, :, ::-1]
        img_t = cv2.resize(img_t, img_shape)
        target_image.append(img_t)

    for i in os.scandir(os.path.join(control_file, 'source')):
        img = cv2.imread(os.path.join(control_file, 'hint', i))
        img = img[:, :, ::-1]
        img = cv2.resize(img, img_shape)
        source_image.append(img)
    appearance_encoder = appearance_encoder.cuda()
    for control, target in zip(control_image, target_image):
        with torch.autocast("cuda"):
            image = pipeline(
                prompt='',
                num_inference_steps=5,
                controlnet_condition=np.expand_dims(control, axis=0),
                generator=generator,
                appearance_encoder=appearance_encoder,
                reference_control_writer=reference_control_writer,
                reference_control_reader=reference_control_reader,
                source_image=[source_image[0]],
                video_length=1,
                height=img_shape[0],
                width=img_shape[1],
            ).videos
        target = target / 255.0
        control = control / 255.0
        image_logs.append(
            {"original": target, "pose": control, "reconstruction": image[0, :, 0]}
        )
    save_val_image(image_logs, f'val_log/inf.jpg')

# log_validation()
appearance_encoder = AppearanceEncoderModel().from_pretrained('/home/zs/disk/my/magic-animate-main/pretrained_models/MagicAnimate/appearance_encoder')

appearance_encoder = PeftModel.from_pretrained(appearance_encoder,'/home/zs/disk1/project/ani/weights/appearance_encoder/diffusion_pytorch_model.ckpt')
appearance_encoder = appearance_encoder.merge_and_unload()

# appearance_encoder_dict = torch.load(
#     '/home/zs/disk1/project/ani/weights/appearance_encoder/diffusion_pytorch_model.ckpt', map_location="cpu")
# print(len(appearance_encoder_dict),"+")
# state_dict = OrderedDict()
# for name, p in appearance_encoder_dict.items():
#     if name.startswith('base_model'):
#         state_dict[".".join(name.split('.')[2:])] = p
# missing, unexpected = appearance_encoder.load_state_dict(state_dict, strict=False)
# print(len(missing), len(unexpected))
import pdb

pdb.set_trace()