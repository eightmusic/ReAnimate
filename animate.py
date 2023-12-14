import argparse
import datetime
import inspect
import os
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from collections import OrderedDict
import torch

from diffusers import AutoencoderKL, DDIMScheduler, UniPCMultistepScheduler
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel

from model.ref_model import AppearanceEncoderModel
from models.atten import ReferenceAttentionControl

class Re():
    def __init__(self, config="configs/prompts/animation.yaml"):
        config = OmegaConf.load(config)

        self.unet = UNet2DConditionModel.from_pretrained()
        self.ref_net= AppearanceEncoderModel.from_pretrained()
        self.reference_control_writer = ReferenceAttentionControl(self.appearance_encoder, do_classifier_free_guidance=True, mode='write', fusion_blocks=config.fusion_blocks)
        self.reference_control_reader = ReferenceAttentionControl(self.unet, do_classifier_free_guidance=True, mode='read', fusion_blocks=config.fusion_blocks)
        self.vae = AutoencoderKL.from_pretrained(config.pretrained_model_path, subfolder="vae")
        self.pose_net = ''
        self.pipeline = ''