#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 17:45:28 2022

@author: user
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import time
import copy
import yaml
import utils.util as ut
from diffusion_diffwave import DDPM
import os
import re

#specifically for diffusion
from diffusers import DDPMScheduler, ScoreSdeVeScheduler
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from torch.utils.data import DataLoader
from gomin.models import GomiGAN
from gomin.config import GANConfig
import torchaudio.transforms as transforms
import utils.bands_transform as bt

class ParameterCountCallback:
    def __init__(self, model):
        self.model = model
        self.updated_params = 0

    def on_batch_end(self, batch, logs=None):
        self.updated_params = 0
        for param in self.model.parameters():
            if param.grad is not None:
                self.updated_params += torch.sum(param.grad != 0).item()
        print(f"Iteration {batch}: Updated {self.updated_params} parameters")

class DiffusionTrainer:
    def __init__(self, setting_data, model, models_path, method, model_name, train_dataset=None, 
                 valid_dataset=None, eval_dataset=None, learning_rate=1e-3, gradient_accumulation_steps=1,
                 lr_warmup_steps=500, diff_steps=1000, schedule='VE', model_chkpt_name=None, model_chkpt_setting_name=None,
                 dtype=torch.FloatTensor, 
                 ltype=torch.LongTensor):
        """
        Initializes the DiffusionTrainer class for training Diffspec diffusion-based spectrogram super-resolution models.

        Args:
        - setting_data: Dictionary containing configuration and metadata for the training session.
        - model: The neural network model to be trained.
        - models_path: Directory path where trained models and checkpoints will be saved.
        - method: Training method or model type (e.g., 'diffusion', 'transcoder').
        - model_name: Name identifier for the model.
        - train_dataset: Dataset used for training (default: None).
        - valid_dataset: Dataset used for validation (default: None).
        - eval_dataset: Dataset used for evaluation (default: None).
        - learning_rate: Learning rate for the optimizer (default: 1e-3).
        - gradient_accumulation_steps: Number of steps to accumulate gradients before optimizer step (default: 1).
        - lr_warmup_steps: Number of steps for learning rate warmup (default: 500).
        - diff_steps: Number of diffusion steps (default: 1000).
        - schedule: Noise schedule type, e.g., 'VE' or 'DDPM' (default: 'VE').
        - model_chkpt_name: Optional checkpoint name to resume training (default: None).
        - model_chkpt_setting_name: Optional checkpoint settings name (default: None).
        - dtype: Data type for model tensors (default: torch.FloatTensor).
        - ltype: Data type for label tensors (default: torch.LongTensor).
        """

        #factors in front of loss functions, if a hybrid training method is chosen
        self.sample_size = model.sample_size  # the generated image resolution
        self.block_out_channels = model.config.block_out_channels
        self.down_block_types = model.config.down_block_types
        self.up_block_types = model.config.up_block_types
        self.layers_per_block = model.layers_per_block
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.diff_steps = diff_steps
        self.lr_warmup_steps = lr_warmup_steps
        self.learning_rate = learning_rate

        # MT: when using fp16 mixed precision, I get nans in the loss
        self.mixed_precision = 'no'  # `no` for float32, `fp16` for automatic mixed precision
        self.output_dir = './ddpm-butterflies-128'  # the model namy locally and on the HF Hub

        self.train_duration = 0

        self.setting_data = setting_data
        self.dtype = dtype
        self.ltype = ltype
        
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.eval_dataset = eval_dataset
        
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        
        self.models_path = models_path
        self.model_name = model_name
        self.method = method

        #for validation (not necessary if first validation is set before train)
        self.best_loss = float('inf')
        self.best_state_dict = copy.deepcopy(self.model.state_dict())
        self.best_epoch = -1
        self.step = 0
        print('TRAINED MODEL')
        ut.count_parameters(self.model)
        
        self.schedule = schedule
        if self.schedule == 'VE':
            self.noise_scheduler = ScoreSdeVeScheduler(num_train_timesteps=self.diff_steps)
        if self.schedule == 'DDPM':
            self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.diff_steps, beta_schedule='sigmoid')

        self.model_chkpt_name = model_chkpt_name

    def train(self, epochs=1, iteration=1, batch_size=1, device=torch.device("cpu")):
        if self.method == 'transcoder':
            self.gomin_model = GomiGAN.from_pretrained(
            pretrained_model_path="gomin/gan_state_dict.pt", **GANConfig().__dict__
            )
            self.gomin_model.eval()
            self.resampler = transforms.Resample(24000, 32000)
            self.gomin_mels_tr = bt.GominMelsTransform(device=device)
            self.mel_inv = transforms.InverseMelScale(n_stft=self.gomin_mels_tr.n_fft, n_mels=self.gomin_mels_tr.mel_bins, 
                                                      sample_rate=self.gomin_mels_tr.sr, f_min=self.gomin_mels_tr.fmin,
                                                      f_max=self.gomin_mels_tr.fmax, norm=self.gomin_mels_tr.norm,
                                                      mel_scale=self.gomin_mels_tr.mel_scale)
            self.gomin_model.to(device)
            self.resampler.to(device)
            self.mel_inv.to(device)

        self.iteration = iteration
        self.batch_size = batch_size
        self.epochs=epochs
        self.train_dataloader = DataLoader(self.train_dataset, batch_size = batch_size, shuffle = True)

        chkpt_epoch = 0
        if self.model_chkpt_name is not None:
            self.load_chkpt(self.model_chkpt_name, device=device)
            if "chkpt" in self.model_chkpt_name:
                chkpt_epoch = os.path.splitext(self.model_chkpt_name)[0].split('_')[-1].split('epoch')[-1]
            else:
                pattern = r'epoch=(\d+)\+'
                chkpt_epoch = int(re.search(pattern, self.model_chkpt_name).group(1))
            chkpt_epoch += 1
            self.lr_warmup_steps = 0

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=(len(self.train_dataloader) * epochs),
        )
        self.target_learning_rate = self.learning_rate
        # Initialize accelerator and tensorboard logging
        accelerator = Accelerator(
            mixed_precision=self.mixed_precision,
            gradient_accumulation_steps=self.gradient_accumulation_steps, 
        )
        
        self.model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        self.model.to(device)

        global_step = 0

        losses_train = []
        
        self.cur_iteration = 1
        # Now you train the model
        #save metadata of training (nb of epochs, of batch_size, size of the model etc...)
        self.save_data()

        for cur_epoch in range(chkpt_epoch, epochs):

            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {cur_epoch}")

            for idx, (_, spec, tho_spec, pann_logit, audio_name, _, _) in enumerate(train_dataloader):

                start_time = time.time()
                spec = spec.unsqueeze(dim=1).to(device)
                tho_spec = tho_spec.unsqueeze(dim=1).to(device)
                clean_images = spec
                clean_images = clean_images * 2 - 1
                tho_spec = tho_spec * 2 - 1
                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                bs = clean_images.shape[0]

                if self.method == 'transcoder':
                    timesteps = torch.zeros(batch_size)
                    timesteps = timesteps.to(device)
                    # dupli_tho_spec = torch.cat((tho_spec, tho_spec), dim=1)
                if self.method == 'diffusion':
                    if self.schedule == 'DDPM':
                        # Sample a random timestep for each image
                        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()
                    else:
                        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()
                        sigma = self.noise_scheduler.sigmas[self.diff_steps - timesteps - 1]
                        sigma = sigma.to(device)
                        timesteps = timesteps.to(device)

                    noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

                    #noisy image conditioned with third octave spectrogram
                    noisy_images_cond = torch.cat((noisy_images, tho_spec), dim=1)

                with accelerator.accumulate(self.model):
                    # Predict the noise residual
                    if self.method == 'diffusion':
                        if self.schedule == 'DDPM':
                            noise_pred = self.model(noisy_images_cond, timesteps, return_dict=False)[0]
                        else:
                            noise_pred = self.model(noisy_images_cond, sigma, return_dict=False)[0]
                        loss = F.mse_loss(noise_pred, noise)

                    accelerator.backward(loss)

                    batch_duration = time.time() - start_time
                    self.train_duration += batch_duration
                    self.optimizer.step()

                    accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                    if self.target_learning_rate > lr_scheduler.get_last_lr()[0]:
                        lr_scheduler.step()
                    optimizer.zero_grad()

                    losses_train.append(float(loss.data))

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}

                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

                if self.iteration is not None:
                    if (self.cur_iteration >= self.iteration):
                        break

                self.cur_iteration += 1

            self.best_state_dict = copy.deepcopy(self.model.state_dict())
            #save chckpt of epoch x if x is a multiple of 5 --> otherwise, too many useless checkpoints
            if cur_epoch % 5 == 0:
                self.save_model(epoch=cur_epoch)

        losses = {
                'losses_train': np.array(losses_train),
            }

        self.best_state_dict = copy.deepcopy(self.model.state_dict())   

        return(losses)

    def load_model(self, device):
        self.model = self.model.to(device)
        load_path = self.models_path + self.model_name +'.pt'
        print(f'LOADING MODEL : {load_path}')
        state_dict = torch.load(load_path, map_location=device)
        self.model.load_state_dict(state_dict)
    
    def load_chkpt(self, model_name, device=torch.device("cpu")):
        self.model = self.model.to(device)
        load_path = self.models_path + model_name +'.pt'
        print(f'WARNING: CHECKPOINT {load_path} LOADED FOR TRAINING')
        state_dict = torch.load(load_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device)

    def save_model(self, epoch=None):
    
        """
        SAVE MODEL
        """
        if epoch is None:
            save_path =  self.models_path + self.model_name + '.pt'
        else:
            save_path =  self.models_path + self.model_name + '__chkpt_epoch' + str(epoch) + '.pt'

        print(f'MODEL SAVED AT: {save_path}')
        torch.save(self.best_state_dict, save_path)
        # torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def save_data(self):
        setting_data = self.setting_data
        method = self.method

        if method == 'diffusion':
            setting_method = {
                'lr_warmup_steps': self.lr_warmup_steps,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "sample_size": self.sample_size,
                "diff_steps": self.diff_steps,
                "schedule": self.schedule,
                "block_out_channels" : self.block_out_channels,
                "down_block_types" : self.down_block_types,
                "up_block_types" : self.up_block_types,
                "layers_per_block" : self.layers_per_block
            }

        setting_model = {
            'iteration': self.cur_iteration,
            'epoch': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }

        setting_model.update(setting_method)
        setting_model.update(setting_data)

        with open(self.models_path + self.model_name + '_settings.yaml', 'w') as file:
            yaml.dump(setting_model, file)

        setting_model.update(setting_method)
        setting_model.update(setting_data)

        with open(self.models_path + self.model_name + '_settings.yaml', 'w') as file:
            yaml.dump(setting_model, file)


class DiffusionTrainerDiffWave:
    def __init__(self, setting_data, model, models_path, method, model_name, train_dataset=None, 
                 valid_dataset=None, eval_dataset=None, learning_rate=1e-3, gradient_accumulation_steps=1,
                 lr_warmup_steps=500, diff_steps=1000, schedule='VE', model_chkpt_name=None, model_chkpt_setting_name=None,
                 dtype=torch.FloatTensor, 
                 ltype=torch.LongTensor):
        """
        DiffusionTrainerDiffWave initializes a trainer for DiffWave-based speech synthesis or enhancement models.
        It seems that DiffWave is a good speech synthesizer, and you can condition it on audio waveform. This should 
        be investigated further.
        https://huggingface.co/speechbrain/tts-diffwave-ljspeech
        
        Args:
        - setting_data: Dictionary with configuration and metadata for the training session.
        - model: The neural network model to be trained (e.g., DiffWave).
        - models_path: Directory path for saving trained models and checkpoints.
        - method: Training method or model type (e.g., 'diffwave', 'transcoder').
        - model_name: Name identifier for the model.
        - train_dataset: Dataset for training (default: None).
        - valid_dataset: Dataset for validation (default: None).
        - eval_dataset: Dataset for evaluation (default: None).
        - learning_rate: Optimizer learning rate (default: 1e-3).
        - gradient_accumulation_steps: Steps to accumulate gradients before optimizer step (default: 1).
        - lr_warmup_steps: Number of steps for learning rate warmup (default: 500).
        - diff_steps: Number of diffusion steps (default: 1000).
        - schedule: Noise schedule type, e.g., 'VE' or 'DDPM' (default: 'VE').
        - model_chkpt_name: Optional checkpoint name to resume training (default: None).
        - model_chkpt_setting_name: Optional checkpoint settings name (default: None).
        - dtype: Data type for model tensors (default: torch.FloatTensor).
        - ltype: Data type for label tensors (default: torch.LongTensor).
        """

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.diff_steps = diff_steps
        self.noise_schedule = np.linspace(1e-4, 0.05, self.diff_steps)

        self.lr_warmup_steps = lr_warmup_steps
        self.learning_rate = learning_rate

        # MT: when using fp16 mixed precision, I get nans in the loss
        self.mixed_precision = 'no'  # `no` for float32, `fp16` for automatic mixed precision
        self.output_dir = './ddpm-butterflies-128'  # the model namy locally and on the HF Hub

        self.train_duration = 0

        self.setting_data = setting_data
        self.dtype = dtype
        self.ltype = ltype
        
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.eval_dataset = eval_dataset
        
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        
        self.models_path = models_path
        self.model_name = model_name
        self.method = method

        #for validation (not necessary if first validation is set before train)
        self.best_loss = float('inf')
        self.best_state_dict = copy.deepcopy(self.model.state_dict())
        self.best_epoch = -1
        self.step = 0
        print('TRAINED MODEL')
        ut.count_parameters(self.model)
        
        self.schedule = schedule
        if self.schedule == 'VE':
            self.noise_scheduler = ScoreSdeVeScheduler(num_train_timesteps=self.diff_steps)
        if self.schedule == 'DDPM':
            self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.diff_steps, beta_schedule='sigmoid')
        self.model_chkpt_name = model_chkpt_name
        self.model_chkpt_setting_name = model_chkpt_setting_name

    def train(self, epochs=1, iteration=1, batch_size=1, device=torch.device("cpu")):
        self.diff_method = DDPM(self.model, device=device, diff_steps=self.diff_steps)
        if self.method == 'transcoder':
            self.gomin_model = GomiGAN.from_pretrained(
            pretrained_model_path="gomin/gan_state_dict.pt", **GANConfig().__dict__
            )
            self.gomin_model.eval()
            self.resampler = transforms.Resample(24000, 32000)
            self.gomin_mels_tr = bt.GominMelsTransform(device=device)
            self.mel_inv = transforms.InverseMelScale(n_stft=self.gomin_mels_tr.n_fft, n_mels=self.gomin_mels_tr.mel_bins, 
                                                      sample_rate=self.gomin_mels_tr.sr, f_min=self.gomin_mels_tr.fmin,
                                                      f_max=self.gomin_mels_tr.fmax, norm=self.gomin_mels_tr.norm,
                                                      mel_scale=self.gomin_mels_tr.mel_scale)
            self.gomin_model.to(device)
            self.resampler.to(device)
            self.mel_inv.to(device)

        self.iteration = iteration
        self.batch_size = batch_size
        self.epochs=epochs
        self.train_dataloader = DataLoader(self.train_dataset, batch_size = batch_size, shuffle = True)

        chkpt_epoch = 0
        if self.model_chkpt_name is not None:
            self.load_chkpt(self.model_chkpt_name, device=device)
            if "chkpt" in self.model_chkpt_name:
                chkpt_epoch = os.path.splitext(self.model_chkpt_name)[0].split('_')[-1].split('epoch')[-1]
            else:
                pattern = r'epoch=(\d+)\+'
                chkpt_epoch = int(re.search(pattern, self.model_chkpt_name).group(1))
            chkpt_epoch += 1
            self.lr_warmup_steps = 0
            # Set seeds
            torch.manual_seed(torch.initial_seed() + chkpt_epoch)
            np.random.seed(np.random.get_state()[1][0] + chkpt_epoch)

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=(len(self.train_dataloader) * epochs),
        )
        self.target_learning_rate = self.learning_rate

        # Initialize accelerator and tensorboard logging
        accelerator = Accelerator(
            mixed_precision=self.mixed_precision,
            gradient_accumulation_steps=self.gradient_accumulation_steps, 
        )
        
        self.model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        global_step = 0

        losses_train = []
        
        self.cur_iteration = 1
        # Now you train the model
        #save metadata of training (nb of epochs, of batch_size, size of the model etc...)
        self.save_data()

        for cur_epoch in range(chkpt_epoch, epochs):
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {cur_epoch}")

            for idx, (_, waveform, tho_spec, _, audio_name) in enumerate(train_dataloader):
                
                start_time = time.time()
                # waveform = waveform.unsqueeze(dim=1)
                tho_spec = tho_spec.unsqueeze(dim=1)
                clean_images = waveform
                # clean_images = clean_images * 2 - 1
                tho_spec = tho_spec * 2 - 1
                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape).to(clean_images.device)

                if self.method == 'transcoder':
                    timesteps = torch.zeros(batch_size)
                    timesteps = timesteps.to(device)
                    # dupli_tho_spec = torch.cat((tho_spec, tho_spec), dim=1)
                if self.method == 'diffwave':
                    t = np.random.randint(len(self.noise_schedule), size=clean_images.shape[0])
                    noisy_images = self.diff_method(clean_images, t, noise)

                with accelerator.accumulate(self.model):
                    # Predict the noise residual
                    if self.method == 'diffwave':
                        if self.schedule == 'DDPM':
                            noisy_images = noisy_images.squeeze(dim=1)
                            tho_spec_input = tho_spec.squeeze(dim=1)
                            #MT: added for matching dimensions
                            pad = 32768 
                            noisy_images = F.interpolate(torch.unsqueeze(noisy_images, dim=0), size=pad, mode='linear', align_corners=False).squeeze(dim=0)

                            noise_pred = self.model(noisy_images, \
                                            torch.tensor(t, device = device), tho_spec_input).squeeze()
                            
                            #MT: added for matching dimensions
                            noise = F.interpolate(torch.unsqueeze(noise, dim=0), size=pad, mode='linear', align_corners=False).squeeze(dim=0)
                            
                        loss = F.mse_loss(noise_pred, noise)

                    accelerator.backward(loss)

                    batch_duration = time.time() - start_time
                    self.train_duration += batch_duration
                    self.optimizer.step()

                    accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                    if self.target_learning_rate > lr_scheduler.get_last_lr()[0]:
                        lr_scheduler.step()
                    optimizer.zero_grad()

                    losses_train.append(float(loss.data))

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}

                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

                if self.iteration is not None:
                    if (self.cur_iteration >= self.iteration):
                        break

                self.cur_iteration += 1

            self.best_state_dict = copy.deepcopy(self.model.state_dict())

            #save chckpt of epoch x if x is a multiple of 5 --> otherwise, too many useless checkpoints
            if cur_epoch % 5 == 0:
                self.save_model(epoch=cur_epoch)

        losses = {
                'losses_train': np.array(losses_train),
            }

        self.best_state_dict = copy.deepcopy(self.model.state_dict())   

        return(losses)

    def load_model(self, device):
        self.model = self.model.to(device)
        state_dict = torch.load(self.models_path / self.model_name + '.pt', map_location=device)
        self.model.load_state_dict(state_dict)

    def load_chkpt(self, model_name, device=torch.device("cpu")):
        self.model = self.model.to(device)
        load_path = self.models_path + model_name +'.pt'
        print(f'WARNING: CHECKPOINT {load_path} LOADED FOR TRAINING')
        state_dict = torch.load(load_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device)

    def save_model(self, epoch=None):
    
        """
        SAVE MODEL
        """
        if epoch is None:
            save_path =  self.models_path + self.model_name + '.pt'
        else:
            save_path =  self.models_path + self.model_name + '__chkpt_epoch' + str(epoch) + '.pt'

        print(f'MODEL SAVED AT: {save_path}')
        torch.save(self.best_state_dict, save_path)
        # torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def save_data(self):
        setting_data = self.setting_data
        method = self.method

        if method == 'diffwave':
            setting_method = {
                'lr_warmup_steps': self.lr_warmup_steps,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "diff_steps": self.diff_steps,
                "schedule": self.schedule,
                "res_channels" : self.model.res_channels,
                "n_layers" : self.model.n_layers,
                "n_mels" : self.model.n_mels,
            }

        setting_model = {
            'iteration': self.cur_iteration,
            'epoch': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }

        setting_model.update(setting_method)
        setting_model.update(setting_data)

        with open(self.models_path + self.model_name + '_settings.yaml', 'w') as file:
            yaml.dump(setting_model, file)

        setting_model.update(setting_method)
        setting_model.update(setting_data)

        with open(self.models_path + self.model_name + '_settings.yaml', 'w') as file:
            yaml.dump(setting_model, file)



