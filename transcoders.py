import numpy as np
import yaml
import numpy as np
import numpy as np
import torch
import utils.pinv_transcoder as bi
import yaml
from utils.util import get_transforms
from diffusion.diffusion_models import UNet2DModel
from diffusion.diffusion_inference import DDPMInference, ScoreSdeVeInference
from diffusers import DDPMScheduler, ScoreSdeVeScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from gomin.models import GomiGAN
from gomin.config import GANConfig
import utils.util as ut
import utils.bands_transform as bt
import utils.pinv_transcoder as pt
import librosa
import requests

class MelDataset(object):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class ThirdOctaveToMelTranscoderPinv():

    def __init__(self, model_path, model_name, device, classifier='PANN'):

        self.device = device

        model_raw_name, _ = os.path.splitext(model_path + "/" + model_name)
        with open(model_raw_name + '_settings.yaml') as file:
            settings_model = yaml.load(file, Loader=yaml.FullLoader)

        self.input_shape = settings_model.get('input_shape')
        self.output_shape = settings_model.get('output_shape')

        #from settings of the model
        classifier = settings_model.get('mels_type')

        self.tho_tr, self.mels_tr = get_transforms(sr=32000, 
                                            flen=4096,
                                            hlen=4000,
                                            classifier=classifier,
                                            device=device)


    def thirdo_to_mels_1s(self, x, dtype=torch.FloatTensor):
        x_inf = bi.pinv(x, self.tho_tr, self.mels_tr, reshape=self.output_shape[0], dtype=dtype, device=self.device)
        x_inf = x_inf[0].T.numpy()
        return(x_inf)

    def wave_to_thirdo_to_mels_1s(self, x, dtype=torch.FloatTensor):
        x_tho = self.tho_tr.wave_to_third_octave(x)
        x_tho = torch.from_numpy(x_tho.T)
        x_tho = x_tho.unsqueeze(0)
        x_tho = x_tho.type(dtype)
        x_mels_inf = self.thirdo_to_mels_1s(x_tho)
        return(x_mels_inf)
    
    def wave_to_thirdo_to_mels(self, x, chunk_size=None):
        if chunk_size is None:
            chunk_size = self.tho_tr.sr
        x_sliced = [x[i:i+chunk_size] for i in range(0, len(x), chunk_size)]
        x_mels_inf_tot = np.empty((self.mels_tr.mel_bins, 0))
        for x_i in x_sliced:
            x_mels_inf = self.wave_to_thirdo_to_mels_1s(x_i)
            x_mels_inf_tot = np.concatenate((x_mels_inf_tot, x_mels_inf), axis=1)
        return(x_mels_inf_tot)

class ThirdOctaveToMelTranscoderDiffusion():
    '''
    Contains all the functions to transcode third-octave to mels and to make
    PANN predictions. 
    '''
    def __init__(self, model_name, model_path, device, dtype):
        '''
        Parameters:
        - model_name: str
            Specifies the filename (excluding the path) of the pre-trained model to be loaded.
        - model_path: str
            Specifies the directory where the model file and its corresponding settings file are stored.
        - device: torch.device
            Specifies the computing device (CPU or GPU) where the PyTorch model will be loaded and executed.
        - flen: int, optional (default=4096)
            Frame length for the third-octave spectrograms. 4096 is a fast third-octave spectrogram.
        - hlen: int, optional (default=4000)
            Hop length for the the third-octave spectrograms. 4000 is a fast third-octave spectrogram.
        - pann_type: str, optional (default='ResNet38')
            Type of PANN model to use. Only applicable if classifier is 'PANN'.
            Specifies the architecture or type of PANN (Pre-trained Audio Neural Network) model to use for classification.
        '''
        self.device = device
        self.dtype = dtype

        if model_name == 'tau':
            model_full_name = 'tau_diff_steps=1000+epoch=70+learning_rate=-4_model.pt'
            model_url = "https://huggingface.co/datasets/modantailleur/fast-to-wav/resolve/main/tau_diff_steps%3D1000%2Bepoch%3D70%2Blearning_rate%3D-4_model.pt"
            yaml_url = "https://huggingface.co/datasets/modantailleur/fast-to-wav/resolve/main/tau_diff_steps%3D1000%2Bepoch%3D70%2Blearning_rate%3D-4_model_settings.yaml"
        else:
            print('Model name not recognized, using default tau model.')
            model_full_name = 'tau_diff_steps=1000+epoch=70+learning_rate=-4_model.pt'
            model_url = "https://huggingface.co/datasets/modantailleur/fast-to-wav/resolve/main/tau_diff_steps%3D1000%2Bepoch%3D70%2Blearning_rate%3D-4_model.pt"
            yaml_url = "https://huggingface.co/datasets/modantailleur/fast-to-wav/resolve/main/tau_diff_steps%3D1000%2Bepoch%3D70%2Blearning_rate%3D-4_model_settings.yaml"

        model_raw_name, _ = os.path.splitext(model_path + "/" + model_full_name)
        fname = model_raw_name + '_settings.yaml'

        model_dest = os.path.join(model_path, model_full_name)
        yaml_dest = fname

        if not os.path.exists(model_dest):
            print(f"Downloading model from {model_url} ...")
            r = requests.get(model_url)
            with open(model_dest, 'wb') as f:
                f.write(r.content)

        if not os.path.exists(yaml_dest):
            print(f"Downloading model settings from {model_url} ...")
            r = requests.get(yaml_url)
            with open(yaml_dest, 'wb') as f:
                f.write(r.content)

        with open(fname) as file:
            settings_model = yaml.load(file, Loader=yaml.FullLoader)

        self.settings_model = settings_model

        self.model = UNet2DModel(
            #WARNING: This is only for square matrices, need to find a solution in case not square
            sample_size=settings_model.get('sample_size'),  # the target image resolution
            in_channels=2,  # the number of input channels, 3 for RGB images
            out_channels=1,  # the number of output channels
            layers_per_block=settings_model.get('layers_per_block'),  # how many ResNet layers to use per UNet block
            block_out_channels=settings_model.get('block_out_channels'),  # the number of output channes for each UNet block
            down_block_types=settings_model.get('down_block_types'), 
            up_block_types=settings_model.get('up_block_types'),
        )

        state_dict = torch.load(model_path + "/" + model_full_name, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        # self.diffusion_inference = diffusion_inference
        if settings_model.get('schedule') == 'VE':
            self.noise_scheduler = ScoreSdeVeScheduler(num_train_timesteps=settings_model.get('diff_steps'))
            self.diffusion_inference = ScoreSdeVeInference(self.model, self.noise_scheduler,settings_model.get('diff_steps'))

        if settings_model.get('schedule') == 'DDPM':
            self.noise_scheduler = DDPMScheduler(num_train_timesteps=settings_model.get('diff_steps'), beta_schedule='sigmoid')
            self.diffusion_inference = DDPMInference(self.model, self.noise_scheduler,settings_model.get('diff_steps'))

        self.gomin_model = GomiGAN.from_pretrained(
        pretrained_model_path="gomin/gan_state_dict.pt", **GANConfig().__dict__
        )
        self.gomin_model.eval()
        self.gomin_model.to(device)

        #chunkers used to chunk audio and spectrograms into overlapping macroscopic frames
        self.audio_chunker_mel = ut.AudioChunks(n=round(self.settings_model.get('mel')['sr']*self.settings_model.get('mel')['chunk_len']), hop=round(self.settings_model.get('mel')['sr']*self.settings_model.get('mel')['hop_len']))
        self.gomin_mel_chunker = ut.AudioChunks(n=self.settings_model.get('mel')['n_time'], hop=self.settings_model.get('mel')['n_time_hop'])
        self.audio_chunker_thopinv = ut.AudioChunks(n=round(self.settings_model.get('third_octave')['sr']*self.settings_model.get('mel')['chunk_len']), hop=round(self.settings_model.get('third_octave')['sr']*self.settings_model.get('mel')['hop_len']))
        self.thirdo_chunker = ut.AudioChunks(n=11, hop=10)

        self.tho_tr = bt.ThirdOctaveTransform(sr=self.settings_model.get('third_octave')['sr'], flen=4096, hlen=4000, refFreq=self.settings_model.get('third_octave')['ref_freq'], n_tho=self.settings_model.get('third_octave')['n_tho'], db_delta=0)
        self.mels_tr = bt.GominMelsTransform()

    def load_mel_chunks(self, fname, output_type='mel'):
        if output_type == 'thirdopinvmel':
            x = librosa.load(fname, sr=self.settings_model.get('third_octave')['sr'])[0]
            audio_n = self.audio_chunker_thopinv.chunks_with_hop(x)
            thirdo_specs = []
            for audio in audio_n:
                thirdo_spec = torch.from_numpy(self.tho_tr.wave_to_third_octave(audio)).T.unsqueeze(dim=0).type(self.dtype)
                spec_from_thirdo = pt.pinv(thirdo_spec, self.tho_tr, self.mels_tr, reshape=self.settings_model.get('mel')['n_time'])
                thirdo_specs.append(spec_from_thirdo)
            thirdo_specs = np.array(thirdo_specs)
            return(thirdo_specs)  
                  
        if output_type == 'gominmel':
            x = librosa.load(fname, sr=self.settings_model.get('mel')['sr'])[0]
            audio_n = self.audio_chunker_mel.chunks_with_hop(x)
            specs = []
            for audio in audio_n:
                spec = self.mels_tr.wave_to_mels(audio).squeeze(dim=0).T.type(self.dtype).cpu().numpy()
                specs.append(spec)
            specs = np.array(specs)
            return(specs)
        
    def load_thirdo_chunks(self, x):
        thirdo_n = self.thirdo_chunker.chunks_with_hop(x)
        thirdo_specs = []
        for thirdo in thirdo_n:
            spec_from_thirdo = pt.pinv(thirdo.unsqueeze(0), self.tho_tr, self.mels_tr, reshape=self.settings_model.get('mel')['n_time'])
            thirdo_specs.append(spec_from_thirdo)
        thirdo_specs = np.array(thirdo_specs)
        return(thirdo_specs)  
    
    def mels_to_mels(self, x, torch_output=False, batch_size=8):
        dataset = MelDataset(x)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        tqdm_it=tqdm(dataloader, desc='EVALUATION')
        x_infs = None
        for (x) in tqdm_it:
            if torch_output:
                with torch.no_grad():
                    x = x.unsqueeze(dim=1)
                    x = x.to(self.device)
                    x = x * 2 - 1
                    x_inf = self.diffusion_inference.inference(x, device=self.device)
                    # wav_gomin = self.gomin_model(x_inf.squeeze(dim=1))
                    if x_infs is None:
                        x_infs = x_inf
                    else:
                        x_infs = torch.cat((x_infs, x_inf))
            else:
                x_inf = self.diffusion_inference.inference(x, device=self.device).detach()
                x_inf = x_inf[0].T.cpu().numpy()
        x_infs = x_infs.squeeze(dim=1).cpu().numpy()
        return(x_infs)

    def mels_to_audio(self, x, torch_output=False, batch_size=8):
        """
        Input x should be a 3D tensor of shape (batch_size, mel_bins, time_frames)
        """
        dataset = MelDataset(x)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        tqdm_it=tqdm(dataloader, desc='EVALUATION')
        wavs = None
        for (x) in tqdm_it:
            if torch_output:
                with torch.no_grad():
                    x = x.to(self.device)
                    x = x.type(self.dtype)
                    wav_gomin = self.gomin_model(x)
                    if wavs is None:
                        wavs = wav_gomin
                    else:
                        wavs = torch.cat((wavs, wav_gomin))
            else:
                x_inf = self.diffusion_inference.inference(x, device=self.device).detach()
                x_inf = x_inf[0].T.cpu().numpy()
        wavs = wavs.cpu().numpy()
        return(wavs)
    
    def thirdo_to_mels_to_audio(self, x, frame_duration=10, n_mels_frames_to_remove=1, n_mels_frames_per_s=100, n_thirdo_frames_per_s=8):
        '''
        Converts a 1-second third-octave spectrogram into Mel spectrogram frames and corresponding logits.

        Parameters:
        - x: torch.Tensor
            The input third-octave data with a shape of (batch_size, time_frames, frequency_bins).
        - frame_duration: int, optional (default=10)
            The duration in seconds for each frame when slicing the waveform. Default is 10 seconds
            because PANN ResNet38 is optimal with 10s chunks.
        - n_mels_frames_to_remove: int, optional (default=1)
            The number of Mel spectrogram frames to remove from the end of each chunk to avoid overlap artifacts.
            Default is 1 (for PANN).
        - n_mels_frames_per_s: int, optional (default=100)
            The number of Mel spectrogram frames expected per second. Used to determine the chunk size.
            Default is 100 frames per second (for PANN).

        Returns:
        - Tuple of torch.Tensor
            A tuple containing:
            1. The Mel spectrogram frames with a shape of (batch_size, time_frames, mel_bins).
            2. The corresponding logits with a shape of (batch_size, n_labels), where n_labels
            is the number of classification labels (527 for PANN).
        '''
        n_thirdo_frames_per_s = 8
        x_sliced = [x[:, i:i+n_thirdo_frames_per_s, :] for i in range(0, x.shape[1], n_thirdo_frames_per_s)]
        x_mels_inf_tot = torch.empty((x.shape[0], 0, self.mels_tr.mel_bins)).to(self.device)
        x_logits_tot = torch.empty((x.shape[0], self.classif_inference.n_labels, 0)).to(self.device)
        for k, x_i in enumerate(x_sliced):
            x_mels_inf = self.thirdo_to_mels_1s(x_i, torch_output=True)
            if k == len(x_sliced)-1:   
                x_mels_inf_tot = torch.cat((x_mels_inf_tot, x_mels_inf), axis=1)
            else:
                x_mels_inf_tot = torch.cat((x_mels_inf_tot, x_mels_inf[:, :-n_mels_frames_to_remove]), axis=1)

        if frame_duration != 1:
            chunk_size = frame_duration*n_mels_frames_per_s
            x_mels_sliced = [x_mels_inf_tot[:, i:i+chunk_size, :] for i in range(0, x_mels_inf_tot.shape[1], chunk_size)]
            for k, x_i in enumerate(x_mels_sliced):
                if x_i.shape[1] == chunk_size:
                    x_logits = self.classifier_prediction(x_i, torch_input=True, torch_output=True)
                    x_logits = x_logits.unsqueeze(dim=-1)
                    x_logits_tot =torch.concatenate((x_logits_tot, x_logits), axis=-1)

        return(x_mels_inf_tot, x_logits_tot)


    def thirdo_to_mels_to_logit(self, x, frame_duration=10, n_mels_frames_to_remove=1, n_mels_frames_per_s=100, n_thirdo_frames_per_s=8):
        '''
        Converts a 1-second third-octave spectrogram into Mel spectrogram frames and corresponding logits.

        Parameters:
        - x: torch.Tensor
            The input third-octave data with a shape of (batch_size, time_frames, frequency_bins).
        - frame_duration: int, optional (default=10)
            The duration in seconds for each frame when slicing the waveform. Default is 10 seconds
            because PANN ResNet38 is optimal with 10s chunks.
        - n_mels_frames_to_remove: int, optional (default=1)
            The number of Mel spectrogram frames to remove from the end of each chunk to avoid overlap artifacts.
            Default is 1 (for PANN).
        - n_mels_frames_per_s: int, optional (default=100)
            The number of Mel spectrogram frames expected per second. Used to determine the chunk size.
            Default is 100 frames per second (for PANN).

        Returns:
        - Tuple of torch.Tensor
            A tuple containing:
            1. The Mel spectrogram frames with a shape of (batch_size, time_frames, mel_bins).
            2. The corresponding logits with a shape of (batch_size, n_labels), where n_labels
            is the number of classification labels (527 for PANN).
        '''
        n_thirdo_frames_per_s = 8
        x_sliced = [x[:, i:i+n_thirdo_frames_per_s, :] for i in range(0, x.shape[1], n_thirdo_frames_per_s)]
        x_mels_inf_tot = torch.empty((x.shape[0], 0, self.mels_tr.mel_bins)).to(self.device)
        x_logits_tot = torch.empty((x.shape[0], self.classif_inference.n_labels, 0)).to(self.device)
        for k, x_i in enumerate(x_sliced):
            x_mels_inf = self.thirdo_to_mels_1s(x_i, torch_output=True)
            if k == len(x_sliced)-1:   
                x_mels_inf_tot = torch.cat((x_mels_inf_tot, x_mels_inf), axis=1)
            else:
                x_mels_inf_tot = torch.cat((x_mels_inf_tot, x_mels_inf[:, :-n_mels_frames_to_remove]), axis=1)

        if frame_duration != 1:
            chunk_size = frame_duration*n_mels_frames_per_s
            x_mels_sliced = [x_mels_inf_tot[:, i:i+chunk_size, :] for i in range(0, x_mels_inf_tot.shape[1], chunk_size)]
            for k, x_i in enumerate(x_mels_sliced):
                if x_i.shape[1] == chunk_size:
                    x_logits = self.classifier_prediction(x_i, torch_input=True, torch_output=True)
                    x_logits = x_logits.unsqueeze(dim=-1)
                    x_logits_tot =torch.concatenate((x_logits_tot, x_logits), axis=-1)

        return(x_mels_inf_tot, x_logits_tot)
