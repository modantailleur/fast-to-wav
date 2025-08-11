
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:02:54 2022

@author: user
"""

import numpy as np
import librosa
import os
import torch
from scipy.io.wavfile import write
from transcoders import ThirdOctaveToMelTranscoderDiffusion
import argparse

def apply_fade(audio, sr, duration=0.1):
    out = audio

    length = int(duration*sr)
    end_fo = audio.shape[0]
    start_fo = end_fo - length
    start_fi = 0
    end_fi = length

    fadeout_curve = np.linspace(1.0, 0.0, length)
    fadein_curve = np.linspace(0, 1.0, length)

    out[start_fo:end_fo] = out[start_fo:end_fo] * fadeout_curve
    out[start_fi:end_fi] = out[start_fi:end_fi] * fadein_curve
    return(out)

def main(config):
    MODEL_PATH = "./ckpt/"
    filename = config.audio_file
    diffusion_model_name = 'tau'
    full_filename = "audio/" + filename
    force_cpu = False
    #manage gpu
    useCuda = torch.cuda.is_available() and not force_cpu

    if useCuda:
        print('Using CUDA.')
        dtype = torch.cuda.FloatTensor
        #MT: add
        device = torch.device("cuda")
    else:
        print('No CUDA available.')
        dtype = torch.FloatTensor
        ltype = torch.LongTensor
        #MT: add
        device = torch.device("cpu")

    x_24k = librosa.load(full_filename, sr=24000)[0]

    transcoder_diffusion = ThirdOctaveToMelTranscoderDiffusion(diffusion_model_name, MODEL_PATH, device=device, dtype=dtype)

    origmel_chunks = transcoder_diffusion.load_mel_chunks(full_filename, output_type='gominmel')
    thirdopinvmel_chunks = transcoder_diffusion.load_mel_chunks(full_filename, output_type='thirdopinvmel')
    diffusionmel_chunks = transcoder_diffusion.mels_to_mels(thirdopinvmel_chunks, torch_output=True, batch_size=8)

    diffusionmel = transcoder_diffusion.gomin_mel_chunker.concat_spec_with_hop(diffusionmel_chunks)
    thirdopinvmel = transcoder_diffusion.gomin_mel_chunker.concat_spec_with_hop(thirdopinvmel_chunks)
    origmel = transcoder_diffusion.gomin_mel_chunker.concat_spec_with_hop(origmel_chunks)

    wav_thirdopinvmel = transcoder_diffusion.mels_to_audio(np.expand_dims(thirdopinvmel, axis=0), torch_output=True, batch_size=8)
    wav_origmel = transcoder_diffusion.mels_to_audio(np.expand_dims(origmel, axis=0), torch_output=True, batch_size=8)
    wav_diffusionmel = transcoder_diffusion.mels_to_audio(np.expand_dims(diffusionmel, axis=0), torch_output=True, batch_size=8)

    save_path = "./predictions/"+filename[:-4]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Directory '{save_path}' created.")
    else:
        print(f"Directory '{save_path}' already exists.")

    print('WRITING WAVS')
    write(save_path + '/' + filename[:-4] + '_ftw.wav', 24000, wav_diffusionmel)
    write(save_path + '/' + filename[:-4] + '_pinv.wav', 24000, wav_thirdopinvmel)
    write(save_path + '/' + filename[:-4] + '_original_mel.wav', 24000, wav_origmel)
    write(save_path + '/' + filename[:-4] + '_original.wav', 24000, x_24k)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform audio into different spectral representations, then transcode back to audio')

    parser.add_argument('audio_file', type=str,
                        help='Name of the original audio file that should be located in the "audio" folder')
    
    config = parser.parse_args()
    main(config)
