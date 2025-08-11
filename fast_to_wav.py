import pandas as pd
import torch
from tqdm import tqdm
from transcoders import ThirdOctaveToMelTranscoderDiffusion
import pickle
import numpy as np
import argparse
import os
import h5py
from scipy.io.wavfile import write
from torch.utils.data import Dataset

class DatasetGenerator(object):
    def __init__(self, spectral_data_path):
        _, self.extension = os.path.splitext(spectral_data_path)
        self.cense_data_path = spectral_data_path
        if self.extension == '.h5':
            with h5py.File(self.cense_data_path, 'r') as hf:
                self.spectral_data = hf['fast_125ms'][:]
        elif self.extension == '.npy':
            self.spectral_data = np.load(spectral_data_path)
        elif self.extension == '.csv':
            df = pd.read_csv(spectral_data_path)
            self.spectral_data = df[[col for col in df.columns if col.lower().startswith("fast_")]].values
            self.date_data = pd.to_datetime(df['epoch'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S').values
        else:
            with open(spectral_data_path, 'rb') as pickle_file:
                self.data_dict = pickle.load(pickle_file)
            self.spectral_data = self.data_dict['spectral_data']
            self.date_data = self.data_dict['date']

        self.len_dataset = len(self.spectral_data)

    def __getitem__(self, idx):
        spectral_data = self.spectral_data[idx]
        date_data = self.date_data[idx]
        return idx, spectral_data, date_data

    def __len__(self):
        return self.len_dataset

class SlidingWindowDataset(Dataset):
    def __init__(self, base_dataset, window_size=11):
        self.base_dataset = base_dataset
        self.window_size = window_size
        self.max_start_idx = len(base_dataset) - window_size

    def __len__(self):
        return self.max_start_idx + 1

    def __getitem__(self, idx):
        spectral_window = []
        date_window = []
        for i in range(self.window_size):
            _, spectral, date = self.base_dataset[idx + i]
            spectral_window.append(spectral)
            date_window.append(date)
        spectral_window = torch.stack(spectral_window, dim=0)  # (11, 29)
        return idx, spectral_window, date_window

#for CNN + PINV
class TranscoderDiffusionEvaluater:
    def __init__(self, transcoder, eval_dataset, dtype=torch.FloatTensor, db_offset=-88):
        self.dtype = dtype
        self.transcoder = transcoder
        self.eval_dataset = eval_dataset
        self.db_offset = db_offset

    def evaluate(self, batch_size=32, device=torch.device("cpu")):
                
        self.eval_dataloader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
        tqdm_it=tqdm(self.eval_dataloader, desc='EVALUATION: Chunk {}/{}'.format(0,0))
        
        waveform = np.array([])
        for (idx, spectral_data, date_data) in tqdm_it:
            if date_data[0] == 0:
                continue
            spectral_data = spectral_data.type(self.dtype)
            spectral_data = spectral_data.to(device)
            
            spectral_data = spectral_data + self.db_offset
            
            thirdopinvmel_chunks = self.transcoder.load_thirdo_chunks(spectral_data)
            diffusionmel_chunks = self.transcoder.mels_to_mels(thirdopinvmel_chunks, torch_output=True, batch_size=8)
            diffusionmel = self.transcoder.gomin_mel_chunker.concat_spec_with_hop(diffusionmel_chunks)

            #MT: temporary, to avoid memory issues
            diffusionmel = np.expand_dims(diffusionmel, axis=0)

            wav_diffusionmel = self.transcoder.mels_to_audio(diffusionmel, torch_output=True, batch_size=1)[0][0]

            if waveform.shape[0] != 0:
                waveform = np.concatenate((waveform, wav_diffusionmel), axis=0)
            else:
                waveform = wav_diffusionmel

        return(waveform) 

class LevelEvaluater:
    def __init__(self, eval_dataset, dtype=torch.FloatTensor):
        self.dtype = dtype
        self.eval_dataset = eval_dataset
        self.fn=np.array([20, 25, 31, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500])

    def evaluate(self, batch_size=1, device=torch.device("cpu")):
                
        self.eval_dataloader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
        tqdm_it=tqdm(self.eval_dataloader, desc='EVALUATION: Chunk {}/{}'.format(0,0))
        
        eval_outputs = np.array([])

        for (idx, spectral_data, date_data) in tqdm_it:
            spectral_data = spectral_data.type(self.dtype)
            spectral_data = spectral_data.to(device)

            puiss_spectral_data = 10**(spectral_data/10)
            sum_puiss_spectral_data = torch.sum(puiss_spectral_data, axis=-1)
            level_spectral_data = 10*torch.log10(sum_puiss_spectral_data)
            level_spectral_data = level_spectral_data.view(-1)
            
            if len(eval_outputs) != 0:
                eval_outputs = torch.cat((eval_outputs, level_spectral_data), dim=0)
            else:
                eval_outputs = level_spectral_data
                        
        eval_outputs = eval_outputs.detach().cpu().numpy()
        return(eval_outputs)

def calculate_db_offset(input_path='./spectral_data/', spectral_data_name= 'test.npy'):

    spectral_path = input_path + spectral_data_name
    dataset = DatasetGenerator(spectral_data_path=spectral_path)

    evaluater = LevelEvaluater(eval_dataset=dataset)
    batch_size = 1

    eval_outputs = evaluater.evaluate(batch_size=batch_size)

    eval_outputs = eval_outputs.reshape(-1)
    db_offset =  - np.percentile(eval_outputs, 99)

    print('DB OFFSET')
    print(db_offset)

    return(db_offset)

def compute_predictions(db_offset, batch_size=1, input_path='./spectral_data/', output_path='./predictions/', spectral_data_name= 'test.npy'):

    #transcoder setup
    MODEL_PATH = "./ckpt/"
    diffusion_model_name = 'tau'
    force_cpu = False
    #manage gpu
    useCuda = torch.cuda.is_available() and not force_cpu

    if useCuda:
        print('Using CUDA.')
        dtype = torch.cuda.FloatTensor
        ltype = torch.cuda.LongTensor
        #MT: add
        device = torch.device("cuda")
    else:
        print('No CUDA available.')
        dtype = torch.FloatTensor
        ltype = torch.LongTensor
        #MT: add
        device = torch.device("cpu")

    spectral_path = input_path + spectral_data_name
    dataset = DatasetGenerator(spectral_data_path=spectral_path)

    transcoder = ThirdOctaveToMelTranscoderDiffusion(diffusion_model_name, MODEL_PATH, device=device, dtype=dtype)
    evaluater = TranscoderDiffusionEvaluater(transcoder=transcoder, eval_dataset=dataset, db_offset=db_offset, dtype=dtype)

    waveform = evaluater.evaluate(batch_size=batch_size, device=device)

    output_file = os.path.splitext(spectral_data_name)[0] + '.wav'

    write(output_path + output_file, 24000, waveform)

def main(config):
    batch_size=480
    input_path=config.input_path
    output_path=config.output_path
    spectral_data_name= config.spectral_data_name

    if config.get_db_offset:
        db_offset = calculate_db_offset(input_path=input_path, spectral_data_name=spectral_data_name)
    else:
        db_offset = config.db_offset
        compute_predictions(db_offset=db_offset, batch_size=batch_size, input_path=input_path, output_path=output_path, spectral_data_name=spectral_data_name)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate 1s Mels and Third-Octave spectrograms')
    parser.add_argument('-i', '--input_path', type=str, default='./spectral_data/',
                        help='The path where the third-octave data files are stored, in a npy format')
    parser.add_argument('-o', '--output_path', type=str, default='./predictions/',
                        help='The path where to store the predictions')
    parser.add_argument('-n', '--spectral_data_name', type=str, default='test_short.csv',
                        help='name of the spectral data file in npy or h5 format')
    parser.add_argument('-dbo', '--db_offset', type=float, default=-88,
                        help='dB offset to apply to the measured third octaves. Needs to be calculated beforehand.')
    parser.add_argument('-gdbo', '--get_db_offset', action='store_true',
                    help='If set, only calculates the dB offset based on the max value of the given dataset.')
    config = parser.parse_args()
    main(config)



