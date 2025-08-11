import numpy as np
import librosa
import os
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import torch
from utils.spectrogram_processing import WaveformToInput as TorchTransform
import math
import torch.nn.functional as F
from torchaudio import functional as Fa

import sys
from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import MelScale
# Add the parent directory of the project directory to the module search path
# project_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(project_parent_dir)
# from gomin.models import GomiGAN, DiffusionWrapper
# from gomin.config import GANConfig, DiffusionConfig
# from utils.util import get_least_power2_above

def get_least_power2_above(x):
    return np.power(2, math.ceil(np.log2(x)))

class GominMelsTransform():
    """Class used to calculate mels bands using the Gomin method for mels.
    (https://github.com/ryeoat3/gomin)

    Public Attributes
    ----------



    Private Attributes
    ----------



    """

    def __init__(self, flen_tho=4096, hlen_tho=4000, device=torch.device("cpu")):

        self.flen_tho = flen_tho
        self.device = device

        # for use:
        self.name = "Gomin"
        self.flen = 1024    
        self.hlen = 256
        self.window = 'hann'
        self.n_labels = 521
        self.mel_bins = 128
        self.sr = 24000
        self.flen_tho = flen_tho
        self.hlen_tho = hlen_tho
        self.fmin = 0
        self.fmax = (self.sr / 2.0)
        self.norm = "slaney"
        self.mel_scale = "slaney"
        self.n_fft = get_least_power2_above(self.flen)
        self.yamnet_tr = TorchTransform(flen_tho=flen_tho, hlen_tho=hlen_tho).to(device)
        # self.model = GomiGAN.from_pretrained(
        # pretrained_model_path="gan_state_dict.pt", **GANConfig().__dict__
        # )

        self.melspec_layer = MelSpectrogram(
            n_mels=self.mel_bins,
            sample_rate=self.sr,
            n_fft=self.n_fft,
            win_length=self.flen,
            hop_length=self.hlen,
            f_min=self.fmin,
            f_max=self.fmax,
            center=True,
            power=2.0,
            mel_scale=self.mel_scale,
            norm=self.norm,
            normalized=True,
            pad_mode="constant",
        )

        self.mel_matrix = MelScale(
            self.mel_bins, self.sr, self.fmin, self.fmax, self.n_fft // 2 + 1, self.norm, self.mel_scale
        )

        self.mel_basis = Fa.melscale_fbanks(self.n_fft // 2 + 1, self.fmin, self.fmax, self.mel_bins, self.sr, self.norm, self.mel_scale)

        self.inv_mel_basis = torch.linalg.pinv(
            self.mel_basis, rcond=1e-15)
        self.inv_mel_basis = F.relu(self.inv_mel_basis)

        self.mel_matrix_tho = MelScale(
            self.mel_bins, sample_rate=32000, f_min=self.fmin, f_max=self.fmax, n_stft=flen_tho // 2 + 1, norm=self.norm, mel_scale=self.mel_scale
        )
        
    def wave_to_mels(self, x, squeeze=True):

        x_wave = torch.Tensor(x).unsqueeze(0)
        melspec = self.melspec_layer(x_wave)
        melspec = self.power_to_db(melspec)
        # melspec = 10 * torch.log10(melspec + 1e-10)
        # melspec = torch.clamp((melspec + 100) / 100, min=0.0)

        #melspec = self.model.prepare_melspectrogram(x_wave)
        # print('AAAA')
        # print(melspec.T.shape)
        # print('BBBBBBB')
        # print(torch.transpose(melspec, 1, 2).shape)
        melspec = torch.transpose(melspec, 1, 2)
        return(melspec)

    def power_to_mels(self, spectrogram):
        spectrogram = torch.swapaxes(spectrogram, 1, 2)
        mel_specgram = self.mel_matrix_tho(spectrogram)
        mel_specgram = self.power_to_db(mel_specgram)
        return(mel_specgram)

    def power_to_db(self, input):
        melspec = 10 * torch.log10(input + 1e-10)
        melspec = torch.clamp((melspec + 100) / 100, min=0.0)
        return(melspec)

    def db_to_power(self, input):
        stft_spec = 100*10**(input/10)-100
        return(stft_spec)        

    def get_stft_num_time_bin(self, audio_len):
        #audio_len is the length of the audio (in s)
        ones_tensor = torch.ones((1, round(audio_len*24000)))
        melspec = self.melspec_layer(ones_tensor)
        return(melspec.shape[-1])
    
    def compensate_energy_loss(self, x_power, target_tr):
        scaling_factor = target_tr.sr / self.sr
        # scaling factor is raised to the power of 2 because it is supposed de be used 
        # on an energy spectrum and x_power is a power spectrum
        scaling_factor = scaling_factor ** 2
        
        # scaling of the power spectrum to fit the scale of the stft used in the Mel transform
        x_power = x_power * scaling_factor
        return(x_power)
    
class ThirdOctaveTransform():
    """Class used to calculate third-octave bands and mel bands. Take 
    a given frame length and hop length at initialisation, and compute 
    the same stft for third-octave calculation and for mel-bands 
    calculation. An instance of this class ensures that every operation
    and parameter is the same for Mel bands and for third-octave bands.

    This third-octave transform is based on Cense third octave according from Cense Lorient project:
    https://github.com/nicolas-f/noisesensor/blob/master/core/src/acoustic_indicators.c#L163

    Public Attributes
    ----------
    sr : str 
        sample rate (in Hz)
    flen : int
        frame length. Only values 4096 (fast) and 32976 (slow) are accepted
        in the current state. At a sampling rate of 32kHz, those windows 
        are therefore of 125ms (fast) and 1s (slow)
    hlen : int
        hop length. At 32kHz, set to 4000 to get exactly a 125ms window, and to 
        32000 to get a 1s window.

    tho_basis : np.ndarray
        full third-octave matrix filterbank.

    n_tho : int
        number of third-octave bands used.

    mel_basis : np.ndarray
        full Mel bands matrix filterbank
        number of Mel bands used 
    n_mel : int


    Private Attributes
    ----------
    f : list
        list of min and max frequencies indices that have none zero weights 
        in each of the 29 third-octave band.
        This list could for exemple take the form: 
            [[3,4], [4,5], [4,6], ..., [1291, 2025]]

    H : list
        list of non zero weights in each 29 third-octave band.

    w : np.ndarray
        array of flen ones, used for fft normalization.

    fft_norm : float
        normalization factor for fft.

    corr_global : float
        Correction for the 'fg' power to dB conversion.
        This should be deducted from outputs in wave_to_third_octave, but is 
        instead covered by lvl_offset_db in data_loader


    """

    def __init__(self, sr, flen, hlen, refFreq=17, n_tho=29, db_delta=26):
        # Constants: process parameters
        self.sr = sr
        self.flen = int(flen)
        self.hlen = int(hlen)
        self.n_tho=n_tho
        self.refFreq = refFreq

        # Construction of the full thrd-octave filter bank
        self.tho_basis_list = np.zeros((self.n_tho, int(1+self.flen/2)))

        # Cense Tho Basis
        #refFreq = 17
        freqByCell = self.sr / self.flen

        #MT: TO BE DELETED
        for iBand in range(self.n_tho):
            fCenter = pow(10, (iBand - refFreq)/10.) * 1000
            fLower = fCenter * pow(10, -1. / 20.)
            fUpper = fCenter * pow(10, 1. / 20.)
            cellFloor = int(math.ceil(fLower / freqByCell))
            cellCeil = int(math.ceil(fUpper / freqByCell))
            for id_cell in range(cellFloor, cellCeil+1):
                self.tho_basis_list[iBand][id_cell] = 1

        self.tho_basis = np.array(self.tho_basis_list, dtype=float)
        self.tho_basis_torch = torch.Tensor(self.tho_basis_list)

        self.inv_tho_basis_torch = torch.linalg.pinv(
            self.tho_basis_torch, rcond=1e-15)
        self.inv_tho_basis_torch = F.relu(self.inv_tho_basis_torch)

        # Declarations/Initialisations
        self.w = np.ones(self.flen)
        self.fft_norm = np.sum(np.square(self.w))/self.flen
        # This should be deducted from outputs in wave_to_third_octave, but is instead covered by lvl_offset_db in data_loader
        self.corr_global = 20*np.log10(self.flen/np.sqrt(2))

        # MT: nicolas ref sound pressure
        # db_delta is related the the sensitivity of the microphone. For cense, the
        # sensitivity is -26dB, so the db_delta is 26 (to add 26dB) to the given
        # calculated value. This is a mistake ! In the future, only use db_delta=0 to have a dbfs scale. Dcase paper trained with it unfortunately.
        # self.db_delta = 26
        self.db_delta = db_delta

        #just to contrast with the other third-octave transform
        self.mel_template = None
        self.tho_freq = True
        self.tho_time = True

    def wave_to_third_octave(self, x, zeropad=True, db=True):
        """Convert an audio waveform to a third-octave spectrogram.

        Parameters
        ----------
        x : np.ndarray
            waveform to convert. 

        zeropad : boolean
            apply zero-padding if True, else truncate array to match window
            length.

        Returns
        -------
        np.ndarray
        """

        # This calculation is based on Nicolas Fortin's work, with Cense
        # censors:
        # https://github.com/nicolas-f/noisesensor/blob/master/core/src/acoustic_indicators.c#L163

        if (x.shape[0]-self.flen) % self.hlen != 0:
            if zeropad:
                x = np.append(x, np.zeros(
                    self.hlen-(x.shape[0]-self.flen) % self.hlen))
            else:
                x = x[:-((x.shape[0]-self.flen) % self.hlen)]

        nFrames = int(np.floor((x.shape[0]-self.flen)/self.hlen+1))

        X_tob = np.zeros((self.n_tho, nFrames))

        tukey_window, energy_correction = self._tukey_window(
            M=self.flen, alpha=0.2)

        self.energy_correction = energy_correction

        #refFreq = 17
        freqByCell = self.sr / self.flen
        for iFrame in range(nFrames):
            x_frame = x[iFrame*self.hlen:iFrame*self.hlen+self.flen]
            x_frame = x_frame * tukey_window
            X = np.fft.rfft(x_frame)
            X = np.square(np.absolute(X))/self.fft_norm
            for iBand in range(self.n_tho):
                fCenter = pow(10, (iBand - self.refFreq)/10.) * 1000
                fLower = fCenter * pow(10, -1. / 20.)
                fUpper = fCenter * pow(10, 1. / 20.)

                cellFloor = int(math.ceil(fLower / freqByCell))
                cellCeil = int(math.ceil(fUpper / freqByCell))

                sumRms = 0
                for id_cell in range(cellFloor, cellCeil+1):
                    sumRms = sumRms + X[id_cell]

                rms = (np.sqrt(sumRms/2) / (self.flen/2)) * energy_correction
                #X_tob[iBand, iFrame] = rms
                X_tob[iBand, iFrame] = rms

            if db:
                X_tob[:, iFrame] = self.power_to_db(X_tob[:, iFrame])
                if X_tob[iBand, iFrame] == 0:
                    X_tob[iBand, iFrame] = 1e-15

        return X_tob

    def power_to_db(self, X):
        """Convert an amplitude squared spectrogram to a decibel (dB) 
        spectrogram using Félix Gontier decibel calculation.

        Parameters
        ----------
        X : np.ndarray
            Amplitude Squared Spectrogram. 

        Returns
        -------
        np.ndarray
        """
        '''
        WARNING: this function should be called "energy_to_db", as it
        takes as input an energy spectrum, but to make
        things easier if another third octave transform is used,
         it is still called "power_to_db"
        '''
        if np.min(X) <= 0:
            print('some values are null or negative. Being replaced by 10e-10.')
            X[X <= 0] = 10e-10

        X_db = 20*np.log10(X) + self.db_delta
        return(X_db)

    def db_to_power_torch(self, X_db, device=torch.device("cpu")):
        """Convert an amplitude squared spectrogram to a decibel (dB) 
        spectrogram.

        Parameters
        ----------
        X_db : np.ndarray
            Decibel Spectrogram. 

        Returns
        -------
        np.ndarray
        """
        '''
        WARNING: this function is a true "db_to_power", in the sense that it
        gets back to a power spectrum and not an energy spectrum. So it is not 
        the inverse of power_to_db (because power_to_db should be called
        energy_to_db in the case of Cense data)
        '''
        # inversion of the db calculation made in power_to_db
        X = 10**((X_db-self.db_delta)/20).to(device)
        # inversion of the rms to get back to sumRms, which is what is used
        X = (X * self.flen * np.sqrt(2) / 2)**2
        # X = 10**((X_db-self.db_delta)/20).to(device)

        #MT: test - it is possible that the sum inside a band is calculated, but we want the mean ?
        # X = (X / (self.flen/2))

        return(X)

    def wave_to_power(self, x, zeropad=True, db=True):
        """Convert an audio waveform to a third-octave spectrogram.

        Parameters
        ----------
        x : np.ndarray
            waveform to convert. 

        zeropad : boolean
            apply zero-padding if True, else truncate array to match window
            length.

        dbtype : string
            apply decibel (dB) spectrogram if not None. 'mt' for Modan 
            Tailleur decibel calculation choice, 'fg' for Felix Gontier
            decibel calculation choice.

        Returns
        -------
        np.ndarray
        """

        # This calculation is based on Nicolas Fortin's work, with Cense
        # censors:
        # https://github.com/nicolas-f/noisesensor/blob/master/core/src/acoustic_indicators.c#L163

        if (x.shape[0]-self.flen) % self.hlen != 0:
            if zeropad:
                x = np.append(x, np.zeros(
                    self.hlen-(x.shape[0]-self.flen) % self.hlen))
            else:
                x = x[:-((x.shape[0]-self.flen) % self.hlen)]

        nFrames = int(np.floor((x.shape[0]-self.flen)/self.hlen+1))

        X_tob = np.zeros((nFrames, 2049))

        tukey_window, energy_correction = self._tukey_window(
            M=self.flen, alpha=0.2)

        #DEACTIVATE
        #energy_correction = 1.0
        self.energy_correction = energy_correction

        refFreq = 17
        freqByCell = self.sr / self.flen
        for iFrame in range(nFrames):
            x_frame = x[iFrame*self.hlen:iFrame*self.hlen+self.flen]
            x_frame = x_frame * tukey_window

            # Squared magnitude of RFFT
            # Félix Gontier version below:
            #X = np.fft.rfft(x[iFrame*self.hlen:iFrame*self.hlen+self.flen]*self.w)
            X = np.fft.rfft(x_frame)
            X = np.square(np.absolute(X))/self.fft_norm
            # X = X / 16

            X_tob[iFrame] = X

        return X_tob

    def _tukey_window(self, M, alpha=0.2):
        """Return a Tukey window, also known as a tapered cosine window, and an 
        energy correction value to make sure to preserve energy.
        Window and energy correction calculated according to:
        https://github.com/nicolas-f/noisesensor/blob/master/core/src/acoustic_indicators.c#L150

        Parameters
        ----------
        M : int
            Number of points in the output window. 
        alpha : float, optional
            Shape parameter of the Tukey window, representing the fraction of the
            window inside the cosine tapered region.
            If zero, the Tukey window is equivalent to a rectangular window.
            If one, the Tukey window is equivalent to a Hann window.

        Returns
        -------
        window : ndarray
            The window, with the maximum value normalized to 1.
        energy_correction : float
            The energy_correction used to compensate the loss of energy due to
            the windowing
        """
        index_begin_flat = int((alpha / 2) * M)
        index_end_flat = int(M - index_begin_flat)
        energy_correction = 0
        window = np.zeros(M)

        for i in range(index_begin_flat):
            window_value = (
                0.5 * (1 + math.cos(2 * math.pi / alpha * ((i / M) - alpha / 2))))
            energy_correction += window_value * window_value
            window[i] = window_value

        # window*window=1
        energy_correction += (index_end_flat - index_begin_flat)
        for i in range(index_begin_flat, index_end_flat):
            window[i] = 1

        for i in range(index_end_flat, M):
            window_value = (
                0.5 * (1 + math.cos(2 * math.pi / alpha * ((i / M) - 1 + alpha / 2))))
            energy_correction += window_value * window_value
            window[i] = window_value

        energy_correction = 1 / math.sqrt(energy_correction / M)

        return(window, energy_correction)

    def compensate_energy_loss(self, x_power, target_tr):
        """
        The energy is not the same depending on the size of the temporal 
        window that is used. The larger the window, the bigger the energy 
        (because it sums on every frequency bin). A scale factor is needed
        in order to correct this. For example, with a window size of 1024, 
        the number of frequency bins after the fft will be 513 (N/2 +1). 
        With a window size of 4096, the number of frequency bins after the
        fft will be 2049 (N/2 +1). The scaling factor is then 2049/513.

        x_power: non dB power spectrogram

        target_tr : transform classes instance (Mels transform or Fine Bands transform)
            target transform to match (see PANNMelsTransform for example)

        """

        scaling_factor = (1+target_tr.flen/2) / (1+self.flen/2)

        if target_tr.name == 'Gomin':
            #compensate normalisation done with gomin spectrogram calculation
            fct = torch.hann_window
            window = fct(1024)
            scaling_factor = scaling_factor / window.pow(2.0).sum().sqrt()

        if target_tr.window == 'hann':
            #hann window loses 50% of energy
            scaling_factor = scaling_factor * 0.5
        else:
            raise Exception("Window unrecognised.") 

        # scaling factor is raised to the power of 2 because it is supposed de be used 
        # on an energy spectrum and x_power is a power spectrum
        scaling_factor = scaling_factor ** 2
        
        # scaling of the power spectrum to fit the scale of the stft used in the Mel transform
        x_power = x_power * scaling_factor
        return(x_power)