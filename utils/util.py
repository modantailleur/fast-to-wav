import os
import torch
import yaml
import yamlloader
import numpy as np
import math
import utils.bands_transform as bt
from prettytable import PrettyTable

def chunks(lst, n):
    """
    Yield successive n-sized chunks from a list.

    Args:
        lst (list): The input list.
        n (int): The size of each chunk.

    Yields:
        list: A chunk of size n from the input list.

    Examples:
        >>> lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> for chunk in chunks(lst, 3):
        ...     print(chunk)
        [1, 2, 3]
        [4, 5, 6]
        [7, 8, 9]
        [10]
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class AudioChunks():
    def __init__(self, n, hop, fade=True):
        #number of elements in each chunk
        self.n = n
        #size of hop
        self.hop = hop
        self.diffhop = n - hop

        #whether to apply a fade in and fade out on each chunk or not
        self.fade = fade

    def calculate_num_chunks(self, wavesize):
        num_chunks = 1
        idx = 0
        audio_truncated=False
        
        if self.n == self.diffhop:
            step = self.n
        else:
            step = self.n-self.diffhop

        for i in range(self.n, wavesize-self.n+self.diffhop, step):
            num_chunks += 1
            idx = i

        if idx+2*(self.n-self.diffhop) == wavesize:
            num_chunks += 1
        else:
            audio_truncated=True

        if self.n == self.diffhop:
            if self.n*num_chunks == wavesize:
                audio_truncated=False
            else:
                audio_truncated=True
            
        return(num_chunks, audio_truncated)
    
    def chunks_with_hop(self, lst):
        if isinstance(lst, np.ndarray):
            return self._chunks_with_hop_np(lst)
        elif isinstance(lst, torch.Tensor):
            return self._chunks_with_hop_torch(lst)
        else:
            raise TypeError("Input must be either a numpy array or a torch tensor")

    def _chunks_with_hop_np(self, lst):
        if self.n != self.diffhop:
            L = []
            L.append(lst[0:self.n])
            idx = 0

            if self.n == self.diffhop:
                step = self.n
            else:
                step = self.n - self.diffhop

            for i in range(self.n, len(lst) - self.n + self.diffhop, step):
                L.append(lst[(i - self.diffhop):(i + self.n - self.diffhop)])
                idx = i
            if idx + 2 * (self.n - self.diffhop) == len(lst):
                L.append(lst[len(lst) - self.n:len(lst)])
        else:
            L = []
            step = self.n
            for i in range(0, len(lst), step):
                to_add = lst[i:i + step]
                if len(to_add) == step:
                    L.append(to_add)

        return np.array(L)

    def _chunks_with_hop_torch(self, lst):
        if self.n != self.diffhop:
            L = []
            L.append(lst[0:self.n])
            idx = 0

            if self.n == self.diffhop:
                step = self.n
            else:
                step = self.n - self.diffhop

            for i in range(self.n, len(lst) - self.n + self.diffhop, step):
                L.append(lst[(i - self.diffhop):(i + self.n - self.diffhop)])
                idx = i
            if idx + 2 * (self.n - self.diffhop) == len(lst):
                L.append(lst[len(lst) - self.n:len(lst)])
        else:
            L = []
            step = self.n
            for i in range(0, len(lst), step):
                to_add = lst[i:i + step]
                if len(to_add) == step:
                    L.append(to_add)

        return torch.stack(L)

    def concat_with_hop(self, L):
        if isinstance(L, np.ndarray):
            return self._concat_with_hop_np(L)
        elif isinstance(L, torch.Tensor):
            return self._concat_with_hop_torch(L)
        else:
            raise TypeError("Input must be either a numpy array or a torch tensor")

    def _concat_with_hop_np(self, L):
        lst = np.zeros(shape=L.shape[1] * L.shape[0] - (L.shape[0] - 1) * self.diffhop)
        if self.fade:
            bef = np.linspace(0, 1, self.diffhop)
            aft = np.linspace(1, 0, self.diffhop)
            mid = np.ones(L.shape[1] - self.diffhop)
            pond_g = np.concatenate((bef, mid))
            pond_d = np.concatenate((mid, aft))
        else:
            pond_g = np.ones(L.shape[1])
            pond_d = np.ones(L.shape[1])

        lst[0:L.shape[1]] = pond_d * L[0, :]
        for i in range(1, L.shape[0]):
            if i != L.shape[0] - 1:
                lst[i * L.shape[1] - i * self.diffhop:(i + 1) * L.shape[1] - i * self.diffhop] += pond_g * pond_d * L[i, :]
            else:
                lst[i * L.shape[1] - i * self.diffhop:(i + 1) * L.shape[1] - i * self.diffhop] += pond_g * L[i, :]

        return lst

    def _concat_with_hop_torch(self, L):
        lst = torch.zeros(L.shape[1] * L.shape[0] - (L.shape[0] - 1) * self.diffhop)
        if self.fade:
            bef = torch.linspace(0, 1, self.diffhop)
            aft = torch.linspace(1, 0, self.diffhop)
            mid = torch.ones(L.shape[1] - self.diffhop)
            pond_g = torch.cat((bef, mid))
            pond_d = torch.cat((mid, aft))
        else:
            pond_g = torch.ones(L.shape[1])
            pond_d = torch.ones(L.shape[1])

        lst[0:L.shape[1]] = pond_d * L[0, :]
        for i in range(1, L.shape[0]):
            if i != L.shape[0] - 1:
                lst[i * L.shape[1] - i * self.diffhop:(i + 1) * L.shape[1] - i * self.diffhop] += pond_g * pond_d * L[i, :]
            else:
                lst[i * L.shape[1] - i * self.diffhop:(i + 1) * L.shape[1] - i * self.diffhop] += pond_g * L[i, :]

        return lst

    def concat_spec_with_hop(self, L):
        if isinstance(L, np.ndarray):
            return self._concat_spec_with_hop_np(L)
        elif isinstance(L, torch.Tensor):
            return self._concat_spec_with_hop_torch(L)
        else:
            raise TypeError("Input must be either a numpy array or a torch tensor")

    def _concat_spec_with_hop_np(self, L):
        lst = np.zeros(shape=(L.shape[1], L.shape[2] * L.shape[0] - (L.shape[0] - 1) * self.diffhop))
        if self.fade:
            bef = np.linspace(0, 1, self.diffhop)
            aft = np.linspace(1, 0, self.diffhop)
            mid = np.ones(L.shape[2] - self.diffhop)
            pond_g = np.concatenate((bef, mid))
            pond_d = np.concatenate((mid, aft))

            pond_g = np.tile(pond_g, (L.shape[1], 1))
            pond_d = np.tile(pond_d, (L.shape[1], 1))
        else:
            pond_g = np.ones((L.shape[1], L.shape[2]))
            pond_d = np.ones((L.shape[1], L.shape[2]))

        lst[:, 0:L.shape[2]] = pond_d * L[0, :, :]
        for i in range(1, L.shape[0]):
            if i != L.shape[0] - 1:
                lst[:, i * L.shape[2] - i * self.diffhop:(i + 1) * L.shape[2] - i * self.diffhop] += pond_g * pond_d * L[i, :, :]
            else:
                lst[:, i * L.shape[2] - i * self.diffhop:(i + 1) * L.shape[2] - i * self.diffhop] += pond_g * L[i, :, :]

        return lst

    def _concat_spec_with_hop_torch(self, L):
        lst = torch.zeros((L.shape[1], L.shape[2] * L.shape[0] - (L.shape[0] - 1) * self.diffhop))
        if self.fade:
            bef = torch.linspace(0, 1, self.diffhop)
            aft = torch.linspace(1, 0, self.diffhop)
            mid = torch.ones(L.shape[2] - self.diffhop)
            pond_g = torch.cat((bef, mid))
            pond_d = torch.cat((mid, aft))

            pond_g = pond_g.repeat(L.shape[1], 1)
            pond_d = pond_d.repeat(L.shape[1], 1)
        else:
            pond_g = torch.ones((L.shape[1], L.shape[2]))
            pond_d = torch.ones((L.shape[1], L.shape[2]))

        lst[:, 0:L.shape[2]] = pond_d * L[0, :, :]
        for i in range(1, L.shape[0]):
            if i != L.shape[0] - 1:
                lst[:, i * L.shape[2] - i * self.diffhop:(i + 1) * L.shape[2] - i * self.diffhop] += pond_g * pond_d * L[i, :, :]
            else:
                lst[:, i * L.shape[2] - i * self.diffhop:(i + 1) * L.shape[2] - i * self.diffhop] += pond_g * L[i, :, :]

        return lst

class SettingsLoader(yaml.SafeLoader):
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super(SettingsLoader, self).__init__(stream)
    
    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, 'r') as f:
            #return yaml.load(f, YAMLLoader)
            return yaml.load(f, yamlloader)
SettingsLoader.add_constructor('!include', SettingsLoader.include)

def load_settings(file_path):
    with file_path.open('r') as f:
        return yaml.load(f, Loader=SettingsLoader)
    
def tukey_window(M, alpha=0.2):
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
    #nicolas' calculation
    index_begin_flat = int((alpha / 2) * M)
    index_end_flat = int(M - index_begin_flat)
    energy_correction = 0
    window = np.zeros(M)
    
    for i in range(index_begin_flat):
        window_value = (0.5 * (1 + math.cos(2 * math.pi / alpha * ((i / M) - alpha / 2))))
        energy_correction += window_value * window_value
        window[i]=window_value
    
    energy_correction += (index_end_flat - index_begin_flat) #window*window=1
    for i in range(index_begin_flat, index_end_flat):
        window[i] = 1
    
    for i in range(index_end_flat, M):
        window_value = (0.5 * (1 + math.cos(2 * math.pi / alpha * ((i / M) - 1 + alpha / 2))))
        energy_correction += window_value * window_value
        window[i] = window_value
    
    energy_correction = 1 / math.sqrt(energy_correction / M)
    
    return(window, energy_correction)

def get_transforms(sr=32000, flen=4096, hlen=4000, classifier='YamNet', device=torch.device("cpu"), tho_freq=True, tho_time=True, mel_template=None):
    if mel_template is None:
        tho_tr = bt.ThirdOctaveTransform(sr=sr, flen=flen, hlen=hlen)
        if classifier == 'PANN':
            mels_tr = bt.PANNMelsTransform(flen_tho=tho_tr.flen, hlen_tho=tho_tr.hlen, device=device)
        if classifier == 'YamNet':
            mels_tr = bt.YamNetMelsTransform(flen_tho=tho_tr.flen, hlen_tho=tho_tr.hlen, device=device)
        if classifier == 'default':
            mels_tr = bt.DefaultMelsTransform(sr=tho_tr.sr, flen=tho_tr.flen, hlen=tho_tr.hlen)
    else:
        tho_tr = bt.NewThirdOctaveTransform(32000, 1024, 320, 64, mel_template=mel_template, tho_freq=tho_freq, tho_time=tho_time)
        if classifier == 'PANN':
            mels_tr = bt.PANNMelsTransform(flen_tho=4096, device=device)
        if classifier == 'YamNet':
            mels_tr = bt.YamNetMelsTransform(flen_tho=4096, device=device)
        if classifier == 'default':
            mels_tr = bt.DefaultMelsTransform(sr=tho_tr.sr, flen=4096, hlen=4000)
    return(tho_tr, mels_tr)   

#count the number of parameters of a model
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def read_metadata(csv_path, classes_num, id_to_ix):
    """Read metadata of AudioSet from a csv file.

    Args:
      csv_path: str

    Returns:
      meta_dict: {'audio_name': (audios_num,), 'target': (audios_num, classes_num)}
    """

    with open(csv_path, 'r') as fr:
        lines = fr.readlines()
        lines = lines[3:]   # Remove heads

    audios_num = len(lines)
    targets = np.zeros((audios_num, classes_num), dtype=np.bool)
    audio_names = []
 
    for n, line in enumerate(lines):
        items = line.split(', ')
        """items: ['--4gqARaEJE', '0.000', '10.000', '"/m/068hy,/m/07q6cd_,/m/0bt9lr,/m/0jbk"\n']"""

        audio_name = 'Y{}.wav'.format(items[0])   # Audios are started with an extra 'Y' when downloading
        label_ids = items[3].split('"')[1].split(',')

        audio_names.append(audio_name)

        # Target
        for id in label_ids:
            ix = id_to_ix[id]
            targets[n, ix] = 1
    
    meta_dict = {'audio_name': np.array(audio_names), 'target': targets}
    return meta_dict


def float32_to_int16(x):
    assert np.max(np.abs(x)) <= 1.2
    x = np.clip(x, -1, 1)
    return (x * 32767.).astype(np.int16)

def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)
    
def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x))), axis=0)
    else:
        return x[0 : audio_length]