import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch import nn
import time
import random
import os
import pandas as pd
import math


class Audio_Embedding():
    def __init__(self, device):
        self.device = device
    
    def __call__(self, audio):
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        fbank = torchaudio.compliance.kaldi.fbank(audio, htk_compat=True, 
                                          sample_frequency=16000,
                                          window_type='hanning', num_mel_bins=128, 
                                          frame_length=25, frame_shift=10)
        return fbank


label_padding_value = 3
device = "cuda:0" if torch.cuda.is_available() else "cpu"
embedding = Audio_Embedding(device)

def write_split_dataset(path, file, num):
    file = os.path.join(path, file)
    f = open(file, "w")
    num = sorted(num)
    for i in num:
        f.write("sample_%s.wav" % (i))
        f.write("\n")
    f.close()

def split_dataset(split_rate=[0.8, 0.1, 0.1]):

    gener_path = os.path.join('./data', 'combined_prescription')
    with open(os.path.join(gener_path, 'sample_num'), 'r') as f:
        dataset_len = int(f.read())

    dataset_num = list(range(dataset_len))
    random.shuffle(dataset_num)
    training_len = round(dataset_len * split_rate[0])
    training_num = dataset_num[:training_len]
    val_len = training_len + int(dataset_len * split_rate[1])
    val_num = dataset_num[training_len:val_len]
    test_num = dataset_num[val_len:]
    
    files = ["training_list.txt", "validation_list.txt", "testing_list.txt"]
    num = [training_num, val_num, test_num]
    for _file, _num in zip(files, num):
        write_split_dataset(gener_path, _file, _num)

def _load_list(root, *filenames):
    output = []
    for filename in filenames:
        filepath = os.path.join(root, filename)
        with open(filepath) as fileobj:
            output += [os.path.abspath(os.path.join(root, 'data', line.strip())) for line in fileobj]
    return output

def caching_data(_walker):
    data = []
    for item in _walker:
        audio, sr = torchaudio.load(item)
        # audio = audio.squeeze()
        with open(item[:-3]+"seg") as f:
            transcript = f.read().strip()
        data.append((audio, transcript))
    return data


class Acupuncture_Prescription(Dataset):

    def __init__(self, root, subset):

        assert subset is None or subset in ["training", "validation", "testing"], (
            "When `subset` not None, it must take a value from "
            + "{'training', 'validation', 'testing'}."
        )

        self._path = root


        if subset == "training":
            self._walker = _load_list(self._path, "training_list.txt")
            self._data = caching_data(self._walker)
            
        elif subset == "validation":
            self._walker = _load_list(self._path, "validation_list.txt")
            self._data = caching_data(self._walker)
            
        elif subset == "testing":
            self._walker = _load_list(self._path, "testing_list.txt")
            self._data = caching_data(self._walker)

    def __getitem__(self, n: int):
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            (Tensor, int, str, str, int):
            ``(waveform, sample_rate, label, speaker_id, utterance_number)``
        """
        return self._data[n]


    def __len__(self) -> int:
        return len(self._data)  


def collate_batch(batch):
    embed_list, seq_len, transcript_list, = [], [], []
   
    for _audio,_transcript in batch:
        _embed = embedding(_audio)
        seq_len.append(_embed.shape[0])
        embed_list.append(_embed)
        _transcript = torch.tensor([int(char) for char in _transcript])
        transcript_list.append(_transcript)
        # debug
        # if _embed.shape[0] != len(_transcript):
        #     print(_embed.shape, len(_transcript))


    embed_list = pad_sequence(embed_list, batch_first=True) # default padding_value=0
    transcript_list = pad_sequence(transcript_list, batch_first=True, padding_value = label_padding_value) # 
    
    seq_len = torch.tensor(seq_len)

    return embed_list.to(device), seq_len.to("cpu"), transcript_list.to(device)
