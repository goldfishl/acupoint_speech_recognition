import torch
import pandas as pd
import torchaudio
import numpy as np

from acpsr.train.dataset import embedding
from acpsr.model.audio_proc import default_fs as fs
import acpsr.model.audio_proc as ap

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Discriminator():
    def __init__(self, model_path='./models/audio_model.pth', 
            labels_csv='./models/class_labels_indices.csv', device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_length = 420
        self.model = torch.load(model_path).to(self.device)
        self.class_df = pd.read_csv(labels_csv)
        self.rnn_seg = Segmentator()

    def inference_single(self, source, topk):
        norm_mean = -6.845978
        norm_std = 5.5654526
        task = "ft_cls"
        if isinstance(source, str):
            waveform, fs = torchaudio.load(source)
        else:
            waveform, fs = source
            waveform = torch.tensor(waveform, dtype=torch.float32)
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True,
            sample_frequency=fs, use_energy=False,
            window_type='hanning', num_mel_bins=128,
            dither=0.0, frame_shift=10)
        n_frames = fbank.shape[0]
        p = self.target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[-self.target_length:, :]

        fbank = (fbank - norm_mean) / (norm_std * 2)

        fbank = fbank.unsqueeze(0)
        fbank = fbank.to(self.device)
        self.model.eval()
        out = self.model(fbank, task)
        out = torch.sigmoid(out)
        top_out = torch.topk(out, topk)
        output = []
        for i in range(topk):
            result = self.class_df.loc[self.class_df["index"] == int(top_out.indices[0, i]), "word"].values[0]
            prob = float(top_out.values[0, i])
            output.append((result, prob))
        return output

    def inference(self, file_path, seg_method='rule', new_file=True):

        if seg_method == "rule":
            waveform, fs = ap.load(file_path)
            segments, fs = ap.split(waveform, fs=fs, verbose=False)

            pred = []
            for seg in segments:
                if new_file is False:
                    pred.append(self.inference_single((seg, fs), 1)[0][0])
                else:
                    ap.export(".tmp.wav", seg, fs)
                    pred.append(self.inference_single(".tmp.wav", 1)[0][0])
            return pred

        if seg_method == "rnn":
            segments, fs = self.rnn_seg(file_path)
            norm_mean = -6.845978
            norm_std = 5.5654526

            pred = []
            for seg in segments:
                fbank = embedding(seg)
                n_frames = fbank.shape[0]
                p = self.target_length - n_frames
                if p > 0:
                    m = torch.nn.ZeroPad2d((0, 0, 0, p))
                    fbank = m(fbank)
                elif p < 0:
                    fbank = fbank[-self.target_length:, :]
                    
                fbank = (fbank - norm_mean) / (norm_std * 2)
                fbank = fbank.unsqueeze(0).to(device)
                self.model.eval()
                out = self.model(fbank, 'ft_cls')
                out = torch.sigmoid(out)
                top_out = torch.topk(out, 1)
                _pred = self.class_df.loc[self.class_df["index"] == int(top_out.indices[0, 0]), "word"].values[0]
                pred.append(_pred)
            return pred

        if seg_method != 'rnn' and seg_method != 'rule':
            print("choose a `rule` or `rnn` segmentator ")
            return



class Segmentator():
    def __init__(self, window_sample_num=400, stride_sample_num=160, 
            model_path = "./models/rnn_segmentator.pth"):
        self.model = torch.load(model_path).to(device)
        self.window_sample_num = window_sample_num
        self.stride_sample_num = stride_sample_num
        
    def get_window_boundary(self, sequence_length, window_size, window_stride):
        window_median_sample_num = int(window_size / 2)
        num_window = int( (sequence_length - window_size + window_stride) / window_stride )
        window_median = list(range(window_median_sample_num, sequence_length, window_stride))[:num_window]
        window_boundary = [_median - window_median_sample_num for _median in window_median]
        return window_boundary
    
    def boundary_detector(self, model_out):
        out = torch.argmax(model_out, dim=1)
        out_list = out.tolist()

        boundaries = []
        begin_index = -1
        boundary_window_size = 10
        boundary_sample_threshold = 5
        mini_seg_sample = 35
        for i in range(len(out_list) - boundary_window_size):
            detect_range = out_list[i:i+10]
            count = detect_range.count(0)
            # ending detect
            if begin_index > 0:
                # 整个框都是0
                if count == 10:
                    # 过滤掉时长过短的segment
                    if ((i+1) - begin_index) > mini_seg_sample:
                        boundaries.append((begin_index, i+1))
                    begin_index = -1
            # beginning detect
            else:
                # 暂时不考虑beginning_label和inside_label的区别
                if (boundary_window_size - count) > boundary_sample_threshold:
                    begin_index = i
        if begin_index > 0:
            boundaries.append((begin_index, len(out_list)-1))
        return boundaries
    
    def split(self, audio, model_out, pre_add=100*16, post_add=100*16):
        audio_segments = []
        model_out_boundaries = self.boundary_detector(model_out)
        window_boundaries = self.get_window_boundary(len(audio), self.window_sample_num, self.stride_sample_num)
        for b, o in model_out_boundaries:
            begin_sample_num = window_boundaries[b] - pre_add
            end_sample_num = window_boundaries[o] + post_add

            if begin_sample_num < 0:
                begin_sample_num = 0

            if end_sample_num >= len(audio):
                end_sample_num = len(audio)-1
            audio_segments.append(audio[begin_sample_num:end_sample_num])
        return audio_segments, fs
    
    def __call__(self, source):
        if isinstance(source, str):
            source, fs = torchaudio.load(source)
        embed = embedding(source).to(device)
        model_out = self.model(embed)
        audio = source.squeeze()
        return self.split(audio, model_out)