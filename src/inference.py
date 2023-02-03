import torch
import pandas as pd
import torchaudio
import audio_proc as ap
#import ast_models



class Discriminator():
    def __init__(self, model_path, labels_csv, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_length = 420
        self.model = torch.load(model_path).to(self.device)
        self.class_df = pd.read_csv(labels_csv)

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

    def inference(self, file_path, new_file=True):
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
