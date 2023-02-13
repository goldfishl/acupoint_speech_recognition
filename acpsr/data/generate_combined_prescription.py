import os
import numpy as np
from scipy import stats
from scipy.io import wavfile
from scipy.signal import spectrogram
from IPython.display import Audio
import random
import itertools
import tqdm

import acpsr.model.audio_proc as ap
import acpsr.data.reader as reader


# 用于记录音频sample index的队列
class IndexQueue():
    def __init__(self):
        self.queue = []
        self.count = 0
    def put(self, data):
        self.queue.append(data)
    def get(self):
        if self.count == len(self.queue):
            return False
        self.count += 1
        return self.queue[self.count-1]
    def is_empty(self):
        if self.count == len(self.queue):
            return True
        else:
            return False
    def __len__(self):
        return len(self.queue)
    def __str__(self):
        return str(self.queue)
    
    
def get_window_median(sequence_length, window_size, window_stride):
    num_window = int( (sequence_length - window_size + window_stride) / window_stride )
    
    return list(range(int(window_size/2), sequence_length, window_stride))[:num_window]
        


def pres_generate(pres, all_words, max_permut=10, repeat=2):
    # Generate audio from a single entry within raw_pres
    # The algorithm works by generating all combinations of
    # 主穴位 and 辅穴,
    # then generate all permutations (TODO: limit permutations?)
    # For each generated permutation, randomly sample from available
    # speakers, and generate a maximum of 2 complete clips.
    
    # Intro/outtro time
    max_fade_time = 2000  # in ms
    
    # Gap between each word is randomly sampled
    # Depending on speaker style, the maximum is also randomly sampled
    max_gap_time = 3000  # in ms
    min_gap_time = 500  # in ms
    max_gap_time = random.randrange(min_gap_time, max_gap_time)
    
    # 主穴位组 and 辅助穴组
    # 一个主穴位组可能有零或多组辅助穴位，所以optional是一个list of list
    res, optional = pres
    
    # Filter out missing words
    res = [item for item in res if item in all_words]
    optional = [[item for item in opt  if item in all_words] for opt in optional]
    
    # 添加原主穴组
    all_res = [res]
    # 添加所有主穴组和辅助穴组的组合
    for item in optional:
        all_res.append(res + item)
        
    
    # Add permutation
    # This part needs to be optimised
    # (limit permutations to a maximum of 10 per entry maybe?)
    all_permuts = []
    for entry in all_res:
        single_item_permut = list(itertools.permutations(entry, len(entry)))
        random.shuffle(single_item_permut)
        if len(single_item_permut) > max_permut:
            single_item_permut = single_item_permut[:max_permut]
        all_permuts += single_item_permut
    
    # Generate audio data
    result = []
    for pres in all_permuts:
        for i in range(repeat):
            # Generate a single complete audio clip
            # randomly sample speakers for each word
            # TODO: optimise speaker selection?
            all_files = [random.choice(all_words[key])[1] for key in pres]
            # Get actual audio data
            # ap.load returns (data, fs) tuple,
            # All wav files should have been converted to sample rate of
            # ap.default_fs already, so "fs" should be consistent
            all_data = [ap.load(file)[0] for file in all_files]
            
            # char transcription
            index_queue=IndexQueue()

            
            # sample noise
            # sample rate by default is ap.default_fs
            
            # intro
            _time = random.randrange(0, max_fade_time)
            wav_data_list = [ap.noise(_time)]
            
            # gaps
            for data in all_data:
                _time = random.randrange(min_gap_time, max_gap_time)
                wav_data_list.append(ap.noise(_time))
                _sample_index = len(np.concatenate(wav_data_list))
                index_queue.put((_sample_index, "B"))
                
                wav_data_list.append(data)
                _sample_index = len(np.concatenate(wav_data_list))
                index_queue.put((_sample_index, "I"))

                
            # outtro
            _time = random.randrange(0, max_fade_time)
            # _sample_index += _time
            wav_data_list.append(ap.noise(random.randrange(0, max_fade_time)))
            
            # merge to a single np.array
            data = np.concatenate(wav_data_list)
            # at this point, the generated clip can be exported as wav files
            # using scipy.io.wavefile.write(filename, ap.default_fs, data)
            
            

            # count = 0
            # sample_index.append((99999999,"x"))
            # bug = len(data)
            # # 320: 每320个sample对应一个audio embedding
            # for _sample_index in range(0,len(data),320):
            #     if _sample_index > sample_index[count][0] and sample_index[count][1] == 'I':
            #         count += 1
            #     if _sample_index > sample_index[count][0] and sample_index[count][1] == 'B':
            #         count += 1
            #         label += beginning_label
            #     if _sample_index < sample_index[count][0] and sample_index[count][1] == 'I':
            #         label += inside_label
            beginning_label = "1"
            inside_label = "2"
            outside_label = "0"
            label = ""
            window_time = 25  # ms
            stride_time = 10  # ms
            window_sample_num = 25 * 16  # 16 samples per ms, 16K HZ sample rate
            stride_sample_num = 10 * 16
            window_median_sample_num = int(window_sample_num / 2)
            window_median = get_window_median(len(data), window_sample_num, stride_sample_num)
            sample_index, tag = index_queue.get()
            for _median in window_median:
                if tag == "B":
                    gap = _median-sample_index
                    gap_upper_bound = window_median_sample_num
                    gap_lower_bound = window_median_sample_num - stride_sample_num
                    if gap >= gap_lower_bound and gap <= gap_upper_bound:
                        label += beginning_label
                        if not index_queue.is_empty():
                            sample_index, tag = index_queue.get()
                    else:
                        label += outside_label
                elif tag == "I":
                    gap = _median-sample_index
                    gap_lower_bound = window_median_sample_num
                    if gap > gap_lower_bound:
                        label += outside_label
                        if not index_queue.is_empty():
                            sample_index, tag = index_queue.get()
                    else:
                        label += inside_label
                    
            result.append((data, label, pres))
    return result


def start_generation():
    print("Start generating")
    all_words = reader.get_all_wav(include_train=True)
    raw_pres = reader.get_raw_pres()
    gener_path = os.path.join('./data', 'combined_prescription')

    # Generating a list of missing words
    # There's not a lot of them, I've noted some
    # observations down in Clean.txt
    missing = []
    for res, optional in raw_pres:
        for item in res:
            if item not in all_words:
                if item not in missing:
                    missing.append(item)
        for term in optional:
            for item in term:
                if item not in all_words:
                    if item not in missing:
                        missing.append(item)

    with open(os.path.join(gener_path, "Missing.txt"), "w") as out_file:
        out_file.write("\n".join(missing + [""]))

    # generation
    all_gen = []
    for pres, opt in tqdm.tqdm(raw_pres):
        all_gen += pres_generate((pres, []), all_words, max_permut=4, repeat=2)
        for item in opt:
            all_gen += pres_generate((pres, item), all_words, max_permut=4, repeat=2)
    

    random.shuffle(all_gen)

    sample_num = len(all_gen)
    print("generated ", len(all_gen), " samples, start write to hard driver.")

    # create directory
    if not os.path.exists(os.path.join(gener_path, "data")):
        os.mkdir(os.path.join(gener_path, "data"))

    # write to the files
    for i in tqdm.tqdm(range(len(all_gen))):
        data, seg_label, txt_label = all_gen[i]
        file_pref = os.path.join(gener_path, 'data', 'sample_')+ str(i)
        #file_pref = "data/combined_prescription/rnn_segmentator_trian/sample_" + str(i)
        ap.export(file_pref + ".wav", data)
        with open(file_pref + ".seg", "w") as f:
            f.write(seg_label)
        with open(file_pref + ".txt", "w") as f:
            f.write("\t".join(txt_label) + "\n")
    with open(os.path.join(gener_path, "sample_num"), "w") as f:
        f.write(str(sample_num))
