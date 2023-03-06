import os
import json
import torchaudio

def start_statistic():
    print('-----------------acupoint-----------------')
    dir_path = os.path.join('./data', 'acupoint')
    splits = ["train", "valid", "test"]
    data = {}
    time_count = {}
    for _split in splits:
        data[_split] = []
        time_count[_split] = 0
        with open(os.path.join(dir_path,_split+".json"), 'r') as fp:
            data_json = json.load(fp)
            data[_split] += [sd["wav"][12:-4]+".wav" for sd in data_json]
            time_count[_split] += sum([sd["length"] for sd in data_json])
        

    total_data = sum([len(data[_split]) for _split in splits])
    print(f'dataset split rate: {len(data["train"]) / total_data:.2f} {len(data["valid"]) / total_data:.2f} {len(data["test"]) / total_data:.2f}')
    for _split in splits:
        print(f'{_split} data count: {len(data[_split])}')
        print(f'{_split} time: {time_count[_split]:.2f} seconds, {time_count[_split]/3600:.2f} hours')
    print(f'total data count: {total_data}')
    print(f'total time: {sum([time_count[_split] for _split in splits]):.2f} seconds, {sum([time_count[_split] for _split in splits])/3600:.2f} hours')

    print('-----------------combined_prescription-----------------')
    dir_path = os.path.join('./data', 'combined_prescription')
    splits = ["training", "validation", "testing"]
    data = {}
    for _split in splits:
        data[_split] = []
        with open(os.path.join(dir_path, _split+"_list.txt"), 'r') as fp:
            for line in fp:
                data[_split].append(line.strip())

        time_count[_split] = 0
        for _data in data[_split]:
            waveform, sample_rate = torchaudio.load(os.path.join(dir_path, 'data',_data))
            time_count[_split] += waveform.shape[1] / sample_rate
    
    total_data = sum([len(data[_split]) for _split in splits])
    print(f'dataset split rate: {len(data["training"]) / total_data:.2f} {len(data["validation"]) / total_data:.2f} {len(data["testing"]) / total_data:.2f}')
    for _split in splits:
        print(f'{_split} data count: {len(data[_split])}')
        print(f'{_split} time: {time_count[_split]:.2f} seconds, {time_count[_split]/3600:.2f} hours')
    print(f'total data count: {total_data}')
    print(f'total time: {sum([time_count[_split] for _split in splits]):.2f} seconds, {sum([time_count[_split] for _split in splits])/3600:.2f} hours')
    


def list_data(dir_path, splits):
    data = []
