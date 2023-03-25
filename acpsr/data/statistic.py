import os
import json
import torchaudio
import pandas as pd

def start_statistic():
    # Print header for acupoint dataset
    print('-----------------acupoint-----------------')

    # Set directory path and list of dataset splits
    dir_path = os.path.join('./data', 'acupoint')
    splits = ["train", "valid", "test"]

    # Initialize dictionaries to store data and time count for each split
    data = {}
    time_count = {}

    # Loop over each split and load data from JSON file
    for _split in splits:
        # Initialize data and time count for this split
        data[_split] = []
        time_count[_split] = 0
        with open(os.path.join(dir_path,_split+".json"), 'r') as fp:
            data_json = json.load(fp)
            # Add filenames to data list and increment time count for this split
            data[_split] += [sd["wav"][12:-4]+".wav" for sd in data_json]
            time_count[_split] += sum([sd["length"] for sd in data_json])

    # Load CSV file into a Pandas dataframe
    df = pd.read_csv(os.path.join(dir_path, 'info.csv'))
    # Filter out rows with quality=0 or 1
    df = df[df['quality'].isin([0, 1])]
    # Remove numbers from end of "sentence" column
    df['sentence'] = df['sentence'].str.replace('\d+$', '', regex=True)
    # Count unique userIDs and sentences
    num_unique_userIDs = df['userID'].nunique()
    num_unique_sentences = df['sentence'].nunique()

    # Calculate total data count and print dataset split rates
    total_data = sum([len(data[_split]) for _split in splits])
    print(f'dataset split rate: {len(data["train"]) / total_data:.2f} {len(data["valid"]) / total_data:.2f} {len(data["test"]) / total_data:.2f}')
    # Print data count and time for each split
    for _split in splits:
        print(f'{_split} data count: {len(data[_split])}')
        print(f'{_split} time: {time_count[_split]:.2f} seconds, {time_count[_split]/3600:.2f} hours')
    # Print total data count and time, and individual and acupoint counts
    print(f'total data count: {total_data}')
    print(f'total time: {sum([time_count[_split] for _split in splits]):.2f} seconds, {sum([time_count[_split] for _split in splits])/3600:.2f} hours')
    print(f'individual count: {num_unique_userIDs}')
    print(f'acupoint count: {num_unique_sentences}')

    # Print header for combined_prescription dataset
    print('-----------------combined_prescription-----------------')

    # Set directory path and list of dataset splits
    dir_path = os.path.join('./data', 'combined_prescription')
    splits = ["training", "validation", "testing"]

    # Initialize dictionary to store data and time count for each split
    data = {}
    time_count = {}

    # Loop over each split and load data from text file
    for _split in splits:
        data[_split] = []
        with open(os.path.join(dir_path, _split+"_list.txt"), 'r') as fp:
            for line in fp:
                data[_split].append(line.strip())

        # Compute time count for this split by loading each audio file and summing their durations
        time_count[_split] = 0
        for _data in data[_split]:
            waveform, sample_rate = torchaudio.load(os.path.join(dir_path, 'data', _data))
            time_count[_split] += waveform.shape[1] / sample_rate
    
    # Calculate total data count and print dataset split rates
    total_data = sum([len(data[_split]) for _split in splits])
    print(f'dataset split rate: {len(data["training"]) / total_data:.2f} {len(data["validation"]) / total_data:.2f} {len(data["testing"]) / total_data:.2f}')

    # Print data count and time for each split
    for _split in splits:
        print(f'{_split} data count: {len(data[_split])}')
        print(f'{_split} time: {time_count[_split]:.2f} seconds, {time_count[_split]/3600:.2f} hours')
    
    # Print total data count and time
    print(f'total data count: {total_data}')
    print(f'total time: {sum([time_count[_split] for _split in splits]):.2f} seconds, {sum([time_count[_split] for _split in splits])/3600:.2f} hours')