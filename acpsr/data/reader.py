import os
from acpsr.model import audio_proc as ap
import json


def get_all_subdir(one_dir):
    __all_dir = []
    for path in os.listdir(one_dir):
        # check if current path is a dir
        if os.path.isdir(os.path.join(one_dir, path)):
            __all_dir.append(os.path.join(one_dir, path))
    return __all_dir


def get_whitelist(dir_path, include_train=False):
    # whitelist based on VAD 
    # whitelist = [line.strip().split(",") for line in list(open(filename, "r"))]
    # whitelist = ["/".join(line[:2] + [line[3]])+".wav" for line in whitelist if line[4] == "0"]
    data_set = ["valid", "test"] 
    if include_train:
       data_set += "train"
    whitelist = []
    for split in ["valid", "test"]:
        with open(os.path.join(dir_path,split+".json"), 'r') as fp:
            data_json = json.load(fp)
            whitelist += [sd["wav"][12:-4]+".wav" for sd in data_json]
    print(whitelist)
    return whitelist


# arguments:
#   `dir_path` must end with '/' !
def get_all_wav(dir_path=r'data/acupoint/', max_level=2, 
            whitelist="vad_record.csv", include_train=False):
    if whitelist is not None:
        whitelist = get_whitelist(dir_path, include_train)

    # Returns tuples in a list
    # Each tuple contains: (tag, speaker, filename)
    dir_paths = [dir_path]
    all_dir = []
    res = []
    # Iterate through max_levels of directories
    for level in range(max_level):
        for one_dir in dir_paths:
            all_dir += get_all_subdir(one_dir)

        dir_paths = all_dir
        all_dir = []

    for one_dir in dir_paths:
        for path in os.listdir(one_dir):
            filename = os.path.join(one_dir, path)
            if os.path.isfile(filename) and ".wav" in filename:
                speaker = "_".join(filename.replace(".wav", "").split("/")[1:-1])
                tag = filename.replace(".wav", "").split("/")[-1]
                tag = tag.replace("_p", "")  # idk why some file names are like this
                if isinstance(whitelist, list) and filename[len(dir_path):] not in whitelist:
                    continue
                res.append((tag, speaker, filename))
    all_words = {}
    for tag, speaker, file in res:
        if tag not in all_words:
            all_words[tag] = []
        all_words[tag].append((speaker, file))
    return all_words


def get_raw_pres(filename="data/combined_prescription/Cleaned.txt"):
    entry = []
    optional = []
    entries = []
    with open(filename, "r") as infile:
        for line in infile:
            if line[0] == "#":
                continue
            if line[0] == "\n" and len(entry) != 0:
                entries.append([entry, optional])
                entry = []
                optional = []
            elif line[0] == "\n":
                continue
            elif line[0] == " ":
                optional.append(line.replace(" ", "").strip().split("、"))
            else:
                entry = line.strip().split("、")
                if entry == [""]:
                    print("Warning, line:", line)
    return entries

