#from acpsr.model import audio_proc
#from acpsr.model import inference
#from acpsr.model import ast_models

from acpsr.download import fetch_content

import os
import sys


if not os.path.exists('./models'):
    sys.stderr.write('create `models` directory!')
    os.mkdir('./models')
if not os.path.exists('./models/audio_model.pth'):
    fetch_content('audio_model.pth', './models')
if not os.path.exists('./models/class_labels_indices.csv'):
    fetch_content('class_labels_indices.csv', './models')
if not os.path.exists('./models/rnn_segmentator.pth'):
    fetch_content('rnn_segmentator.pth', './models')