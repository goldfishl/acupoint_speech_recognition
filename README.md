# Experiments on Acupoints(isolated words)
The purpose of these experiments is to train a model for classifying an audio recording that contains an isolated word to its corresponding label.

## Dataset

This dataset consists of 16,830 speech samples, collected from 287 contributors, specifically focusing on 418 acupuncture points. The total duration of the dataset is 10.47 hours.

Taking into account the size of our dataset, we partition it into training, validation, and test sets using a **balanced** 0.7:0.15:0.15 ratio. To ensure an even distribution, we randomly select samples from each class for every subset, maintaining a balanced representation of all classes.

## Results on AST/SSAST

Metrics on the test set

| $recall$ | $precision$ | $f1$ | $accuracy$ | $AP$ |
| -------- | -------- | -------- | -------- |  -------- | 
| 95.35%  | 96.01%  | 95.34%  | 95.35% | 97.15% |

## Results on TDNN

Metrics on the test set

| $recall$ | $precision$ | $f1$ | $accuracy$ | $AP$ |
| -------- | -------- | -------- | -------- |  -------- | 
| 88.99%  | 90.16%  | 88.67%  | 88.99% | 95.91% |

## VAD Experiments


# Experiments on Acupuncture prescription(sequence of isolated words)

The purpose of these experiments was to train a model to transcribe acupuncture prescription speech into text.

## Dataset

There are two datasets - one is generated and is referred to as the *generation* dataset, and the other is collected from crowdsourcing platforms and is referred to as the *real* dataset.

## Results on segmenter + classifier

The following two experiments use the same AST/SSAST classifier but different segmenters.

### rule-based segmenter

Metrics on the *generation* test set

| $recall$ | $precision$ | $f1$ |
| -------- | -------- | -------- |
| 91.23%  | 89.63%  | 90.42%  |

Metrics on the *real* test set

| $recall$ | $precision$ | $f1$ |
| -------- | -------- | -------- |
| 91.23%  | 89.63%  | 90.42%  |

### RNN-based segmenter

Metrics on the *generation* test set

| $recall$ | $precision$ | $f1$ | 
| -------- | -------- | -------- | 
| 53.49%  | 60.72%  | 58.2%  | 

Metrics on the *real* test set

| $recall$ | $precision$ | $f1$ | 
| -------- | -------- | -------- | 
| 0.032%  | 0.024%  | 0.021%  | 

## Wav2Vec2 + CTC + n-gram

Metrics on the *generation* test set

| Best Val WER | Test WER |
| --- | --- | 
| 93.30 | 96.35 |

## Whisper

Metrics on the *generation* test set

| Model Size | Parameters | Best Val WER | Test WER |
| --- | --- | --- | --- |
| tiny | 39M | 17.07 | 16.17 |
| base | 74M | 34.58 | 35.94 |
| small | 244M | 1.8 | 2.86 |

Metrics on the *generation* test set

| Model Size | Parameters |Test WER |
| --- | --- | --- |
| small | 244M | 48.86 |