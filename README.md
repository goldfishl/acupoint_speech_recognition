# build the package

I not really sure if it's necessary.

```shell
python -m pip install -e ./
```
# usage

Let's infer a prescription audio as an example:

```python
from acpsr import Discriminator

model = Discriminator()
file_path = "./sample_data/sample_4738.wav"

## rule-based inference
model.inference(file_path, seg_method='rule')

## rnn-segmentor inference
model.inference(file_path, seg_method='rule')
```


# data and pre-trained model

It will automatically download all the required files including audio `.wav` data and pre-trained model `.pth` file.

# segmentator evaluation

Evaluating `rule` and `rnn` based segmentator.

The two evaluations together can take up to one-half hour, which may vary depending on your machine.

My rule-based evalution results are in [rule_based_eval.txt](./result/rule_based_eval.txt).

```
Precision: 0.896329
Recall:    0.912349
F1:        0.904268
```

My rnn-based evalution results are in [rnn_seg_eval.txt](./result/rnn_seg_eval.txt).

```
The average editing distance is 14.218649517684888
The average accuracy of editing distance is 0.9893145572964336
Precision: 0.927368
Recall:    0.927368
F1:        0.927368
```

Follow the steps below to conduct your own evaluation.

## Generating data {#generate}

First you need to generate combined prescription audio data, the default setting generates 9330 samples, this will take a while.

```shell
python -m acpsr generate
```

## rule-based segmentator evaluation 

```shell
python -m acpsr evaluate rule_seg > ./result/rule_based_eval.txt

cat ./result/rule_based_eval.txt
```
## rnn-based segmentator evaluation 

```shell
python -m acpsr evaluate rnn_seg > ./result/rnn_seg_eval.txt

cat ./result/rnn_seg_eval.txt
```

# train rnn segmentator from scratch

Training can take up to 37 minutes(on my machine with Tesla v100).

Model structure in [model.py](./acpsr/train/model.py#L12-L24).

The hyperparameters of the training are set at [train.py](./acpsr/train/train.py#L111-L115).

The `val loss` and `test loss` information for each epoch during training is shown in [rnn_seg_train.txt](./result/rnn_seg_train.txt)

## Generating data

Follow the [Generating data](#generate) part of the evaluation.

## training the model

Training takes the data generated above and loads it from the hard drive into memory, which can take a lot of time. This may require 13G or more of RAM.

```shell
python -m acpsr train rnn_seg > ./result/rnn_seg_train.txt

cat ./result/rnn_seg_train.txt 
```