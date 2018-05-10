# Image Caption

Yulin Shen \
Shenghui Zhou \
Yiyan Chen 

## Getting Started

MSCOCO Dataset

### Prerequisites

Python 3.6 for train and output \
Python 2.7 for eval(API use version2.7) \
Pytorch


## Example

LSTM18.txt is the output(resnet18 + LSTM) \
sort_caption.txt is the annotation for val image set, I extract it for eval. \
eval_lstm18.txt is the bleu scores.

### Command

```bash
$ python build_vocab.py   
$ python resize.py
$ python train.py
$ python sample.py
$ python create_json_references.py -i sort_caption.txt -o sort_caption.json
$ python run_evaluations.py -i LSTM18.txt -r sort_caption.json
```
