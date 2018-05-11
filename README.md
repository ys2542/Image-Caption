# Image Caption

Yulin Shen \
Shenghui Zhou \
Yiyan Chen 

## Prerequisites

Python 3.6 for Model \
Python 2.7 for Evaluation(COCOAPI use version2.7) \
Pytorch

## Folder Description

Model: all python files to do image caption \
Evaluation: evaluate the result produced by Model to get bleu scores \
Score: 9 combinations of models bleu scores outputs \
Result: predicted captions of 9 combinations of models using val image set \
Train: loss and perplexity in the train process in 9 combinations of models

## Getting Started

In Model folder: 

Step 1: get COCO Dataset at first
```bash
$ ./data.sh   
```
Step 2: get annation wrapper pickle file
```bash
$ python build_vocab.py 
```
Step 3: resize all images in train image set
```bash
$ python resize.py 
```
Step 4: train the model
```bash
$ python train.py 
```
Step 5: get predicted result of val image set
```bash
$ python sample.py 
```
Notice: change the paths(model setting pickle file, annation pickle file, image sets folder) in each file 

In Evaluation foler: 

Step 1: get COCOAPI
```bash
$ ./COCOAPI.sh 
```
Step 2: convert sort_caption.txt into a new annotation json to fit our evaluation format 
```bash
$ python create_json_references.py -i ./sort_caption.txt -o ./sort_caption.json 
```
Step 3: choose a result txt file in Result folder to get its bleu score
```bash
$ python run_evaluations.py -i ../Result/LSTM152_Result.txt -r ./sort_caption.json
```
Notice: 

Thanks for [vsubhashin](https://github.com/vsubhashini/caption-eval) sharing us a general evaulation tools. We just need to convert the old annotation json file into a new txt file with image name and its caption.

I have already extracted all necessray information in the val annotation json file to a new txt file called sort_caption.txt, if you need to know how to convert it, please check convert.py in the Evaluation folder for reference.




