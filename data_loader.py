import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO

class CocoDataset(data.Dataset):
    
    # initialize path and captions and vocabulary wrapper file
    def __init__(self, root, json, vocab, transform=None):
        
        self.root = root # image directory.
        self.coco = COCO(json) # coco annotation file path
        self.ids = list(self.coco.anns.keys()) # file names
        self.vocab = vocab # vocabulary wrapper file
        self.transform = transform # image transformer

    # get both image id and its caption
    def __getitem__(self, index):
        
        coco = self.coco # coco annotation file path
        vocab = self.vocab # vocabulary wrapper file
        ann_id = self.ids[index] # file names
        caption = coco.anns[ann_id]['caption'] # load caption
        img_id = coco.anns[ann_id]['image_id'] # load image id
        path = coco.loadImgs(img_id)[0]['file_name'] # load image location path

        # transform images
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # convert caption in string format to word ids
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target
    
    # get size of file name
    def __len__(self):
        return len(self.ids)

# create mini-batch tensors from the list of tuples including image and caption    
def collate_fn(data):
    
    # sort a data list by caption length in descending order
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # merge images from tuple of 3D tensor to 4D tensor
    images = torch.stack(images, 0)

    # merge captions from tuple of 1D tensor to 2D tensor
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    
    # get torch tensor of shape including batch_size and padded_length
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]  
        
    return images, targets, lengths

# use DataLoader in torch.utils.data for dataset
def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
 
    # load COCO caption dataset
    coco = CocoDataset(root=root, json=json, vocab=vocab, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    
    return data_loader
