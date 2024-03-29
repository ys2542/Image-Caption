import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torch.autograd import Variable 
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
from os import listdir
from os.path import isfile, join

def to_var(x, volatile=False): # wrap the tensor
    if torch.cuda.is_available():
        x = x.cuda()
        
    return Variable(x, volatile=volatile)

def load_image(image_path, transform=None): # load the image and resize it
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image
    
def main(args):
    # Val images folder
    filepath = '/scratch/ys2542/pytorch-tutorial/tutorials/03-advanced/image_captioning/data/resizedval2014'
    onlyfiles = [fl for fl in listdir(filepath) if isfile(join(filepath, fl))]

    # image preprocessing
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    # load vocabulary wrapper pickle file 
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    encoder = EncoderCNN(args.embed_size) # build encoder
    encoder.eval() # evaluation mode by moving mean and variance
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers) # build decoder
    
    # load the trained CNN and RNN parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))
    
    # Load all images in val folder
    for i in onlyfiles:
 
        badsize = 0 # count the unload images
        args_image = filepath + '/' # val folder path with image names
        args_image = args_image + i

        # transform image and wrap it to tensor
        image = load_image(args_image, transform)
        image_tensor = to_var(image, volatile=True)
    
        if torch.cuda.is_available(): # load GPU
            encoder.cuda()
            decoder.cuda()
    
            # generate caption from image
            try:
                feature = encoder(image_tensor)
                sampled_ids = decoder.sample(feature)
                sampled_ids = sampled_ids.cpu().data.numpy()
    
                # decode word_ids to words
                sampled_caption = []
                for word_id in sampled_ids:
                    word = vocab.idx2word[word_id]
                    sampled_caption.append(word)
                    if word == '<end>':
                        break
                sentence = ' '.join(sampled_caption)
    
                # print out image and generated caption without start and end
                print ('beam_size_1' + '\t' + i + '\t' + sentence[8:-8])

            except:
                badsize = badsize + 1 # count some wrong images
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default='./models/encoder-5-3000.pkl',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./models/decoder-5-3000.pkl',
                        help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='/scratch/ys2542/pytorch-tutorial/tutorials/03-advanced/image_captioning/data/vocab.pkl',
                        help='path for vocabulary wrapper')
    
    # same hyperparameter with train
    parser.add_argument('--embed_size', type=int , default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
