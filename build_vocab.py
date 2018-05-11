import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO

class Vocabulary(object):
    
    # initialize vocabulary object for each caption
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    # add caption
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    # get caption
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
    
    # get size of caption
    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold):
    
     # write a wrapper pickle file
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
        if i % 1000 == 0:
            print("[%d/%d] Tokenized the captions." %(i, len(ids)))

    words = [word for word, cnt in counter.items() if cnt >= threshold] # keep work count bigger than threshold number

    # add special tokens
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(args):
    vocab = build_vocab(json=args.caption_path, threshold=args.threshold) # load caption file
    vocab_path = args.vocab_path # path to store pickle file
    with open(vocab_path, 'wb') as f: # write to pickle file
        pickle.dump(vocab, f)
    print("Total vocabulary size: %d" %len(vocab))
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, default='./data/annotations/captions_train2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', 
                        help='path for saving vocabulary wrapper file')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold number')
    args = parser.parse_args()
    main(args)
