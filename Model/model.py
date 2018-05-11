import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

class EncoderCNN(nn.Module):
   
    def __init__(self, embed_size):

        # initialize the pretrained CNN models
        super(EncoderCNN, self).__init__()
        resnet = models.resnet18(pretrained=True)
        #resnet = models.resnet34(pretrained=True)
        #resnet = models.resnet50(pretrained=True)
        #resnet = models.resnet101(pretrained=True)
        #resnet = models.resnet152(pretrained=True)
        #resnet = models.inception_v3(pretrained=True)
        
        modules = list(resnet.children())[:-1] # remove the fully connected layer
        
        # load specific function for CNN
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()
        
    def init_weights(self):
        
        # initialize the weights
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
        
    def forward(self, images):
        
        # embed the image feature vectors
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        
        return features
    
class DecoderRNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        
        # initialize DNN model with hyperparameters
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        
        # initialize weights
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths):
        
        # decode image feature vectors
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        
        outputs = self.linear(hiddens[0]) # generate caption
        
        return outputs
    
    # find captions for given val image features
    def sample(self, features, states=None):
     
        sampled_ids = []
        inputs = features.unsqueeze(1)
        
        for i in range(20): # max sentence length   
            # find caption
            hiddens, states = self.lstm(inputs, states)          
            outputs = self.linear(hiddens.squeeze(1))  
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1) 
        
        sampled_ids = torch.cat(sampled_ids, 0)                  
        
        return sampled_ids.squeeze()
