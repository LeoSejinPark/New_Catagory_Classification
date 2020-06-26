import argparse
import pickle

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from tokenization import Vocab, Tokenizer
from dataset_utils import Corpus
from models import CBoWClassifier, LSTMClassifier, TripletNet, LSTM_CNN, LSTM_Attention
from losses import TripletLoss, OnlineTripletLoss
from utils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector
from easydict import EasyDict as edict
import torch.nn.functional as F
from sklearn.utils import shuffle
#import nltk
#nltk.download('punkt')

TOKENIZER = ('treebank', 'mecab')
MODEL = ('cbow', 'lstm','lstm_cnn','lstm_attention')

class OnlineTestTriplet(nn.Module):
    def __init__(self, marg, triplet_selector):
        super(OnlineTestTriplet, self).__init__()
        self.marg = marg
        self.triplet_selector = triplet_selector
    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)
        return triplets

def argparser():
    p = argparse.ArgumentParser()

    # Required parameters
    p.add_argument('--train_corpus', default=None, type=str, required=True)
    p.add_argument('--valid_corpus', default=None, type=str, required=True)
    p.add_argument('--vocab', default=None, type=str, required=True)
    p.add_argument('--model_type', default=None, type=str, required=True,
                   help='Model type selected in the list: ' + ', '.join(MODEL))

    # Input parameters
    p.add_argument('--is_sentence', action='store_true',
                   help='Whether the corpus is already split into sentences')
    p.add_argument('--tokenizer', default='treebank', type=str,
                   help='Tokenizer used for input corpus tokenization: ' + ', '.join(TOKENIZER))
    p.add_argument('--max_seq_length', default=64, type=int,
                   help='The maximum total input sequence length after tokenization')

    # Train parameters
    p.add_argument('--cuda', default=True, type=bool,
                   help='Whether CUDA is currently available')
    p.add_argument('--epochs', default=30, type=int,
                   help='Total number of training epochs to perform')
    p.add_argument('--batch_size', default=128, type=int,
                   help='Batch size for training')
    p.add_argument('--learning_rate', default=5e-3, type=float,
                   help='Initial learning rate')
    p.add_argument('--shuffle', default=True, type=bool, 
                   help='Whether to reshuffle at every epoch')
    
    # Model parameters
    p.add_argument('--embedding_trainable', action='store_true',
                   help='Whether to fine-tune embedding layer')
    p.add_argument('--embedding_size', default=100, type=int,
                   help='Word embedding vector dimension')
    p.add_argument('--hidden_size', default=128, type=int,
                   help='Hidden size')
    p.add_argument('--dropout_p', default=.5, type=float,
                   help='Dropout rate used for dropout layer')
    p.add_argument('--n_layers', default=2, type=int,
                   help='Number of layers in LSTM')
    
    p.add_argument('--gpu_num', default=0, type=int,
                   help='GPU Number')
    
    p.add_argument('--weight_decay', default=0.0001, type=float,
                   help='weight decay rate')

    config = p.parse_args()
    return config

def train():
    n_batches, n_samples = len(train_loader), len(train_loader.dataset)

    model.train()
    losses, accs = 0, 0
    for iter_, batch in enumerate(train_loader):
        inputs, targets = batch
        # |inputs|, |targets| = (batch_size, max_seq_length), (batch_size)
        
        preds = model(inputs)
        # |preds| = (batch_size, n_classes)

        loss = loss_fn(preds, targets) #F.cross_entropy(preds, targets) 
        losses += loss.item()
        acc = (preds.argmax(dim=-1) == targets).sum()
        accs += acc.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iter_ % (n_batches//5) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.4f} \tAccuracy: {:.3f}%'.format(
                    epoch, iter_, n_batches, 100.*iter_/n_batches, loss.item(), 100.*acc.item()/config.batch_size))

    print('====> Train Epoch: {} Average loss: {:.4f} \tAccuracy: {:.3f}%'.format(
            epoch, losses/n_batches, 100.*accs/n_samples))

def validate():
    n_batches, n_samples = len(valid_loader), len(valid_loader.dataset)

    model.eval()
    losses, accs = 0, 0
    with torch.no_grad():
        for iter_, batch in enumerate(valid_loader):
            inputs, targets = batch

            preds = model(inputs)

            loss = loss_fn(preds, targets)
            losses += loss.item()
            acc = (preds.argmax(dim=-1) == targets).sum()
            accs += acc.item()

    print('====> Validate Epoch: {} Average loss: {:.4f} \tAccuracy: {:.3f}%'.format(
            epoch, losses/n_batches, 100.*accs/n_samples))
    

if __name__=='__main__':
    #config = argparser()
    config = edict({"train_corpus":  "corpus/corpus.train.txt", "valid_corpus": "corpus/corpus.valid.txt" , "vocab": "vocab.train.pkl" , "model_type":"lstm_attention" , "epochs": 30,  "learning_rate": 0.001,'tokenizer':'treebank', 'cuda':True,'is_sentence':'store_true','max_seq_length':64,'batch_size':128,'shuffle':True ,'embedding_size':100,'hidden_size':128,'dropout_p':.5,'n_layers':2,'embedding_trainable': 'store_true','gpu_num':0,'weight_decay':0.00001})

    print(config)

    # Load vocabulary
    with open(config.vocab, 'rb') as reader:
        vocab = pickle.load(reader)

    # Select tokenizer
    config.tokenizer = config.tokenizer.lower()
    if config.tokenizer==TOKENIZER[0]:
        from nltk.tokenize import word_tokenize
        tokenization_fn = word_tokenize
    elif config.tokenizer ==TOKENIZER[1]:
        from konlpy.tag import Mecab
        tokenization_fn = Mecab().morphs
        
    tokenizer = Tokenizer(tokenization_fn=tokenization_fn, vocab=vocab,
                          is_sentence=config.is_sentence, max_seq_length=config.max_seq_length)

    # Build dataloader
    train_dataset = Corpus(corpus_path=config.train_corpus, tokenizer=tokenizer, cuda=config.cuda)
    valid_dataset = Corpus(corpus_path=config.valid_corpus, tokenizer=tokenizer, cuda=config.cuda)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=config.shuffle)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size, shuffle=config.shuffle)
    
    # Build Model : CBoW, LSTM
    config.model_type = config.model_type.lower()
    if config.model_type==MODEL[0]:
        model = CBoWClassifier(input_size=len(vocab), # n_tokens
                               embedding_size =config.embedding_size,
                               hidden_size=config.hidden_size,
                               output_size=len(train_dataset.ltoi), # n_classes
                               dropout_p=config.dropout_p,
                               embedding_weights=vocab.embedding_weights,
                               embedding_trainable=config.embedding_trainable)
    
    elif config.model_type==MODEL[1]:
        
        model = LSTMClassifier(input_size=len(vocab), # n_tokens
                               embedding_size=config.embedding_size,
                               hidden_size=config.hidden_size,
                               output_size=len(train_dataset.ltoi)*2, # n_classes
                               n_layers=config.n_layers,
                               dropout_p=config.dropout_p,
                               embedding_weights=vocab.embedding_weights,
                               embedding_trainable=config.embedding_trainable)
        
    elif config.model_type==MODEL[2]:   
        model= LSTM_CNN(input_size=len(vocab), # n_tokens
                            embedding_size=config.embedding_size,
                            hidden_size=config.hidden_size,
                            lstm_drop_rate=0.5, cnn_drop_rate=0.5,
                            class_num=len(train_dataset.ltoi))
        
    elif config.model_type==MODEL[3]:
        model= LSTM_Attention(input_size=len(vocab), # n_tokens
                            embedding_size=config.embedding_size,
                            hidden_size=config.hidden_size,
                            lstm_drop_rate=0.5, fc_drop_rate=0.5,
                            class_num=len(train_dataset.ltoi))
    
    #loss_fn = TripletLoss(1)#nn.NLLLoss()
    marg = 1
    trip_loss_fun = torch.nn.TripletMarginLoss(margin=marg, p=2)
    triplet_selector2 = AllTripletSelector()
    TripSel = OnlineTestTriplet(marg, triplet_selector2)
    margin = 1
    loss_fn = nn.NLLLoss()
    #loss_fn = OnlineTripletLoss(margin,  AllTripletSelector())#RandomNegativeTripletSelector(margin))
    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr= config.learning_rate, weight_decay =config.weight_decay)
    
    if config.cuda:
        torch.cuda.set_device(config.gpu_num)
        torch.cuda.empty_cache()
        model = model.cuda()
       
        loss_fn = loss_fn.cuda()
        
    print('=========model=========\n',model)
    print('=========optimizer=========\n',optimizer)
    # Train & validate
    for epoch in range(config.epochs):
        train()
        validate()
      
    
    # Save model
    torch.save(model.state_dict(), '{}.pth'.format(config.model_type))