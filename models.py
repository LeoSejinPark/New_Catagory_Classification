import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        # dist_a = F.pairwise_distance(E1, E2, 2)
        # dist_b = F.pairwise_distance(E1, E3, 2)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class TripletNet(nn.Module):
    def __init__(self, embeddingNet):
        super(TripletNet, self).__init__()
        self.embeddingNet = embeddingNet

    def forward(self, i1, i2, i3):
        E1 = self.embeddingNet(i1)
        E2 = self.embeddingNet(i2)
        E3 = self.embeddingNet(i3)
        # dist_a = F.pairwise_distance(E1, E2, 2)
        # dist_b = F.pairwise_distance(E1, E3, 2)
        return E1, E2, E3

class CBoWClassifier(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, dropout_p,
                 embedding_weights, embedding_trainable):
        super(CBoWClassifier, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        # layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        if embedding_weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
        if not embedding_trainable:
            self.embedding.weight.requires_grad = False
        self.fc = nn.Linear(embedding_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, input):
        # |input| = (batch_size, max_seq_length)
        batch_size = input.size(0)

        embeds = self.embedding(input)
        # |embeds| = (bathc_size, max_seq_length, embedding_size)
        mean_embeds = torch.mean(embeds, dim=1)
        # |mean_embeds| = (batch_size, embedding_size)
        
        fc_out = self.dropout(self.relu(self.fc(mean_embeds)))
        # |fc_out| = (batch_size, hidden_size)
        output = self.softmax(self.fc2(fc_out))
        # |output| = (batch_size, output_size)
        
        return output
    
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, n_layers, dropout_p, 
                 embedding_weights, embedding_trainable):
        super(LSTMClassifier, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        # layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        if embedding_weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
        if not embedding_trainable:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            num_layers=n_layers,
                            dropout=dropout_p,
                            bidirectional=True,
                            batch_first=True)
        self.fc = nn.Linear(self.hidden_size*2, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, input):
        # |input| = (batch_size, max_seq_length)
        batch_size = input.size(0)

        embeds = self.embedding(input)
        # |embeds| = (bathc_size, max_seq_length, embedding_size)
      
        lstm_out, hidden = self.lstm(embeds)
        # If bidirectional=True, num_directions is 2, else it is 1.
        # |lstm_out| = (batch_size, max_seq_length, num_directions*hidden_size)
        # |hidden[0]|, |hidden[1]| = (num_layers*num_directions, batch_size, hidden_size)
        
        mean_lstm_out = torch.mean(lstm_out, dim=1)
        # |lstm_out| = (batch_size, hidden_size*2)
        output = self.softmax(self.fc(mean_lstm_out))
        # |output| = (batch_size, output_size)

        return output
   
class LSTM_CNN(nn.Module):
    def __init__(self, input_size, embedding_size,hidden_size,  class_num=41,kernel_num=100,kernel_sizes=[3,4,5],lstm_drop_rate=0.5, cnn_drop_rate=0.5,n_layers=2):
        super(LSTM_CNN, self).__init__()
        

        #D = embedding_size
        D = hidden_size*n_layers
        C = class_num
        Ci = 1
        Co = kernel_num
        Ks = kernel_sizes
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(input_size, embedding_size)
        
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            num_layers=n_layers,
                            dropout=lstm_drop_rate,
                            bidirectional=True,
                            batch_first=True)
        
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(cnn_drop_rate)
        
        self.fc1 = nn.Linear(len(Ks)*Co, C)
        self.softmax = nn.LogSoftmax(dim=-1)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    
    def forward(self, x):
        x = self.embed(x)  # (N, W, D) # W is num_features
        x, (final_hidden_state, final_cell_state) = self.lstm(x)
        
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
  
        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.softmax(self.fc1(x))  # (N, C)
        return logit
    
   
class LSTM_Attention(nn.Module):
    
    def __init__(self, input_size, embedding_size,hidden_size,  class_num=41,lstm_drop_rate=0.5, fc_drop_rate = 0.4, n_layers=2):
        super(LSTM_Attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            num_layers=n_layers,
                            dropout=lstm_drop_rate,
                            bidirectional=True,
                            batch_first=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(fc_drop_rate)
        self.fc = nn.Linear(hidden_size*n_layers, class_num)
        self.softmax = nn.LogSoftmax(dim=-1)
   
    def attention_net(self, lstm_output, final_hidden_state):
        hidden = final_hidden_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
		
        return new_hidden_state
    
    def forward(self, x):
        x = self.embed(x)
        x, (final_hidden_state, final_cell_state) = self.lstm(x)
        sum_hidden = torch.cat((final_hidden_state[0],final_hidden_state[1]),1)
        sum_hidden1 = torch.cat((final_hidden_state[2],final_hidden_state[3]),1)
        sum_hidden_3d = torch.Tensor([sum_hidden.cpu().detach().numpy(),sum_hidden1.cpu().detach().numpy()])
        mean_hiddem_3d = torch.mean(sum_hidden_3d, dim=0).unsqueeze(0)
        mean_hiddem_3d = mean_hiddem_3d.to(torch.device('cuda'))
        x = self.attention_net(x,mean_hiddem_3d)
        x = self.relu(x)
        x = self.dropout(x) 
        logit = self.softmax(self.fc(x))
        return logit