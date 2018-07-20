import numpy as np
import os 
import os.path as path
import matplotlib.pyplot as plt
from PIL import Image as im

import torch as T
import torch.nn as nn
import torchvision.models as models

from torch.autograd import Variable
import torch.optim as optim

#PARAMETERS
TrainOrNot = False
###-----Text Batch-----###
batch_size	= 1
num_steps	= 5
###----------------------- 
HIDDEN_SIZE = 100
NUM_LAYERS = 1

EPOCH_ = 200
LR = 0.001
FILENAME = './save/rnn_test.pkl'



# -------------------------------------------------------------#
# CAL TIME PASS
# -------------------------------------------------------------#
import time, math

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def get_axis(matrix):
    e=matrix
    lis = []
    print (e)
    for i in range(5):
        lis.append(e[i][1:])
    return np.matrix(lis)
    

if os.path.exists('./save') == 0:
    os.makedirs('./save')

    
# -------------------------------------------------------------#
# Load data and proprocessing
# -------------------------------------------------------------#

data_URL = 'shakespeare_train.txt'
Val_URL = 'shakespeare_valid.txt'
with open(data_URL, 'r') as f:
    text = f.read()
    
with open(Val_URL, 'r') as f2:
    text2 = f2.read()
#text = 'NCTU is good.'
# -------------------------------------------------------------#
# Text encode
# -------------------------------------------------------------#

# Characters' collection
vocab = set(text)
N_CHARACTERS = len(vocab)
print ('N_CHARACTERS',N_CHARACTERS)

# Construct character dictionary
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))

# Encode data, shape = [# of characters]
train_encode = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
valid_encode = np.array([vocab_to_int[c] for c in text2], dtype=np.int32)


# -------------------------------------------------------------#
# Divide data into mini-batches
# -------------------------------------------------------------#
def get_batches(arr, n_seqs, n_steps):
    
    '''
    arr: data to be divided
    n_seqs: batch-size, # of input sequences
    n_steps: timestep, # of characters in a input sequences
    '''
    
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    arr = arr[:batch_size * n_batches]
    arr = arr.reshape((n_seqs, -1))
    
    for n in range(0, arr.shape[1], n_steps):
        #print (arr.shape[1],n_steps,arr.shape[1]/n_steps)
        #exit()
        x = arr[:, n:n+n_steps]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y


        
# Function above define a generator, call next() to get one mini-batch

train_batches = get_batches(train_encode, batch_size, num_steps)
valid_batches = get_batches(valid_encode, batch_size, num_steps)
x, y = next(train_batches)
# print (len(list(train_batches)))

# print (np.array(x).reshape(-1).shape)
# print ('----------')
# print (y)
# exit()

print ('Data length:',len(text))
    
###-----NET-----###
#decoder = T.nn.RNN(N_CHARACTERS, HIDDEN_SIZE, NUM_LAYERS)
#hidden = T.zeros(NUM_LAYERS, HIDDEN_SIZE)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        #print (self.input_size)
        self.encoder = nn.Embedding(input_size, hidden_size)
        #print (self.encoder)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers,batch_first = True)
        self.decoder = nn.Linear(hidden_size, output_size)
        #print (self.gru)
        #print (self.decoder)
    def forward(self, input, hidden):
        #print (input.shape)
        #exit()
        input = self.encoder(input)
        #print (input.size())
        output, hidden = self.lstm(input)
        #print (hidden.size())
        #print (output.size())
        
        output = self.decoder(output.contiguous().view(-1,self.hidden_size))
        output = output.contiguous().view(batch_size,num_steps,N_CHARACTERS)
        #print (output.size())
        #exit()
        return output, hidden

    def init_hidden(self):
        return Variable(T.zeros(self.n_layers, batch_size, self.hidden_size))



decoder = RNN(N_CHARACTERS, HIDDEN_SIZE, N_CHARACTERS)
#print ('hidden',hidden.shape)
#decoder = T.nn.DataParallel(decoder, device_ids=[0,1,2,3]).cuda()
decoder.cuda()

###-----OPTIMIZER-----###
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(decoder.parameters(), lr = LR)

###-----Start Training-----###
if TrainOrNot == True:
    print ('-----Start Training-----')
    BAT = len(list(train_batches))
    BAT_val = len(list(valid_batches))
    start = time.time()
    all_losses = []
    valid_loss = []
    loss_avg = 0
    plot_every = 1
    #print (decoder)
    # u= 0
    # for x, y in get_batches(train_encode, batch_size, num_steps):
        # u+=1
        # print ('u:',u)
        # print ('x:',x.shape)
        # print ('y:',y.shape)
    # exit()
    
    for epoch in range(EPOCH_):
        hidden = decoder.init_hidden()
        
        #print (hidden.size())
        print ('epoch:',epoch+1)
        correct = 0
        #old = Variable(T.ones(batch_size*num_steps)).cuda()
        for x, y in get_batches(train_encode, batch_size, num_steps):
            
            x_ = Variable(T.from_numpy(x).long()).cuda() #50,100
            y_ = Variable(T.from_numpy(y).long()).cuda() #50,100
            #print (x_.size())
            #exit()
            decoder.zero_grad()
            output, hidden = decoder(x_, hidden)  #(50,100,67) (1,50,100)
            
            
            output = output.view(-1,N_CHARACTERS) #5000,67
            #print ('---------',output.shape)
            y_ = y_.view(-1)                      #5000
            
            ###Cal Accuracy
            # out_ = T.argmax(output,dim=1).reshape(200,50)
            # pred_ = y_.reshape(200,50)
            
            # acc = T.sum(out_ == pred_).cpu().numpy()/(batch_size*num_steps)
            #T.equal(out_,pred_).sum()
            #print (x_.reshape(200,50).cpu().numpy())
            #print (out_.cpu().numpy())
            #print (pred_.cpu().numpy())
            #print (T.sum(out_ == pred_))
            
            #print ('---------------------------------------')
            #get_axis()
            
            loss = criterion(output, y_)
            loss.backward()
            optimizer.step()
            
        runningloss = loss.item()
        print('[%d/%d] Epoch, Loss: %.4f' % (epoch, EPOCH_,runningloss))
        
        
            
        if epoch % plot_every == 0:
            T.save(decoder,FILENAME)
            all_losses.append(runningloss)
            #all_losses = np.array(all_losses)
            np.save('./train_lstm.npy',all_losses)
            
            ###validation
            print ('-----Start validating-----')
            valbatchloss = 0
            NN = T.load(FILENAME)
            
            hidden_ = NN.init_hidden()
            
            for x, y in get_batches(valid_encode, batch_size, num_steps):
                x_ = Variable(T.from_numpy(x).long()).cuda()
                y_ = Variable(T.from_numpy(y).long()).cuda()
                
                NN.zero_grad()
                output_, hidden_ = NN(x_, hidden_)
                output_ = output_.view(-1,N_CHARACTERS)
                
                y_ = y_.view(-1)                      #5000
            
                loss_ = criterion(output_, y_)
                loss_.backward()
                
            valid_loss.append(loss_.item())
            np.save('./valid_lstm.npy',valid_loss)
            print ('Validation loss:%.4f'%(loss_.item()))
            
# print ('-----Start generating-----')
# prime = 'ROMEO'
# generator = np.array([vocab_to_int[c] for c in prime], dtype=np.int32)
# generator = np.expand_dims(generator, axis=0)
# net = T.load(FILENAME).cuda()
# #print (generator[0])
# for i in generator[0]:
    # print (int_to_vocab[i])
# hidden_ = net.init_hidden()
# x_ = Variable(T.from_numpy(generator).long()).cuda()
# print (int_to_vocab)
# pattern = []
# for i in range(100):
    
    # #print (x_.size())
    # out, hidden_ = net(x_,hidden_)
    # #print (out.size())
    # out = out.view(-1,N_CHARACTERS)
    # sentence = T.argmax(out,dim=1)
    # #print (out.size())
    # nu = sentence.cpu().numpy()
    # #print (nu)
    
    # for i in nu:
        # pattern.append(int_to_vocab[i])
    
    # new_words = np.expand_dims(nu, axis=0)
    # x_ = Variable(T.from_numpy(new_words).long()).cuda()
    # #print (new_words)
    # #print (new_words)


# print (np.array(pattern))
