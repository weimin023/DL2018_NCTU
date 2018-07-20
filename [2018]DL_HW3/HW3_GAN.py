import matplotlib.pyplot as plt
import numpy as np
import os, time
import os.path as path

import torch.nn.functional as F
import torch as T
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torch.utils.data as Data
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.optim as optim

from PIL import Image as im
import cv2

if os.path.exists('./save') == 0:
    os.makedirs('./save')
    
TRANSFORM = transforms.Compose([transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])
            
# Hyper Parameters
EPOCH_ = 100               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 80
TEST_BATCH_SIZE = 1
LR = 0.0001              # learning rate
FILENAME1 = './save/DCGAN_D.pkl'
FILENAME2 = './save/45_33/DCGAN_G.pkl'

###DATASET
trainpath = '/home/wmchen/DL2018/HW3/face_training/'
trainset = datasets.ImageFolder(trainpath ,transform = TRANSFORM)

trainloader = Data.DataLoader(trainset, shuffle = True,
                            batch_size = BATCH_SIZE,
                            num_workers = 2)
                              
testpath = '/home/wmchen/DL2018/HW3/face_testing/'
testset = datasets.ImageFolder(testpath ,transform = TRANSFORM)

testloader = Data.DataLoader(testset, shuffle = True,
                            batch_size = TEST_BATCH_SIZE,
                            num_workers = 2)

class netG(nn.Module):
    '''
        Generative Network
    '''
    def __init__(self, z_size=100, out_size=3, ngf=128):
        super(netG, self).__init__()
        self.z_size = z_size
        self.ngf = ngf
        self.out_size = out_size

        self.main = nn.Sequential(
            # input size is z_size
            nn.ConvTranspose2d(self.z_size, self.ngf * 8, kernel_size = 4, stride = 1, padding = 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(inplace=True),
            # state size: (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(inplace=True),
            # state size: (ngf * 4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(inplace=True),
            # state size: (ngf * 2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(inplace=True),
            # state size: ngf x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.out_size, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: out_size x 64 x 64
            )
            
    def forward(self, input):
        output = self.main(input)
        return output

        
class netD(nn.Module):
    '''
        Discriminative Network
    '''
    def __init__(self, in_size=3, ndf=128):
        super(netD, self).__init__()
        self.in_size = in_size
        self.ndf = ndf

        self.main = nn.Sequential(
            # input size is in_size x 64 x 64
            nn.Conv2d(self.in_size, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: ndf x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size: 1 x 1 x 1
        )


    def forward(self, input):
        output = self.main(input)
        return output
        

        
def generate_img(epoch,tensor_in,for_ori,i,j):
    z_ = T.randn((80, 100)).view(-1, 100, 1, 1)
    z_ = Variable(z_.cuda())
    
    test_images = G(z_)
    
    ###img folder
    DIR = './DCGAN_results/'+str(epoch)+'/'
    if not os.path.isdir(DIR):
        os.mkdir(DIR)
    ###generate img
    tensor_in = np.squeeze(tensor_in.detach().cpu().numpy())
    tensor_in = np.rollaxis(tensor_in,0,3)
    
    if for_ori.all() == 1:
        for_ori = tensor_in
    else:
        if i != 5:
            for_ori = np.hstack((for_ori,tensor_in))
            i+=1
        else:
            i=0
            for_ori = cv2.cvtColor(for_ori, cv2.COLOR_BGR2RGB)
            cv2.imwrite(DIR+'gan_'+str(j)+'.bmp', for_ori*255)
            for_ori = np.ones(64*64).reshape(64,64)
            j += 1
    
    G.train()
    return i, j, for_ori
        
        
        
D = T.nn.DataParallel(netD(), device_ids=[0]).cuda()
G = T.nn.DataParallel(netG(), device_ids=[0]).cuda()

#OPTIMIZER
D_optim = optim.Adam(D.parameters(), lr = LR)
G_optim = optim.Adam(G.parameters(), lr = LR)

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

if not os.path.isdir('DCGAN_results'):
    os.mkdir('DCGAN_results')
    

    
if path.exists(FILENAME2) == False:
    print('-----Training start-----')
    D.train()
    G.train()
    
    start_time = time.time()
    D_list = []
    G_list = []
    for epoch in range(EPOCH_):
        D_losses = []
        G_losses = []
    
        
        # i = 0 #five imgs a row 
        # j = 0 #five imgs a column
        # for_ori = np.ones(64*64).reshape(64,64)
        epoch_start_time = time.time()
        for x_, _ in trainloader:
            # train discriminator D
            D.zero_grad()
            
            mini_batch = x_.size()[0]
            #print (x_.size()) 80,3,64,64
            y_real_ = T.ones(mini_batch)
            y_fake_ = T.zeros(mini_batch)
    
            
            x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
            D_result = D(x_).squeeze()
            #print (D_result.size())
            
            D_real_loss = BCE_loss(D_result, y_real_)
    
            z_ = T.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            z_ = Variable(z_.cuda())
            G_result = G(z_)
    
            D_result = D(G_result).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake_)
            #D_fake_score = D_result.data.mean()
    
            D_train_loss = D_real_loss + D_fake_loss
    
            D_train_loss.backward()
            D_optim.step()
    
            D_losses.append(D_train_loss.item())
            
            # train generator G
            G.zero_grad()
    
            z_ = T.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            z_ = Variable(z_.cuda())
    
            G_result = G(z_)
            D_result = D(G_result).squeeze()
            G_train_loss = BCE_loss(D_result, y_real_)
            G_train_loss.backward()
            G_optim.step()
    
            G_losses.append(G_train_loss.item())
            ###generate imgs
            # if epoch%10 == 0 and j < 5:
                # i, j, for_ori = generate_img(epoch,G_result,for_ori,i,j)

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
    
    
        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), EPOCH_, per_epoch_ptime, T.mean(T.FloatTensor(D_losses)),T.mean(T.FloatTensor(G_losses))))
        D_list.append(T.mean(T.FloatTensor(D_losses)))
        G_list.append(T.mean(T.FloatTensor(G_losses)))
        np.save('./DCGAN_results/D_losses.npy',D_list)
        np.save('./DCGAN_results/G_losses.npy',G_list)
        T.save(D,FILENAME1)
        T.save(G,FILENAME2)
        
    end_time = time.time()
    total_ptime = end_time - start_time
    
    
    
    #print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (T.mean(T.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
    print("Training finish!... save training results")


print ('-----Start Generating-----')
Dmodel = T.load(FILENAME1)
Gmodel = T.load(FILENAME2)
print ('--------------------------------------------')
print ('|               Model loaded               |')
print ('|        D:',FILENAME1,'         |')
print ('|        G:',FILENAME2,'         |')
print ('|------------------------------------------|')
#Dmodel.eval()
Gmodel.eval()
for_ori = np.zeros(64*64).reshape(64,64)
i = 0
j = 0
for epoch in range(EPOCH_):
    z_ = T.randn((1, 100)).view(-1, 100, 1, 1)
    z_ = Variable(z_.cuda())
    
    test_images = Gmodel(z_)
    test_images = np.squeeze(test_images.detach().cpu().numpy())
    test_images = np.rollaxis(test_images,0,3)
    
    if for_ori.all() == 0:
        for_ori = test_images
    else:
        if i != 5:
            for_ori = np.hstack((for_ori,test_images))
            i+=1
        else:
            i=0
            for_ori = cv2.cvtColor(for_ori, cv2.COLOR_BGR2RGB)
            cv2.imwrite('./GAN_'+str(j)+'.bmp', for_ori*255)
            for_ori = np.zeros(64*64).reshape(64,64)
            j += 1
    if j == 5:
        break

###merge five row imgs into single one
for row in range(5):
    if row == 0:
        r1 = cv2.imread('./GAN_'+str(row)+'.bmp')
    else:
        next = cv2.imread('./GAN_'+str(row)+'.bmp')
        r1 = np.vstack((r1,next))
        
RESULT = './GAN_result.bmp'
cv2.imwrite(RESULT, r1)
###del five row imgs
for row in range(5):
    file = './GAN_'+str(row)+'.bmp'
    os.remove(file)
print ('-----Image is saved-----')

###show
img = cv2.imread(RESULT) 
cv2.imshow("RESULT", img)
cv2.waitKey (0)  
cv2.destroyAllWindows() 