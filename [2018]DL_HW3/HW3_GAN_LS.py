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
FILENAME1 = './save/DCGAN_D_ls.pkl'
FILENAME2 = './save/DCGAN_G_ls.pkl'

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
            
            # state size: 1 x 1 x 1
        )


    def forward(self, input):
        output = self.main(input)
        return output
        
        
        
        
        
D = T.nn.DataParallel(netD(), device_ids=[0,1]).cuda()
G = T.nn.DataParallel(netG(), device_ids=[0,1]).cuda()

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
            #y_real_ = T.ones(mini_batch)
            #y_fake_ = T.zeros(mini_batch)
    
            
            x_ = Variable(x_.cuda())
            D_result = D(x_).squeeze()
            #print (D_result.size())
            
            ###LSGAN
            D_real_loss = T.mean(T.pow(D_result-1,2))
    
            z_ = T.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            z_ = Variable(z_.cuda())
            G_result = G(z_)
    
            D_result = D(G_result).squeeze()
            ###LSGAN
            D_fake_loss = T.mean(T.pow(D_result, 2))
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
            
            ###LSGAN
            G_train_loss = T.mean(T.pow(D_result-1, 2))
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
        np.save('./DCGAN_results/D_losses_ls.npy',D_list)
        np.save('./DCGAN_results/G_losses_ls.npy',G_list)
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
            cv2.imwrite('./GAN_LS_'+str(j)+'.bmp', for_ori*255)
            for_ori = np.zeros(64*64).reshape(64,64)
            j += 1
    if j == 5:
        break

###merge five row imgs into single one
for row in range(5):
    if row == 0:
        r1 = cv2.imread('./GAN_LS_'+str(row)+'.bmp')
    else:
        next = cv2.imread('./GAN_LS_'+str(row)+'.bmp')
        r1 = np.vstack((r1,next))
        
RESULT = './GAN_LS_result.bmp'
cv2.imwrite(RESULT, r1)
###del five row imgs
for row in range(5):
    file = './GAN_LS_'+str(row)+'.bmp'
    os.remove(file)
print ('-----Image is saved-----')

###show
img = cv2.imread(RESULT) 
cv2.imshow("RESULT", img)
cv2.waitKey (0)  
cv2.destroyAllWindows() 