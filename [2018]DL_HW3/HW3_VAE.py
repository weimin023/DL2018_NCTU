import matplotlib.pyplot as plt
import numpy as np
import os 
import os.path as path

import cv2

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

if os.path.exists('./save') == 0:
    os.makedirs('./save')

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

TRANSFORM = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])
            
# Hyper Parameters
EPOCH_ = 150               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 80
TEST_BATCH_SIZE = 1
LR = 0.0001              # learning rate
FILENAME = './save/HW3_VAE.pkl'
LATENT_CODE_NUM = 32 

HEIGHT = 96
WIDTH = 96
CHANNEL = 3


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




class VAE(nn.Module):
    def __init__(self):
          super(VAE, self).__init__()
    
          self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True)
                    
                
                )
          
          self.fc11 = nn.Linear(128 * 24 * 24, LATENT_CODE_NUM)
          self.fc12 = nn.Linear(128 * 24 * 24, LATENT_CODE_NUM)
          self.fc2 = nn.Linear(LATENT_CODE_NUM, 128 * 24 * 24)
          
          self.decoder = nn.Sequential(                
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()
                )
 
    def reparameterize(self, mu, logvar):
          eps = Variable(T.randn(mu.size(0), mu.size(1))).cuda()
          z = mu + eps * T.exp(logvar/2)            
          
          return z
    
    def forward(self, x):
           #print (x.size()) batch,3,h,w
           out1, out2 = self.encoder(x), self.encoder(x)  # batch_s, 8, 7, 7
           #print (out1.size()) #batch, 128, 24, 24
           #print (out1.view(out1.size(0),-1).size())
           #exit()
           mu = self.fc11(out1.view(out1.size(0),-1))     # batch_s, latent
           #print (mu.size()) 64, 32
           
           logvar = self.fc12(out2.view(out2.size(0),-1)) # batch_s, latent
           #print (logvar.size())
           
           z = self.reparameterize(mu, logvar)      # batch_s, latent      
           #print ('z',z.size())
           out3 = self.fc2(z).view(z.size(0), 128, 24, 24)    # batch_s, 8, 7, 7
           #print ('out3',out3.size()) #64, 128, 24, 24
           #print (self.decoder(out3).size()) 64, 3, 96, 96
           #exit()
           return self.decoder(out3), mu, logvar
           
    def evalu_decoder(self, inp):
        out_ = self.fc2(inp).view(1, 128, 24, 24)
        return self.decoder(out_)
    

model = VAE().cuda()
# net = VAE(nc=3, ngf=96, ndf=96, latent_variable_size=100)
# model = T.nn.DataParallel(net, device_ids=[0,1,2,3]).cuda()
        
###LOSS FUNCTION
reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False

def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x.view(-1, 3*96*96), x.view(-1, 3*96*96))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = T.sum(KLD_element).mul_(-0.5)

    return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr=LR)
#optimizer =  optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

def loss_func(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x,  size_average=False)
    KLD = -0.5 * T.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE+KLD

    
    
def torch2cv_(data,title = 'img'):
    ori_image = np.squeeze(data.detach().cpu().numpy())
    ori_image = np.rollaxis(ori_image,0,3)
    #print (ori_image.shape)
    ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    cv2.imshow(title,ori_image)
    cv2.waitKey(0)
    
    
    
    

loss_list = []
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(trainloader):
        data = Variable(data).cuda()

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        
        #print (recon_batch.size())
        #print (data.size())
        #exit()
        #aaa = loss_func(recon_batch, data, mu, logvar)
        loss = loss_function(recon_batch, data, mu, logvar)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 5 == 0:
            loss_list.append(loss.item())
            np.save('./loss_list.npy',loss_list)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader),
                loss.item() / len(data)))
                
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(trainloader)))
    
    
         
def test(epoch):
    model.eval()
    test_loss = 0
    for data, _ in testloader:
        data = Variable(data).cuda()
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(testloader)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    
if path.exists(FILENAME) == False:
    print ('-----Start Training-----')
    for epoch in range(EPOCH_):
        train(epoch)
        test(epoch)
        T.save(model,FILENAME)
        print ('Model saved.')
        
print ('-----Start Evaluating-Reconstruct-----')
net = T.load(FILENAME)
net.eval()
i = 0
j = 0
for_ori = np.ones(96*96).reshape(96,96)
for_dec = np.ones(96*96).reshape(96,96)
for data, _ in testloader:
    ###original img
    ori_image = np.squeeze(data.detach().cpu().numpy())
    ori_image = np.rollaxis(ori_image,0,3)
    ###decoded img
    data = Variable(data).cuda()
    recon_batch, mu, logvar = net(data)
    #torch2cv_(recon_batch,title = 'img')
    dec_image = np.squeeze(recon_batch.detach().cpu().numpy())
    dec_image = np.rollaxis(dec_image,0,3)
    
    if for_ori.all() == 1:
        for_ori = ori_image
        for_dec = dec_image
    else:
        if i != 5:
            for_ori = np.hstack((for_ori,ori_image))
            for_dec = np.hstack((for_dec,dec_image))
            i+=1
        else:
            i=0
            for_ori = cv2.cvtColor(for_ori, cv2.COLOR_BGR2RGB)
            for_dec = cv2.cvtColor(for_dec, cv2.COLOR_BGR2RGB)
            cv2.imwrite('./ori_'+str(j)+'.bmp', for_ori*255)
            cv2.imwrite('./dec_'+str(j)+'.bmp', for_dec*255)
            for_ori = np.ones(96*96).reshape(96,96)
            for_dec = np.ones(96*96).reshape(96,96)
            j += 1

            if j == 5:
                break

print ('-----Start Evaluating-Create-----')
i = 0
j = 0
for_create = np.zeros(96*96).reshape(96,96)
for num in range(40):
    #with T.no_grad():
        sample = T.randn(1, 32).cuda()
        sample = net.evalu_decoder(sample)
        
        ###puzzle
        created_image = np.squeeze(sample.detach().cpu().numpy())
        created_image = np.rollaxis(created_image,0,3)
        
        if for_create.all() == 0:
            for_create = created_image
            
        else:
            if i != 5:
                for_create = np.hstack((for_create,created_image))
                i+=1
            else:
                i=0
                for_create = cv2.cvtColor(for_create, cv2.COLOR_BGR2RGB)
                cv2.imwrite('./created_'+str(j)+'.bmp', for_create*255)
                for_create = np.zeros(96*96).reshape(96,96)
                j += 1
                
                if j == 5:
                    break
#----------------------image files processing-----------------------
p1 = './ori_0.bmp'
p2 = './ori_1.bmp'
p3 = './ori_2.bmp'
p4 = './ori_3.bmp'
p5 = './ori_4.bmp'

o1 = './dec_0.bmp'
o2 = './dec_1.bmp'
o3 = './dec_2.bmp'
o4 = './dec_3.bmp'
o5 = './dec_4.bmp'

i1 = './created_0.bmp'
i2 = './created_1.bmp'
i3 = './created_2.bmp'
i4 = './created_3.bmp'
i5 = './created_4.bmp'

img_1 = cv2.imread(p1)
img_2 = cv2.imread(p2)
img_3 = cv2.imread(p3)
img_4 = cv2.imread(p4)
img_5 = cv2.imread(p5)

img1 = cv2.imread(o1)
img2 = cv2.imread(o2)
img3 = cv2.imread(o3)
img4 = cv2.imread(o4)
img5 = cv2.imread(o5)

created_img1 = cv2.imread(i1)
created_img2 = cv2.imread(i2)
created_img3 = cv2.imread(i3)
created_img4 = cv2.imread(i4)
created_img5 = cv2.imread(i5)

img__ = np.vstack((img_1,img_2))
img__ = np.vstack((img__,img_3))
img__ = np.vstack((img__,img_4))
img__ = np.vstack((img__,img_5))

img_ = np.vstack((img1,img2))
img_ = np.vstack((img_,img3))
img_ = np.vstack((img_,img4))
img_ = np.vstack((img_,img5))

create_ = np.vstack((created_img1,created_img2))
create_ = np.vstack((create_,created_img3))
create_ = np.vstack((create_,created_img4))
create_ = np.vstack((create_,created_img5))

cv2.imwrite('./dec_result.bmp', img_)
cv2.imwrite('./ori_result.bmp', img__)
cv2.imwrite('./created_result.bmp', create_)

os.remove(p1)
os.remove(p2)
os.remove(p3)
os.remove(p4)
os.remove(p5)

os.remove(o1)
os.remove(o2)
os.remove(o3)
os.remove(o4)
os.remove(o5)

os.remove(i1)
os.remove(i2)
os.remove(i3)
os.remove(i4)
os.remove(i5)
    
