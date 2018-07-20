import matplotlib.pyplot as plt
import numpy as np
import os 
import os.path as path

import cv2

import torch as T
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torch.utils.data as Data
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import torch.optim as optim
from PIL import Image as im

# Hyper Parameters
GPU_MODE = True
EPOCH_ = 20               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 20
TEST_BATCH_SIZE = 1
LR = 0.001              # learning rate
MOMENTUM = 0.9
L2_PENALTY = 0 #1e-5
FILENAME = './save/HW2_Food11.pkl'

if os.path.exists('./save') == 0:
    os.makedirs('./save')


normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

TRANSFORM = transforms.Compose([transforms.CenterCrop(512),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
     

###DATASET
trainpath = './HW2/Dataset/Food_11/training/'
trainset = datasets.ImageFolder(trainpath ,transform = TRANSFORM)

trainloader = Data.DataLoader(trainset, shuffle = True,
                            batch_size = BATCH_SIZE,
                            num_workers = 2)
                              
                              
###CLASSES
classes = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit']


###NET  
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size = 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(4))
        self.fc = nn.Linear(32*32*32,11)
    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = out3.view(out3.size(0), -1)
        outfc = self.fc(out4)

        #for cnn visual
        # np.save('./HW2/ori.npy',x.detach().cpu().numpy())
        # np.save('./HW2/out1.npy',out1.detach().cpu().numpy())
        # np.save('./HW2/out2.npy',out2.detach().cpu().numpy())
        # np.save('./HW2/out3.npy',out3.detach().cpu().numpy())
        # exit()
        return outfc

net = CNN()
net = T.nn.DataParallel(net, device_ids=[0,1,2,3]).cuda()
net.cuda()
#print (net)

if GPU_MODE:net.cuda()

###OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = LR, momentum = MOMENTUM, weight_decay = L2_PENALTY)

###for plot
lo_list = []

if path.exists(FILENAME) == False:
    print ('-----Start Training-----')
###START TRAINING
    for epoch in range(EPOCH_):
        for _, (images, labels) in enumerate(trainloader):
        
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs,labels)
            
            loss.backward()
            optimizer.step()
            
            #for plot
            lo_list.append(loss.data[0])
            step_list.append(_)
            
            printflag = (_+1)
            if printflag%100 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss:%.4f'
                    %(epoch+1, EPOCH_, printflag, len(trainset)//BATCH_SIZE, loss.data[0]))
            
        T.save(net,FILENAME)
        print ('Checkpoint file is saved.')
    print('Finished Training')

    np.save('./HW2/Dataset/npy/loss_list.npy',lo_list)
    
    
trainloader = Data.DataLoader(trainset, shuffle = True,
                            batch_size = TEST_BATCH_SIZE,
                            num_workers = 2)
###show training acc
model = T.load(FILENAME)

total = 0
correct = 0
for data in trainloader:
    images, labels = data
    
    outputs = model(Variable(images).cuda())
    predicted = T.max(outputs.data, 1)[1]
    
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()
    # if (predicted.cpu().numpy()!=labels.cpu().numpy()):
        # #print (images.cpu().numpy().shape)
        # piic = np.squeeze(images.cpu().numpy())
        # piic = np.rollaxis(piic,0,3)
        # print (piic.shape)
        # piic = cv2.cvtColor(piic, cv2.COLOR_BGR2RGB)
        
        # p_name_in_classes = classes[predicted.cpu().numpy()[0]-1]
        # l_name_in_classes = classes[labels.cpu().numpy()[0]-1]
        
        # name = 'predicted:'+ p_name_in_classes +',label:'+ l_name_in_classes
        # cv2.imshow(name,piic)
        # cv2.waitKey(0)
        
print('Accuracy on the training set: %d %%' % (100 * correct / total))
    
###TESTING

###Dataset
testpath = './HW2/Dataset/Food_11/evaluation/'
testset = datasets.ImageFolder(testpath ,transform = TRANSFORM)

testloader = Data.DataLoader(testset, shuffle = True,
                            batch_size = TEST_BATCH_SIZE,
                            num_workers = 2)

correct = 0
total = 0
NN = T.load(FILENAME)

print ('-----Start Testing-----')
# for data in testloader:
    # images, labels = data
    # outputs = NN(Variable(images).cuda())
    # predicted = T.max(outputs.data, 1)[1]

    # total += labels.size(0)
    # correct += (predicted == labels.cuda()).sum()


# print('Testing Accuracy: %d %%' % (100 * correct / total))

# print ("-----PLOT/DEBUG MODE-----")
# para = NN.state_dict()

# fc_weight = para['module.fc.weight'].cpu().numpy()
# fc_weight = np.reshape(fc_weight,(-1))
# hist = np.hstack(fc_weight)
# print (hist)
# plt.hist(hist, bins=250)
# plt.title("Histogram of FC Layer")
# new_ticks = np.linspace(-0.1, 0.1, 5)
# plt.xticks(new_ticks)
# plt.xlabel('Value')
# plt.ylabel('Number')
# plt.show()
# lo_list = np.load('./HW2/Dataset/npy/loss_list.npy')
# ste = []
# for i in range(988):
    # ste.append(i)
# print (len(ste))

# loo = []
# for j in range(9880):
    # if j%10 ==0:
        # loo.append(lo_list[j])
        
# print (len(loo))

# plt.title('Learning Curve')
# plt.xlabel('Iteration')
# plt.ylabel('Cross Entropy')
# plt.plot(ste,loo)
# plt.show()
# plt.grid()
