import csv, os 
import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

###PARAMETERS
LR = 0.1

reTRAIN = False
NEURAL = 3
#BATCHES = 2000
EPOCHES = 500
TEST_EPOCH = 500

###DATASET PREPROCESS 
file = sio.loadmat('./Dataset/spam_data.mat')

train_x = file['train_x']
train_y = file['train_y']
test_x = file['test_x']
test_y = file['test_y']

print ('-----Training set dimension-----')
print (train_x.shape)
print (train_y.shape)
print ('-----Testing set dimension-----')
print (test_x.shape)
print (test_y.shape)

# (2000, 40)
# (2000, 2)
# (500, 40)
# (500, 2)

W_0 = np.ones((40,NEURAL)) * 0.01
B_0 = np.ones((1,3))* 0.01
W_1 = np.ones((NEURAL,2)) * 0.01
B_1 = np.ones((1,2))* 0.01
#random.random
###ACTIVATION
def SIGMOID_(x,der = False):
    if (der == True):
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))
        
def SOFTMAX(x):
    #print(x)
    return np.exp(x) / np.sum(np.exp(x))
        
###START TRAINING
train_xy = []
num = 0
for row in train_x:
    train_xy.append(np.append(row,train_y[num]))
    num+=1

train_xy = np.array(train_xy)

if reTRAIN == True:
    print ('-----Start Linear Regression!-----')
    ERROR_RATE = []
    ce_list = []
    epoch_list = []
    for epoch in range(EPOCHES):
        #shuffle
        training_set = np.random.permutation(train_xy)
        #batch
        #batch = training_set.shape[0]//BATCHES
        #train_batches = np.array_split(training_set,batch)
        CE = 0
    
        for i in training_set:
            feature = i[0:40]
            label = i[40:42]
            
            ###reshape
            feature = np.array(feature).reshape(1,40)
            label = np.array(label).reshape(1,2)
            ###fORWARD PROP
            Z = np.matmul(feature,W_0)+B_0 #1,3
            Layer1 = SIGMOID_(Z)
            output = np.matmul(Layer1,W_1)+B_1 #1,2
            predict = SIGMOID_(output)
    
            ###COST
            CE += -np.sum(label*np.log(predict))
            
            ###BACKWARD PROP
            output_delta = (label - predict)*SIGMOID_(predict,der=True) #g
            L1_error = np.dot(output_delta,W_1.T) #dE/ds
            L1_delta = L1_error*SIGMOID_(Layer1,der=True)  #(1,3)
            
            W_1 += np.matmul(np.transpose(Layer1),output_delta)*LR #(3,2)
            B_1 += output_delta*LR
            W_0 += np.matmul(np.transpose(feature),L1_delta)*LR
            B_0 += L1_delta*LR
        averce = CE/len(training_set)
        
        ###CALCULATE ERROR RATE
        cal_error_training = np.random.permutation(training_set)
        error_ = 0
        
        for j in cal_error_training:
            f_ = j[0:40]
            l_ = j[40:42]
            L1_ = SIGMOID_(np.matmul(f_,W_0)+B_0)
            OUT_ = SIGMOID_(np.matmul(L1_,W_1)+B_1)
            
            ###CLASSIFICATION
            set1_ = np.where(OUT_[0]>=0.5)[0]
            set0_ = np.where(OUT_[0]<0.5)[0]
            OUT_[0][set1_] = 1
            OUT_[0][set0_] = 0
            
            if np.array_equal(OUT_[0] ,l_) == False:
                error_ += 1
        er = error_/len(cal_error_training)
        #print ('er:',er)
        ERROR_RATE.append(er)

        
        ###PLOT
        ce_list.append(averce)
        epoch_list.append(epoch)
        #print ('Average Loss:',averce)
        
    np.save('./Dataset/npy/HW1_2/W1.npy',W_1)
    np.save('./Dataset/npy/HW1_2/W0.npy',W_0)
    np.save('./Dataset/npy/HW1_2/B1.npy',B_1)
    np.save('./Dataset/npy/HW1_2/B0.npy',B_0)
    
    np.save('./Dataset/npy/HW1_2/CE.npy',ce_list)
    np.save('./Dataset/npy/HW1_2/EPOCH.npy',epoch_list)
    np.save('./Dataset/npy/HW1_2/ERROR_RATE.npy',ERROR_RATE)
    #print (ERROR_RATE)

        
###START TESTING
print ('-----Start testing-----')
test_xy = []
num_ = 0
for row in test_x:
    test_xy.append(np.append(row,test_y[num_]))
    num_+=1

test_xy = np.array(test_xy)

W_0 = np.load('./Dataset/npy/HW1_2/W0.npy')
W_1 = np.load('./Dataset/npy/HW1_2/W1.npy')
B_0 = np.load('./Dataset/npy/HW1_2/B0.npy')
B_1 = np.load('./Dataset/npy/HW1_2/B1.npy')

test_error = []
for test_epoch in range(TEST_EPOCH):
    #shuffle
    test_set = np.random.permutation(test_xy)
    predict_er = 0
    for X in test_set:
        feature = X[0:40]
        label = X[40:42]
        
        ###RESHAPE
        feature = np.array(feature).reshape(1,40)
        label = np.array(label).reshape(1,2)
        #print (feature)
        ###fORWARD PROP
        Layer1 = SIGMOID_(np.matmul(feature, W_0)+B_0) #(1,3)
        output = SIGMOID_(np.matmul(Layer1, W_1)+B_1)#(1,2)
        
        ###CLASSIFICATION
        set1 = np.where(output[0]>=0.5)[0]
        set0 = np.where(output[0]<0.5)[0]
        output[0][set1] = 1
        output[0][set0] = 0
        
        if np.array_equal(output[0] ,label[0]) == True:
            predict_er += 1
        
    tester = predict_er/len(test_xy)
    test_error.append(1-tester)
    print ('Epoch [%d/%d], Testing Accuracy:%f %%'%(test_epoch+1,TEST_EPOCH,tester*100))
    
x_axis = np.load('./Dataset/npy/HW1_2/EPOCH.npy')
y_axis = np.load('./Dataset/npy/HW1_2/CE.npy')

plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Cross Entropy')
plt.plot(x_axis, y_axis)
#plt.legend(['Label', 'Predict'], loc='upper right')
plt.show()
plt.grid()

###plot training error rate
training_ER = np.load('./Dataset/npy/HW1_2/ERROR_RATE.npy')
plt.title('Training Error Rate')
plt.xlabel('Epoch')
plt.ylabel('Error Rate')
plt.plot(x_axis, training_ER)
plt.show()
plt.grid()

###plot testing error rate
plt.title('Testing Error Rate')
plt.xlabel('Epoch')
plt.ylabel('Error Rate')
plt.plot(x_axis, test_error)
plt.show()
plt.grid()