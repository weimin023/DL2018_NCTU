import csv, os 
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

###PARAMETERS
LR = 0.00001
TRAINING_EPOCH = 2000
reTRAIN = False

###DATASET PREPROCESS 
tmpDATASET = []
with open('./Dataset/energy_efficiency_data.csv','r') as file:
    for row in csv.reader(file):
        tmpDATASET.append(row)

print ('There are %d samples'%(len(tmpDATASET[1:])))

DATASET = tmpDATASET[1:]
TRAINING = []
TESTING = []
for ele in range(len(DATASET)):
    if ele%4 == 0:
        TESTING.append(DATASET[ele])
    else:
        TRAINING.append(DATASET[ele])
print ('Training Data:',len(TRAINING))
print ('Testing Data:',len(TESTING))


if reTRAIN == True:
    print ('-----Start Linear Regression!-----')
    #y = Wx+b
    Bias = 0.001
    Weight = np.ones(8).reshape(1,8)
    
    #error
    
    plt_x = []
    plt_y = []
    for epoch in range(TRAINING_EPOCH):
        training_set = np.random.permutation(TRAINING)
        
        
        N = len(TRAINING)
        W_grad = 0
        B_grad = 0
        
        feature = []
        label = []
        sumSE = 0
        
        ###FOR PLOT
        predict = []
        true = []
        y_axis = []
        num = 0
        for X in training_set:
            ###SUM OF ERROR
            X = np.float32(X)
            feature = np.array(X[:8]).reshape(1,8)
            label = float(X[8])
            
            out_label = np.matmul(feature, Weight.T)+Bias
            heat_label = label
            
            SE = (heat_label-out_label)**2
            sumSE += SE
            
            
            ###UPGRADE GRADIENT
            W_grad = -(2/N)*feature*(heat_label-(np.matmul(feature, Weight.T)+Bias))
            B_grad = -(2/N)*(heat_label-(np.matmul(feature, Weight.T)+Bias))
            
            Weight = Weight - (LR*W_grad)
            Bias = Bias - (LR*B_grad)

            ###PLOT
            num += 1
            predict.append(out_label[0][0])
            true.append(heat_label)
            y_axis.append(num)
            
        RMS = np.power(sumSE/len(TRAINING),0.5)
        plt_x.append(epoch)
        plt_y.append(float(sumSE))
    
        if epoch%50 == 0:
            print ('After %d epochs, RMS: %f, Bias: %f' %(epoch, RMS, Bias))
            print ('Weight:', Weight)
            
    np.save('./Dataset/npy/RMS.npy', RMS)
    np.save('./Dataset/npy/Weight.npy', Weight)
    np.save('./Dataset/npy/Bias.npy', Bias)
    np.save('./Dataset/npy/plt_x.npy', plt_x)
    np.save('./Dataset/npy/plt_y.npy', plt_y)
    ###FOR PIC 2nd
    np.save('./Dataset/npy/plt_x2.npy', true)
    np.save('./Dataset/npy/plt_y2.npy', predict)
    np.save('./Dataset/npy/train_axis.npy', y_axis)
    
print ('------------------------------')
###SHOW TRAINING RESULTS
train_result = np.load('./Dataset/npy/RMS.npy')
print ('Training RMS:',train_result[0][0])
print ('-----Start Model Testing.-----')
W = np.load('./Dataset/npy/Weight.npy')
B = np.load('./Dataset/npy/Bias.npy')
plt_x = np.load('./Dataset/npy/plt_x.npy')
plt_y = np.load('./Dataset/npy/plt_y.npy')

plt_x2 = np.load('./Dataset/npy/plt_x2.npy')
plt_y2 = np.load('./Dataset/npy/plt_y2.npy')
train_axis = np.load('./Dataset/npy/train_axis.npy')

###FOR PLOT
test_predict = []
test_true = []
test_axis = []
num = 0

sumSE = 0
for X in TESTING:
    X = np.float32(X)
    feature = np.array(X[:8]).reshape(1,8)
    
    heat_label = float(X[8])
    out_label = np.matmul(feature, W.T) + B
    
    SE = (heat_label-out_label)**2
    sumSE += SE
    
    ###FOR PLOT
    test_predict.append(out_label[0][0])
    test_true.append(heat_label)
    num += 1 
    test_axis.append(num)
    
    
RMS = np.power(sumSE/len(TESTING),0.5)
print ('Testing RMS:',RMS[0][0])

#---------------------------------------
#plt.subplot(311)
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.plot(plt_x, plt_y)
plt.show()
plt.grid()


    
#plt.subplot(312)
plt.title('Heat load for training dataset')
plt.xlabel('num of the case')
plt.ylabel('Heat Load')
plt.plot(train_axis, plt_x2)
plt.plot(train_axis, plt_y2)
plt.legend(['Label', 'Predict'], loc='upper right')
plt.show()
plt.grid()


    
#plt.subplot(313)
plt.title('Heat load for testing dataset')
plt.xlabel('num of the case')
plt.ylabel('Heat Load')
plt.plot(test_axis, test_predict)
plt.plot(test_axis, test_true)
plt.legend(['Label', 'Predict'], loc='upper right')
plt.show()
plt.grid()


