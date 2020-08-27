import matplotlib.pyplot as plt
import numpy as np
import math
import random


# first load the data base
in_file = open("features.txt")
features = []
while True:
    temp = in_file.readline()
    if len(temp) == 0:
        break
    features.append(eval(temp))

in_file_2 = open("labels.txt")
labels = []
while True:
    temp = in_file_2.readline()
    if len(temp) == 0:
        break
    labels.append(eval(temp))

in_file.close()
in_file_2.close()

# train and test
x_train = features[:-20]
x_test = features[-20:]
y_train = labels[:-20]
y_test = labels[-20:]

print(f"train data length: {len(x_train)}")
print(f"test data length: {len(x_test)}")

new_xtrain = np.array(x_train)
new_xtrain = new_xtrain.reshape(-1,1)
# print(new_xtrain.shape)

new_xtest = np.array(x_test)
new_xtest = new_xtest.reshape(-1,1)

# training phase
#1
# phi_train=np.concatenate((np.ones((len(new_xtrain),1)),new_xtrain),axis=1)

#2
# phi_train=np.concatenate((np.ones((len(new_xtrain),1)),new_xtrain,np.power(new_xtrain,2),np.power(new_xtrain,3)),axis=1)

#3
# phi_train=np.concatenate((np.ones((len(new_xtrain),1)),new_xtrain,np.power(new_xtrain,2),np.power(new_xtrain,3),np.power(new_xtrain,4),np.power(new_xtrain,5)),axis=1)

#4
phi_train=np.concatenate((np.ones((len(new_xtrain),1)),new_xtrain,np.power(new_xtrain,2),np.power(new_xtrain,3),np.power(new_xtrain,4),np.power(new_xtrain,5),np.power(new_xtrain,6),np.power(new_xtrain,7),np.power(new_xtrain,8),np.power(new_xtrain,9)),axis=1)

weight=np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(phi_train),phi_train)),np.transpose(phi_train)),y_train)

#1
# phi_test=np.concatenate((np.ones((len(new_xtest),1)),new_xtest),axis=1)

#2
# phi_test=np.concatenate((np.ones((len(new_xtest),1)),new_xtest,np.power(new_xtest,2),np.power(new_xtest,3)),axis=1)

#3
# phi_test=np.concatenate((np.ones((len(new_xtest),1)),new_xtest,np.power(new_xtest,2),np.power(new_xtest,3),np.power(new_xtest,4),np.power(new_xtest,5)),axis=1)

#4
phi_test=np.concatenate((np.ones((len(new_xtest),1)),new_xtest,np.power(new_xtest,2),np.power(new_xtest,3),np.power(new_xtest,4),np.power(new_xtest,5),np.power(new_xtest,6),np.power(new_xtest,7),np.power(new_xtest,8),np.power(new_xtest,9)),axis=1)


# model
y_pred = np.matmul(phi_test,weight)

# sort for visualization
zip_ = zip(new_xtest,y_pred)
sorted_ = sorted(zip_)

tuple_ = zip(*sorted_)
new_xtest,y_pred_sort = [list(tuple) for tuple in tuple_]

# MSE
temp = 0
for i in range (len(y_test)):
        temp += math.pow((y_pred[i] - y_test[i]),2)
temp /= len(y_test)
print("MSE:",temp)

plt.scatter(new_xtrain, y_train,  color='black')
plt.plot(new_xtest, y_pred_sort, color='blue', linewidth=3)
plt.scatter(new_xtest, y_test, color='green')

plt.show()
    
    
        