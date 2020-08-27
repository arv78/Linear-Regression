import matplotlib.pyplot as plt
import numpy as np
import math
import random


# first load the data base
in_file = open("F:\\AI-1\\projects\\tamrin\\features.txt")
features = []
while True:
    temp = in_file.readline()
    if len(temp) == 0:
        break
    features.append(eval(temp))

in_file_2 = open("F:\\AI-1\\projects\\tamrin\\labels.txt")
labels = []
while True:
    temp = in_file_2.readline()
    if len(temp) == 0:
        break
    labels.append(eval(temp))

in_file.close()
in_file_2.close()

new_features = np.array(features)
new_features = new_features.reshape(-1,1)
new_labels = np.array(labels)
new_labels = new_labels.reshape(-1,1)

# training phase
phi_train=np.concatenate((np.ones((len(new_features),1)),new_features),axis=1)

t = 0.8
y_pred = np.zeros(len(new_features))


for i in range(len(new_features)):
    w = np.mat(np.eye((len(new_features))))
    for j in range(len(new_features)):
        a = -1 * pow((new_features[j][0] - new_features[i][0]),2)
        b = 2 * pow(t,2)
        w[j,j] = np.exp(a/b)
    weight = np.matmul(np.linalg.inv(np.matmul(np.transpose(phi_train), np.matmul(w, phi_train))), np.matmul(np.matmul(np.transpose(phi_train), w), new_labels))
    # model
    y_pred[i] = np.matmul(phi_train[i],weight)

plt.scatter(new_features, new_labels,  color='black')
# sort for visualization
zip_ = zip(new_features,y_pred)
sorted_ = sorted(zip_)

tuple_ = zip(*sorted_)
new_features,y_pred_sort = [list(tuple) for tuple in tuple_]

# MSE
temp = 0
for i in range (len(labels)):
        temp += math.pow((y_pred[i] - labels[i]),2)
temp /= len(labels)
print("MSE:",temp)

plt.plot(new_features, y_pred_sort, color='blue', linewidth=3)
plt.show()