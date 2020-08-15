"""
=====================================
Visualization of MLP weights on MNIST
=====================================

Sometimes looking at the learned coefficients of a neural network can provide
insight into the learning behavior. For example if weights look unstructured,
maybe some were not used at all, or if very large coefficients exist, maybe
regularization was too low or the learning rate too high.

This example shows how to plot some of the first layer weights in a
MLPClassifier trained on the MNIST dataset.

The input data consists of 28x28 pixel handwritten digits, leading to 784
features in the dataset. Therefore the first layer weight matrix have the shape
(784, hidden_layer_sizes[0]).  We can therefore visualize a single column of
the weight matrix as a 28x28 pixel image.

To make the example run faster, we use very few hidden units, and train only
for a very short time. Training longer would result in weights with a much
smoother spatial appearance.
"""
print(__doc__)

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA
from sklearn.decomposition import FactorAnalysis
from sklearn.manifold import Isomap

import umap

import numpy as np
import cv2
import glob
import time
import math
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split

import pandas as pd


plt.close("all")

#mnist = fetch_mldata("MNIST original")

#img = cv2.imread('honolulu-airport.jpg',0)




#images = [  np.reshape( np.reshape (cv2.imread(file, 0)/255 , 394*1200) ,(1200,394))  for file in glob.glob("*.jpg")]
#plt.figure(1)
#plt.imshow(images[0], cmap = 'gray', interpolation = 'bicubic')

#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()


#wx , wy =25, 25
Nhid1, Nhid2=50,50

#I = plt.imread("./Input/OPCarray_InputMask/1.png")
#plt.imshow(I)

# read Input mask images
images = [   plt.imread(file)  for file in sorted(glob.glob("C:/Users/HP/Downloads/To_Chandni/To_Chandni/Input_NoAugmented/*.png") )]


#lll=[str(kk) for kk in range(1,51)]
#lll=sorted(lll)

#for hh in range(0,50):
 #   plt.imsave('./Input/OPCarray_InputMask/d/'+ lll[hh]+'.png', images[hh],cmap=cm.gray)
  

#plt.imshow(images[1])
#plt.pause(1) 

#images = [   cv2.resize(cv2.imread(file, 0) , ( 
     #     int(wx),int(wy)  )  )  for file in glob.glob("./Input/OPCarray_InputMask/*.tif")]

EntireImageSizexi=600
EntireImageSizeyi=400
EntireImageSizexo=65
EntireImageSizeyo=38

for idx in enumerate(images):
    #resizing images, mainly downsampling, CV2 order reversed
    images[idx[0]]= np.round (cv2.resize(images[idx[0]],(EntireImageSizexi,EntireImageSizeyi)))
    #Padding input images to larger size
    #images[idx[0]]=np.pad(images[idx[0]], ((50,50),(50,50)),
    #      'constant', constant_values=((0,0), (0,0)) ) 

#images=cv2.resize(images,(50,50))
# read binarized SEM images     
images2 = [   plt.imread(file)  for file in sorted( glob.glob("C:/Users/HP/Downloads/To_Chandni/To_Chandni/Output/*.png") )]

for idx in enumerate(images2):
    images2[idx[0]]=np.round ( cv2.resize(images2[idx[0]],(EntireImageSizexo,EntireImageSizeyo)) )

#plt.imshow(images2[1])
#plt.pause(1) 


for ii in range(len(images)):
    images[ii]= np.reshape (images[ii] , EntireImageSizexi*EntireImageSizeyi ) 

    
for ii in range(len(images2)):
    images2[ii]= np.reshape (images2[ii] , EntireImageSizexo*EntireImageSizeyo) 



subimages=np.asarray(images)

subimages2=np.asarray(images2)


split=45
X, y = np.float64(subimages) , np.float64(subimages2)

del subimages, subimages2

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

start1 = time.time()
##################################################################################
pca1=PCA(n_components=1,svd_solver = 'auto',  random_state=1)
#ica1=FastICA(n_components=40, random_state=1)
#tsne1 = TSNE(n_components=2,random_state=1,method='exact' )
#fa1=FactorAnalysis(n_components=40, random_state=1)
#embedding = Isomap(n_components=20)
#reducer = umap.UMAP(n_components=20, random_state=1)
##############################################################
X_train_t =pca1.fit_transform(X_train)
#X_train_t =ica1.fit_transform(X_train)
#X_train_t = fa1.fit_transform(X_train)
#X_train_t=tsne1.fit_transform(X_train)
#X_train_t=embedding.fit_transform(X_train)
#X_train_t =reducer.fit_transform(X_train)


sc1=MinMaxScaler()
X_train_t =sc1.fit_transform(X_train_t)
#X_train_t=X_train



## convert your array into a dataframe
#df = pd.DataFrame (X_train)
#
### save to xlsx file
#
#filepath = 'my_excel_file.xlsx'
#
#df.to_excel(filepath, index=False)

###########################################
#X_test_t= pca1.transform(X_test)
#X_test_t=ica1.transform(X_test)
#X_test_t= fa1.transform(X_test)
#X_test_t=embedding.transform(X_test)
X_test_t=reducer.transform(X_test)




X_test_t=sc1.transform(X_test_t)
#X_test_t=X_test
end1 = time. time()

print('execuation time of DR is %f'  % (end1 - start1))
#
#plt.figure(1)
#plt.scatter(X_train_t[:,0], X_train_t[:,1] )
#plt.scatter(X_test_t[:,0], X_test_t[:,1],marker='^', c='g' )
#############################@@START
start2=time.time()
# mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1)
mlp = MLPClassifier(hidden_layer_sizes=(Nhid1,Nhid2 ), activation='relu', solver='adam', alpha=0.0001,batch_size='auto', learning_rate='constant', learning_rate_init=1e-3, power_t=0.5, max_iter=20000, shuffle=True, random_state=1, tol=1e-4,verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,epsilon=1e-08)
#mlp= MLPRegressor(hidden_layer_sizes=(Nhid1,Nhid2),  activation='relu', solver='adam',    alpha=0.001,batch_size='auto',
#               learning_rate='constant', learning_rate_init=1e-4, power_t=0.5, max_iter=100000, shuffle=True,
#               random_state=None, tol=1e-6, verbose=True, warm_start=False, momentum=0.9,
#               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
#               epsilon=1e-08)

mlp.fit(X_train_t, y_train)

ytrain_pred=mlp.predict(X_train_t)
ytest_pred=mlp.predict(X_test_t)
end2 = time. time()



print('execuation time is of NN %f'  % (end2 - start2))


#
#print("Training set score: %f" % mlp.score(X_train_t, y_train))
#
#print("Test set score: %f" % mlp.score(X_test_t, y_test))

print('log loss')
#print (log_loss(y_test, mlp.predict_proba(X_test)  )  )

print('accuracy score')
#print (accuracy_score(y_test, ytest_pred )  )

print('mean squared error in training set')
print (mean_squared_error(y_train, ytrain_pred )  )
print('mean squared error in test set')
print (mean_squared_error(y_test, ytest_pred )  )


#X_train =sc1.inverse_transform(X_train_t)
#X_train=tsne1.inverse_transform(X_train_t)
#X_test =sc1.inverse_transform(X_test_t)
#X_test=tsne1.inverse_transform(X_test_t)




y_train=np.reshape(y_train , (split, EntireImageSizeyo,EntireImageSizexo))
ytrain_pred=np.reshape(ytrain_pred , (split,EntireImageSizeyo,EntireImageSizexo))


y_test=np.reshape(y_test , (len(images)-split,EntireImageSizeyo,EntireImageSizexo))
ytest_pred=np.reshape(ytest_pred , (len(images)-split,EntireImageSizeyo,EntireImageSizexo))

#save real images---in case of random train test split, no ordering is required 
#since order already lost
lll=[str(kk) for kk in range(1,51)]
lll=sorted(lll)


for hh in range(0,45):
    plt.imsave(' train/'+ lll[hh]+'.png', y_train[hh],cmap=cm.gray)
    
    

for hh in range(0,5):
    plt.imsave('C:/Users/HP/Downloads/To_Chandni/To_Chandni/new data plots/PCA/test/'+ lll[hh+45]+'.png', y_test[hh],cmap=cm.gray)
# save preidcted images
lll=[str(kk) for kk in range(1,51)]
lll=sorted(lll)


for hh in range(0,45):
    plt.imsave('C:/Users/HP/Downloads/To_Chandni/To_Chandni/new data plots/PCA/pred_train/'+ lll[hh]+'.png', ytrain_pred[hh],cmap=cm.gray)
    



for hh in range(0,5):
    plt.imsave('C:/Users/HP/Downloads/To_Chandni/To_Chandni/new data plots/PCA/pred_test/'+ lll[hh+45]+'.png', ytest_pred[hh],cmap=cm.gray)

#plt.figure(2)
#plt.imshow(np.uint8(ytrain_pred[2]), cmap = 'gray', interpolation = 'bicubic')
#
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()
#
#plt.figure(3)
#plt.imshow(np.uint8(y_train[2]), cmap = 'gray', interpolation = 'bicubic')
#
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()
#
#
##import sys
##sys.exit("Error message")
#
#plt.figure(4)
#plt.imshow(np.uint8(ytest_pred[3]), cmap = 'gray', interpolation = 'bicubic')
#
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()
#
#plt.figure(5)
#plt.imshow(np.uint8(y_test[3]), cmap = 'gray', interpolation = 'bicubic')
#
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()

