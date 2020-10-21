#!/usr/bin/env python
# coding: utf-8

# # Your first autoencoder
# 
# (C) 2020 - Umberto Michelucci, Michela Sperti
# 
# This notebook is part of the book _Applied Deep Learning: a case based approach, **2nd edition**_ from APRESS by [U. Michelucci](mailto:umberto.michelucci@toelt.ai) and [M. Sperti](mailto:michela.sperti@toelt.ai).
# 
# This notebook is referenced in Chapter 25 and 26 in the book.

# ## Notebook learning goals
# 
# At the end of this notebook you will be able to build a simple autoencoder with Keras, using `Dense` layers in Keras and apply to images, in particular to the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset and the [fashion MNIST](https://keras.io/api/datasets/fashion_mnist/) dataset as examples.

# In[1]:


import numpy as np
import tensorflow.keras as keras
import pandas as pd
import time

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


# ## MNIST and FASHION MNIST dataset 

# In[2]:


from keras.datasets import mnist
import numpy as np
(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()


# In[3]:


mnist_x_train = mnist_x_train.astype('float32') / 255.
mnist_x_test = mnist_x_test.astype('float32') / 255.
mnist_x_train = mnist_x_train.reshape((len(mnist_x_train), np.prod(mnist_x_train.shape[1:])))
mnist_x_test = mnist_x_test.reshape((len(mnist_x_test), np.prod(mnist_x_test.shape[1:])))


# In[4]:


from keras.datasets import fashion_mnist
import numpy as np
(fashion_x_train, fashion_y_train), (fashion_x_test, fashion_y_test) = fashion_mnist.load_data()


# In[5]:


fashion_x_train = fashion_x_train.astype('float32') / 255.
fashion_x_test = fashion_x_test.astype('float32') / 255.
fashion_x_train = fashion_x_train.reshape((len(fashion_x_train), np.prod(fashion_x_train.shape[1:])))
fashion_x_test = fashion_x_test.reshape((len(fashion_x_test), np.prod(fashion_x_test.shape[1:])))


# ## Function to create the autoencoders

# In[6]:


def create_autoencoders (feature_layer_dim = 16):
  input_img = Input(shape = (784,), name = 'Input_Layer')
  encoded = Dense(feature_layer_dim, activation = 'relu', name = 'Encoded_Features')(input_img)
  decoded = Dense(784, activation = 'sigmoid', name = 'Decoded_Input')(encoded)

  autoencoder = Model(input_img, decoded)
  encoder = Model(input_img, encoded)

  encoded_input = Input(shape = (feature_layer_dim,))
  decoder = autoencoder.layers[-1]
  decoder = Model(encoded_input, decoder(encoded_input))

  return autoencoder, encoder, decoder


# ## (784,16,784)
# 
# 
# 

# In[7]:


autoencoder, encoder, decoder = create_autoencoders (16)


# In[8]:


keras.utils.plot_model(autoencoder, show_shapes=True)


# ### Model compilation

# In[9]:


autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


# In[10]:


history = autoencoder.fit(mnist_x_train, mnist_x_train,
                epochs=30,
                batch_size=256,
                shuffle=True,
                validation_data=(mnist_x_test, mnist_x_test))


# In[11]:


encoded_imgs = encoder.predict(mnist_x_test)
decoded_imgs = decoder.predict(encoded_imgs)


# In[12]:


import matplotlib.pyplot as plt

n = 10  # how many digits we will display
fig = plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(mnist_x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.tight_layout()
plt.show()

fig.savefig('comparison_16.png')


# # Autoencoders (784,64,784)
# 
# 
# 
# 
# 
# 

# In[13]:


autoencoder, encoder, decoder = create_autoencoders (64)


# In[14]:


autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


# In[15]:


autoencoder.fit(mnist_x_train, mnist_x_train,
                epochs=30,
                batch_size=256,
                shuffle=True,
                validation_data=(mnist_x_test, mnist_x_test))


# In[16]:


encoded_imgs = encoder.predict(mnist_x_test)
decoded_imgs = decoder.predict(encoded_imgs)


# In[17]:


import matplotlib.pyplot as plt

n = 10  # how many digits we will display
fig = plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(mnist_x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.tight_layout()
plt.show()

fig.savefig('comparison_32.png')


# #(784,8,784)

# In[18]:


autoencoder, encoder, decoder = create_autoencoders (8)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(mnist_x_train, mnist_x_train,
                epochs=30,
                batch_size=256,
                shuffle=True,
                validation_data=(mnist_x_test, mnist_x_test))


# In[19]:


encoded_imgs = encoder.predict(mnist_x_test)
decoded_imgs = decoder.predict(encoded_imgs)


# In[20]:


encoded_imgs.shape


# In[21]:


import matplotlib.pyplot as plt

n = 10  # how many digits we will display
fig = plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(mnist_x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.tight_layout()
plt.show()

fig.savefig('comparison_8.png')


# ## kNN Study

# In[22]:


encoded_train_imgs = encoder.predict(mnist_x_train)
#decoded_imgs = decoder.predict(encoded_imgs)


# In[23]:


def plot_confusion_matrix(data, labels, output_filename):
    """Plot confusion matrix using heatmap.
 
    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.
 
    """
    sns.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))
 
    #plt.title("Confusion Matrix")
 
    sns.set(font_scale=1.3)
    ax = sns.heatmap(data, annot=True, cmap="Blues", cbar_kws={'label': 'Scale'},fmt='d')
 
    ax.set_xticklabels(labels, fontsize = 16)
    ax.set_yticklabels(labels, fontsize = 16)
 
    ax.set_xlabel("Predicted Label", fontsize = 16)
    ax.set_xlabel("True Label", fontsize = 16)
 
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


# In[24]:


# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import seaborn as sns
  


# In[25]:


start = time.time()
 
# training a KNN classifier 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 7).fit(encoded_train_imgs, mnist_y_train) 
  
# accuracy on X_test 
accuracy = knn.score(encoded_imgs, mnist_y_test) 
print (accuracy )

end = time.time()
print(end - start)
  


# In[26]:


# creating a confusion matrix 
knn_predictions = knn.predict(encoded_imgs)  
cm = confusion_matrix(mnist_y_test, knn_predictions)
cm 
plot_confusion_matrix(cm, [0,1,2,3,4,5,6,7,8,9], "confusion_matrix.png")


# ### kNN With all the features

# In[27]:


start = time.time()

from sklearn.neighbors import KNeighborsClassifier 
knn2 = KNeighborsClassifier(n_neighbors = 7).fit(mnist_x_train, mnist_y_train) 
# accuracy on X_test 
accuracy = knn2.score(mnist_x_test, mnist_y_test) 
print (accuracy )

end = time.time()
print(end - start)


# In[ ]:


1000/60 
# 16 minutes


# In[ ]:


# creating a confusion matrix 
knn_predictions = knn2.predict(encoded_imgs)  
cm = confusion_matrix(mnist_y_test, knn_predictions)
cm 
plot_confusion_matrix(cm, [0,1,2,3,4,5,6,7,8,9], "confusion_matrix_total.png")


# # MSE

# In[ ]:


dim = 16

input_img = Input(shape = (784,))
encoded = Dense(dim, activation = 'relu')(input_img)
decoded = Dense(784, activation = 'sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)


# In[ ]:


encoded_input = Input(shape = (dim,))
decoder = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder(encoded_input))


# In[ ]:


autoencoder.compile(optimizer='adam', loss='mse')


# In[ ]:


autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


# In[ ]:


encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)


# In[ ]:


import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# ## Fashion MNIST

# In[ ]:


autoencoder, encoder, decoder = create_autoencoders (8)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(fashion_x_train, fashion_x_train,
                epochs=30,
                batch_size=256,
                shuffle=True,
                validation_data=(fashion_x_test, fashion_x_test))


# In[ ]:


encoded_imgs = encoder.predict(fashion_x_test)
decoded_imgs = decoder.predict(encoded_imgs)


# In[ ]:


import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(fashion_x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[ ]:


start = time.time()

from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 7).fit(fashion_x_train, fashion_y_train) 
# accuracy on X_test 
accuracy = knn.score(fashion_x_test, fashion_y_test) 
print (accuracy )

end = time.time()
print(end - start)


# ## kNN on learned representation

# In[ ]:


encoded_fashion_train_imgs = encoder.predict(fashion_x_train)
encoded_fashion_test_imgs = encoder.predict(fashion_x_test)


# In[ ]:


start = time.time()
 
# training a KNN classifier 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 7).fit(encoded_fashion_train_imgs, fashion_y_train) 
  
# accuracy on X_test 
accuracy = knn.score(encoded_fashion_test_imgs, fashion_y_test) 
print (accuracy )

end = time.time()
print(end - start)


# # (784,16,784)

# In[ ]:


autoencoder, encoder, decoder = create_autoencoders (16)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(fashion_x_train, fashion_x_train,
                epochs=30,
                batch_size=256,
                shuffle=True,
                validation_data=(fashion_x_test, fashion_x_test))


# In[ ]:


encoded_fashion_train_imgs = encoder.predict(fashion_x_train)
encoded_fashion_test_imgs = encoder.predict(fashion_x_test)


# In[ ]:


start = time.time()
 
# training a KNN classifier 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 7).fit(encoded_fashion_train_imgs, fashion_y_train) 
  
# accuracy on X_test 
accuracy = knn.score(encoded_fashion_test_imgs, fashion_y_test) 
print (accuracy )

end = time.time()
print(end - start)


# In[ ]:




