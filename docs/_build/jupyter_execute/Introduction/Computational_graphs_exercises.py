#!/usr/bin/env python
# coding: utf-8

# # Computational Graphs - Exercises
# 
# Version 1.02
# 
# (C) 2020 - Umberto Michelucci, Michela Sperti
# 
# This notebook is part of the book _Applied Deep Learning: a case based approach, **2nd edition**_ from APRESS by [U. Michelucci](mailto:umberto.michelucci@toelt.ai) and [M. Sperti](mailto:michela.sperti@toelt.ai).

# In[ ]:


# tensorflow setup

get_ipython().run_line_magic('tensorflow_version', '1.x')
import tensorflow as tf


# ## Notebook Learning Goals
# 
# At the end of this notebook you will know how to create an easy computational graph with the basic types in TensorFlow.
# 
# To show you how the different datatypes worked in TensorFlow 1.X we will need this version and therefore at the beginning we use the magic command
# 
#     %tensorflow_version 1.x
# 
# Since if we use TensorFlow 2.X we would have Eager execution enabled and the examples would not work.

# ## Computational Graph - Sum of Two Variables

# Let's see some examples that will show you how you can solve the exercises that you will find in this notebook.
# 
# Let's start with something easy.
# 
# $$
# z = x_1+x_2
# $$ 

# This is simply the sum of two variables. Now let's create the computational graph with different datatypes in TensorFlow to see the differences in how they work.

# ## Graph with ```tf.constant```

# In[ ]:


x1 = tf.constant(1)
x2 = tf.constant(2)
z = tf.add(x1, x2)


# In[ ]:


print(z)


# In[ ]:


print(x1)


# Now let's create a session and let's evaluate the graph

# In[ ]:


sess= tf.Session()
print(sess.run(z))


# In[ ]:


print(sess.run(x1))


# In[ ]:


sess.close()


# ## Graph with ```tf.Variable```

# In[ ]:


x1 = tf.Variable(1)
x2 = tf.Variable(2)
z = tf.add(x1,x2)


# In[ ]:


print(z)


# In[ ]:


#
# WARNING: you will get an error from this cell. This is normal and is designed
# to return an error. Continue to read below for the explanation.
#
sess = tf.Session()
print(sess.run(z))


# The error message is normal... First you have to initialize the variables in TensorFlow! TensorFlow ***does not automatically initialize the variables***.

# In[ ]:


sess.run(x1.initializer)
sess.run(x2.initializer)
print(sess.run(z))


# In[ ]:


init = tf.global_variables_initializer()


# In[ ]:


sess.run(init)
print(sess.run(z))


# In[ ]:


sess.close()


# ## Graph with ```tf.placeholder```

# In[ ]:


x1 = tf.placeholder(tf.float32, 1)
x2 = tf.placeholder(tf.float32, 1)


# Note that in this declaration no values were passed. The "1" is the dimension of the placeholder, in this case a scalar!

# In[ ]:


z = tf.add(x1, x2)


# In[ ]:


print(z)


# In[ ]:


sess = tf.Session()


# In[ ]:


feed_dict = {x1: [1], x2: [2]}
print(sess.run(z, feed_dict))


# In[ ]:


sess.close()


# Now we can also use vectors as input, to show that we don't need to deal only with scalars.

# In[ ]:


x1 = tf.placeholder(tf.float32, [2])
x2 = tf.placeholder(tf.float32, [2])

z = tf.add(x1,x2)
feed_dict = {x1: [1,5], x2: [1,1]}

sess = tf.Session()
sess.run(z, feed_dict)


# ## Exercises

# ### Exercise 1 (Difficulty: easy)

# Draw and develop in TensorFlow with ```tf.constant``` the computational graphs for the following operations
# 
# A) ```w1*x1 + w2*x2 + x1*x1```
# 
# B) ```A*x1 + 3 + x2/2```
# 
# Use as input values ```x1 = 5``` and ```x2 = 6```.

# ### Exercise 2 (Difficulty: medium)

# Draw and develop in TensorFlow with ```tf.Variable``` the computational graph for the following operation ```A*(w1*x1 + w2*x2)```.
# 
# Build the computational graph and then evaluate it two times (without re-building it) with the initial values in the same session
# 
# A) ```x1 = 3, x2 = 4```
# 
# B) ```x1 = 5, x2 = 7```

# ### Exercise 3 (Difficulty: FUN)

# Consider two vectors
# 
# ``` x1 = [1,2,3,4,5],  x2 = [6,7,8,9,10]```
# 
# Draw and build in TensorFlow the computational graph for the dot-product operation between the two vectors. If you don't know what a dot-product is you can check it here (we covered that in our introductory week) [](https://en.wikipedia.org/wiki/Dot_product).
# 
# Build it in two different ways:
# 
# A) Do it with loops. Build a computational graph that takes as input scalars and in the session/evaluation phase build a loop to go over all the inputs and then sums the results.
# 
# B) Do it in one shot with tensorflow. Build a computational graph that takes as input vectors and do the entire operation directly in TensorFlow. 
# 
# *Hint*: you can use in TensorFlow two methods: ```tf.reduce_sum(tf.multiply(x1, x2))``` or ```tf.matmul(x1, tf.reshape(x2, [-1, 1]))```. Try to understand why they work checking the official documentation.

# ## Difference between ```run``` and ```eval```

# Consider the following code:

# In[ ]:


x1 = tf.constant(1)
x2 = tf.constant(2)
z = tf.add(x1, x2)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
sess.run(z)


# You can evaluate more nodes at the same time.

# In[ ]:


sess.run([x1,x2,z])


# You can also use the ```eval()``` method to evaluate nodes, but you have to provide the session name.

# In[ ]:


z.eval(session = sess)


# ## Dependencies between Nodes

# Sometime TensorFlow may evaluate some nodes multiple times. Pay attention. Consider the following code:

# In[ ]:


c = tf.constant(5)
x = c + 1
y = x + 1
z = x + 2
sess = tf.Session()
print('y is',sess.run(y))
print('z is',sess.run(z))
sess.close()


# Here ```x``` is evaluated twice! You can do it like this:

# In[ ]:


c = tf.constant(5)
x = c + 1
y = x + 1
z = x + 2
sess = tf.Session()
print('y,z are',sess.run([y,z]))
sess.close()


# Now ```x``` is evaluated only once.

# ## How to Close a Session

# You can close a session in this way:

# In[ ]:


sess = tf.Session()
# Do something here
sess.close()


# You can also do it in this way:

# In[ ]:


with tf.Session() as sess:
    # Do something here


# You need some code to avoid the error. Consider the following code:

# In[ ]:


c = tf.constant(5)
x = c + 1
y = x + 1
z = x + 2
sess = tf.Session()
print('y is',sess.run(y))
print('z is',sess.run(z))
sess.close()


# This can also be written as the following:

# In[ ]:


c = tf.constant(5)
x = c + 1
y = x + 1
z = x + 2
with tf.Session() as sess:
    print('y is',sess.run(y))
    print('z is',sess.run(z))


# Now you don't need to close the session explicitly. Is done for you.

# ## Exercises

# ### Exercise 4 (Difficulty: medium)

# Write a function that build a computational graph for the operation ```x1+x2``` where the input ```x1``` and ```x2``` are input with given dimensions. Your ```x1``` and ```x2``` should be declared as ```tf.placeholder```. 
# Your functions should accept as input:
# 
# - dimensions of ```x1``` as list, for example ```[3]```
# - dimensions of ```x2``` as list, for example ```[3]```
# 
# The function should return a tensor ```z = x1 + x2```. 
# Then open a session and evaluate ```z``` with the following inputs:
# 
# - ```x1 = [4,6,7], x2 = [1,2,9]```
# - ```x1 = [1,2,....., 1000], x2 = [10001, 10002, ...., 11000]```
# 
# and print the result.

# ### Exercise 5 (Difficult: FUN)

# ### Linear Regression with TensorFlow

# https://onlinecourses.science.psu.edu/stat501/node/382/

# Consider the following dataset:

# In[ ]:


x = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
y = [33, 42, 45, 51, 53, 61, 62]


# We want to find the best parameters $p_0$ and $p_1$ that minimise the MSE (Mean Squared Error) for the data given, in other words we want to do a linear regression on the data $(x,y)$. Given that a matrix solution to find the best parameter is
# 
# $$
# {\bf p} =(X^TX)^{-1} X^T Y
# $$
# 
# where $X^T$ is the transpose of the matrix $X$. The matrix $X$ is defined as
# 
# $$
# X = 
# \begin{bmatrix}
# 1 & x_1  \\
# ... &  ...  \\
# 1 &  x_n 
# \end{bmatrix}
# $$
# 
# The matrix $Y$ is simply a matrix $n\times 1$ containing the values $y_i$.
# 
# Dimensions are:
# 
# - $X$ has dimensions $n\times 2$
# - $Y$ has dimensions $n\times 1$
# - ${\bf p}$ has dimensions $2\times 1$

# Build a computational graph that evaluates $\bf p$ as given above, given the matrices $X$ and $Y$. Note you will have to build the matrices from the data given at the beginning. If you need more information a beatifully long explanation can be found here https://onlinecourses.science.psu.edu/stat501/node/382/.

# In[ ]:




