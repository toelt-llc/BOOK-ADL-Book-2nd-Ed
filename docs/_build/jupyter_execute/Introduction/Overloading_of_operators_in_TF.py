#!/usr/bin/env python
# coding: utf-8

# # Operator Overloading in TensorFlow
# 
# Version 1.02
# 
# (C) 2020 - Umberto Michelucci, Michela Sperti
# 
# This notebook is part of the book _Applied Deep Learning: a case based approach, **2nd edition**_ from APRESS by [U. Michelucci](mailto:umberto.michelucci@toelt.ai) and [M. Sperti](mailto:michela.sperti@toelt.ai).

# ## Notebook Learning Goals
# 
# At the end of this notebook you will be able to undertsand the difference between operations like ```tf.multiply(x,y)``` and ```x*y``` when used with tensors or numpy arrays.

# ## Libraries Import

# In[ ]:


import tensorflow as tf
import numpy as np


# ## Operator Overloading

# What is the difference between ```tf.add(x,y)``` and ```x+y```? Or between ```tf.multiply(x,y)``` and ```x*y```?

# If at least one of the arguments ```x``` and ```y``` is a tensor (```tf.Tensor```) then   ```tf.add(x,y)``` and ```x+y``` are equivalent. The reason you may want to use ```tf.add()``` is to specify a ```name```  keyword argument for the created op, which is not possible with the overloaded operator version. The keyword will be used in tensorboard when visualising the computational graph. If you don't need it you can use the overloaded operator, meaning ```+```.

# **Note** that if neither ```x``` nor ```y``` is a ```tf.Tensor``` (for example if they are NumPy arrays) then ```x + y``` will not create a TensorFlow node. ```tf.add()``` always creates a TensorFlow node and converts its arguments to ```tf.Tensor``` objects. 

# The following operators are overloaded in the TensorFlow Python API:
# 
# __neg__ (unary -)
# 
# __abs__ (abs())
# 
# __invert__ (unary ~)
# 
# __add__ (binary +)
# 
# __sub__ (binary -)
# 
# __mul__ (binary elementwise *)
# 
# __div__ (binary / in Python 2)
# 
# __floordiv__ (binary // in Python 3)
# 
# __truediv__ (binary / in Python 3)
# 
# __mod__ (binary %)
# 
# __pow__ (binary **)
# 
# __and__ (binary &)
# 
# __or__ (binary |)
# 
# __xor__ (binary ^)
# 
# __lt__ (binary <)
# 
# __le__ (binary <=)
# 
# __gt__ (binary >)
# 
# __ge__ (binary >=)
# 
# See: https://www.tensorflow.org/api_docs/cc/group/math-ops.

# __eq__ ( binary == ) 
# 
# is not overloaded. ```x == y``` will simply return a Python boolean array. You need to use ```tf.equal()``` to do element-wise equality checks. 

# ## Programming Style
# 
# On the official documentation you can find the following style guideline:
# 
#     Prefer using a Tensor's overloaded operators rather than TensorFlow functions. For example, prefer **, +, /, *, - and/or over tf.pow, tf.add, tf.div, tf.mul, tf.subtract, and tf.logical * unless a specific name for the operation is desired. 
# 
# Again ```name``` is referred to the use of TensorBoard.
# 
# **REMEMBER**: if your inputs are not tensors, you may end up with other datatypes other than tensors. With the ```tf.XXXX``` functions you always have a tensor at the end.

# ## Example

# In[ ]:


x = np.array([1, 2, 3])
y = np.array([3, 4, 5])


# In[ ]:


x + y


# The sum is a numpy array.

# In[ ]:


type(x + y)


# In[ ]:


z = tf.add(x, y)


# In[ ]:


type(z)


# In this case, also if the inputs were numpy arrays, the output is a tensor.
