---
title: Temporal Convolutional Networks
subtitle: A short introduction into how Temporal Convolutional Networks function

# Summary for listings and search engines
summary: An introduction into dilated causal convolutions, and a look into how Temporal Convolutional Networks (TCN) function.

# Link this post with a project
projects: []

# Date published
date: "2021-11-10T12:00:00Z"

# Date updated
lastmod: "2021-11-10T12:00:00Z"

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: false

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: 'Residual blcok'
  focal_point: ""
  placement: 2
  preview_only: false

authors:
- admin

tags:
- Academic

categories:
---

Temporal Convolutional Networks (TCN) which are a variaton of Convolutional Neural Networks (CNN), recently have been used by deep learning practitioners to solve time series tasks with promising and successful outcomes as seen here [CITE]. I for one have employed TCNs for detecting Arythmia in ECG signals with great success. In this short post I want to explain how these networks work, how they differ from normal CNNs and take a look into the computational workload.

For sake of illusration I will explain all of these concepts here in 1D, but they also work in higher dimensions. First let us look at normal a CNN, let's assume that we have one layer, which has a kernel size of 3 and 1 filter. And let's assume that we have a input time series that looks like the one here below:
![Time series example](uploads/time_series.PNG "Example of time series")

When we then want to apply the 1D convolution to this input time series we do the following: We take our kernel size, which is 3, and slide it over the input time series to produce a output time series. Now how does this actually look like? Let's look at the first output of the output time series and see how that is produced,
![Showing how first sample of output time series is formed](uploads/conv.gif "Showing how first sample of output time seris is formed")
We then slide the kernel over the whole input time series and get the following output:
![Output time series](uploads/time_series_output.PNG "Output time series")
Now first thing we notice is that the output time series is not the same length as the input time series. This is because we do not do any padding, and we can calculate the output length by the following formula:
$$
T_{out} = T_{in} - (k-1)
$$
Where $k$ is the kernel size. TCNs work in a very similar way, with one addidional factor which is called dilation. Dilation is a way to increase the receptive field size of the network, with low cost to the number of operations needed. Let's look at a similar 1D convolution as before, but here we add the factor of $D = 2$ where $D$ stands for dilation. Note that in normal CNNs, dilation is fixed at $1$:
![Showing how the first sample of output time series is formed with dilation.](uploads/dilated_1.gif "First sample formed with dilation")
![Showing how the second sample of output time series is formed with dilation.](uploads/dilated_2.gif "Second sample formed with dilation")
As we can see adding the factor of dilation into our simple convolutional example radically changes the output time series:
![Output time series dilated](uploads/time_series_dilated_output.PNG "Output time series dilated")
Another thing that has changed is the size of our output series, as it is now not of length 6 but of length 4. This is since our formula before changes slightly with the addition of a dilation factor:
$$
T_{out} = T_{in} - (k-1)*D
$$
We can also see that this also holds true for normal convolutions as the dilation there is simply $D = 1$. One thing I noted here above was the 'Receptive Field Size (RFS)'. This is essentially how much of the time series each output node sees for computation. In this simple case we have here above the formula is simply:
$$
RFS = (k-1) * D + 1
$$
For the first case our RFS was simply $RFS = 3$ since $D =1$ and $k = 3$. Now for the dilated case this RFS increases to $RFS = 5$. Often when working with time series problems we want our network to be able output a time-series that is causal, meaning that when we calculate each time step we do not look into the future. To do so we need to add zero padding on the left hand side of the input time series. The size of the padding depends on both the kernel size and the dilation factor:
$$
Padding = (k-1) * D
$$
![Output time series dilated causal](uploads/time_series_padded_dilated_output.PNG "Output time series dilated causal")
Having this causal padding introduces an output time series that is the same length as our previous one simply because we know that $T_{in}^* = T_{in} + (k-1) * D$ and plugging that into the formula above gives us $T_{out} = T_{in}$. 

The last building block we need to introduce to be able to fully introduce the TCN network is the Residual block. 
![Residual Block](uploads/residual_block.png "Residual block")
The residual block consists of two dilated causal convolutions with normaliztion, non-linear activation and dropout inbetween. These residual blocks are then stacked on top of each other to build a network that has a receptive field size that fits the task at hand. Note that in these TCN networks the dilation factor is exponentially increased the more blocks you add to the network. The calculation of the receptive field size then changes a bit and becomes:
$$
RFS = 1 + (2^L -1)(k-1)*2
$$
Where $L$ stands for the number of residual blocks that are stacked on top of each other. 

Now let's look at a code example of a TCN tackling a time series task (Both in PyTorch and Tensorflow/Keras). 

We will focus on the FordA dataset from the [UCR/UEA archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/). We base the data preprocessing of the one availale online from Keras, for further information on why certain steps in the data preprocessing were done please take a look at the source: See [here](https://keras.io/examples/timeseries/timeseries_classification_from_scratch/) 


```python
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
x_test, y_test = readucr(root_url + "FordA_TEST.tsv")
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
num_classes = len(np.unique(y_train))
print("We have " + str(num_classes) + " classes")
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]
y_train[y_train == -1] = 0
y_test[y_test == -1] = 0
```

    We have 2 classes


Next we make the TCN model, we have a time series of length 500 in the dataset so we must model the receptive field size of the network to be equal or more than 500. We have two variables we can change to influence the receptive field size, the kernel size and then the number of layers of the TCN.

We will work with a kernel size of 10 and 5 layers as that gives us a RFS = 1 + (2^5-1)(10-1)*2 = 559 > 500.




```python
from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.layers.convolutional import Conv1D
from keras.layers import Dropout, Add, Lambda, Flatten, Input, BatchNormalization


def TCN(nb_classes,Chans=1, Samples=500, layers=5, kernel_s=10,filt=10, dropout=0,activation='elu'):
    regRate=.25
    input1 = Input(shape = (Samples, Chans))
    x1 = Conv1D(filt,kernel_size=kernel_s,dilation_rate=1,activation=activation, padding = 'causal',kernel_initializer='he_uniform')(input1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(dropout)(x1)
    x1 = Conv1D(filt,kernel_size=kernel_s,dilation_rate=1,activation=activation, padding = 'causal',kernel_initializer='he_uniform')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(dropout)(x1)
    conv = Conv1D(filt,kernel_size=1,padding='same')(input1)
    added_1 = Add()([x1, conv])
    out = Activation(activation)(added_1)

    
    for i in range(layers-1):
        x = Conv1D(filt,kernel_size=kernel_s,dilation_rate=2**(i+1),activation=activation, padding = 'causal',kernel_initializer='he_uniform')(out)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Conv1D(filt,kernel_size=kernel_s,dilation_rate=2**(i+1),activation=activation, padding = 'causal',kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

        added = Add()([x, out])
        out = Activation(activation)(added)
    out = Lambda(lambda x: x[:,-1,:])(out)
    dense        = Dense(nb_classes, name = 'dense')(out)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1,outputs=softmax)

TCN_model_1 = TCN(nb_classes = 2,filt=5)
TCN_model_1.summary()

```

    Model: "model"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_1 (InputLayer)           [(None, 500, 1)]     0           []                               
                                                                                                      
     conv1d (Conv1D)                (None, 500, 5)       55          ['input_1[0][0]']                
                                                                                                      
     batch_normalization (BatchNorm  (None, 500, 5)      20          ['conv1d[0][0]']                 
     alization)                                                                                       
                                                                                                      
     dropout (Dropout)              (None, 500, 5)       0           ['batch_normalization[0][0]']    
                                                                                                      
     conv1d_1 (Conv1D)              (None, 500, 5)       255         ['dropout[0][0]']                
                                                                                                      
     batch_normalization_1 (BatchNo  (None, 500, 5)      20          ['conv1d_1[0][0]']               
     rmalization)                                                                                     
                                                                                                      
     dropout_1 (Dropout)            (None, 500, 5)       0           ['batch_normalization_1[0][0]']  
                                                                                                      
     conv1d_2 (Conv1D)              (None, 500, 5)       10          ['input_1[0][0]']                
                                                                                                      
     add (Add)                      (None, 500, 5)       0           ['dropout_1[0][0]',              
                                                                      'conv1d_2[0][0]']               
                                                                                                      
     activation (Activation)        (None, 500, 5)       0           ['add[0][0]']                    
                                                                                                      
     conv1d_3 (Conv1D)              (None, 500, 5)       255         ['activation[0][0]']             
                                                                                                      
     batch_normalization_2 (BatchNo  (None, 500, 5)      20          ['conv1d_3[0][0]']               
     rmalization)                                                                                     
                                                                                                      
     dropout_2 (Dropout)            (None, 500, 5)       0           ['batch_normalization_2[0][0]']  
                                                                                                      
     conv1d_4 (Conv1D)              (None, 500, 5)       255         ['dropout_2[0][0]']              
                                                                                                      
     batch_normalization_3 (BatchNo  (None, 500, 5)      20          ['conv1d_4[0][0]']               
     rmalization)                                                                                     
                                                                                                      
     dropout_3 (Dropout)            (None, 500, 5)       0           ['batch_normalization_3[0][0]']  
                                                                                                      
     add_1 (Add)                    (None, 500, 5)       0           ['dropout_3[0][0]',              
                                                                      'activation[0][0]']             
                                                                                                      
     activation_1 (Activation)      (None, 500, 5)       0           ['add_1[0][0]']                  
                                                                                                      
     conv1d_5 (Conv1D)              (None, 500, 5)       255         ['activation_1[0][0]']           
                                                                                                      
     batch_normalization_4 (BatchNo  (None, 500, 5)      20          ['conv1d_5[0][0]']               
     rmalization)                                                                                     
                                                                                                      
     dropout_4 (Dropout)            (None, 500, 5)       0           ['batch_normalization_4[0][0]']  
                                                                                                      
     conv1d_6 (Conv1D)              (None, 500, 5)       255         ['dropout_4[0][0]']              
                                                                                                      
     batch_normalization_5 (BatchNo  (None, 500, 5)      20          ['conv1d_6[0][0]']               
     rmalization)                                                                                     
                                                                                                      
     dropout_5 (Dropout)            (None, 500, 5)       0           ['batch_normalization_5[0][0]']  
                                                                                                      
     add_2 (Add)                    (None, 500, 5)       0           ['dropout_5[0][0]',              
                                                                      'activation_1[0][0]']           
                                                                                                      
     activation_2 (Activation)      (None, 500, 5)       0           ['add_2[0][0]']                  
                                                                                                      
     conv1d_7 (Conv1D)              (None, 500, 5)       255         ['activation_2[0][0]']           
                                                                                                      
     batch_normalization_6 (BatchNo  (None, 500, 5)      20          ['conv1d_7[0][0]']               
     rmalization)                                                                                     
                                                                                                      
     dropout_6 (Dropout)            (None, 500, 5)       0           ['batch_normalization_6[0][0]']  
                                                                                                      
     conv1d_8 (Conv1D)              (None, 500, 5)       255         ['dropout_6[0][0]']              
                                                                                                      
     batch_normalization_7 (BatchNo  (None, 500, 5)      20          ['conv1d_8[0][0]']               
     rmalization)                                                                                     
                                                                                                      
     dropout_7 (Dropout)            (None, 500, 5)       0           ['batch_normalization_7[0][0]']  
                                                                                                      
     add_3 (Add)                    (None, 500, 5)       0           ['dropout_7[0][0]',              
                                                                      'activation_2[0][0]']           
                                                                                                      
     activation_3 (Activation)      (None, 500, 5)       0           ['add_3[0][0]']                  
                                                                                                      
     conv1d_9 (Conv1D)              (None, 500, 5)       255         ['activation_3[0][0]']           
                                                                                                      
     batch_normalization_8 (BatchNo  (None, 500, 5)      20          ['conv1d_9[0][0]']               
     rmalization)                                                                                     
                                                                                                      
     dropout_8 (Dropout)            (None, 500, 5)       0           ['batch_normalization_8[0][0]']  
                                                                                                      
     conv1d_10 (Conv1D)             (None, 500, 5)       255         ['dropout_8[0][0]']              
                                                                                                      
     batch_normalization_9 (BatchNo  (None, 500, 5)      20          ['conv1d_10[0][0]']              
     rmalization)                                                                                     
                                                                                                      
     dropout_9 (Dropout)            (None, 500, 5)       0           ['batch_normalization_9[0][0]']  
                                                                                                      
     add_4 (Add)                    (None, 500, 5)       0           ['dropout_9[0][0]',              
                                                                      'activation_3[0][0]']           
                                                                                                      
     activation_4 (Activation)      (None, 500, 5)       0           ['add_4[0][0]']                  
                                                                                                      
     lambda (Lambda)                (None, 5)            0           ['activation_4[0][0]']           
                                                                                                      
     dense (Dense)                  (None, 2)            12          ['lambda[0][0]']                 
                                                                                                      
     softmax (Activation)           (None, 2)            0           ['dense[0][0]']                  
                                                                                                      
    ==================================================================================================
    Total params: 2,572
    Trainable params: 2,472
    Non-trainable params: 100
    __________________________________________________________________________________________________


While looking at the summary of the TCN we see that we have roughly 2372 trainable parameters, we can decrease and increase this number by increasing the number of filters the network has. Let's first work with 5 filters and see how well the network does.

We do the same proccedure as done in the Keras example, with learning rate reduction and early stopping depending on validation loss.


```python
epochs = 500
batch_size = 32

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model_1.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]
TCN_model_1.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
history = TCN_model_1.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1,
)
```

    Epoch 1/500
    90/90 [==============================] - 11s 34ms/step - loss: 0.9717 - sparse_categorical_accuracy: 0.5448 - val_loss: 1.2359 - val_sparse_categorical_accuracy: 0.5520 - lr: 0.0010
    Epoch 2/500
    90/90 [==============================] - 2s 20ms/step - loss: 0.4918 - sparse_categorical_accuracy: 0.7351 - val_loss: 0.4279 - val_sparse_categorical_accuracy: 0.7961 - lr: 0.0010
    Epoch 3/500
    90/90 [==============================] - 2s 21ms/step - loss: 0.3400 - sparse_categorical_accuracy: 0.8604 - val_loss: 0.4122 - val_sparse_categorical_accuracy: 0.7864 - lr: 0.0010
    Epoch 4/500
    90/90 [==============================] - 1s 12ms/step - loss: 0.2839 - sparse_categorical_accuracy: 0.8847 - val_loss: 0.2717 - val_sparse_categorical_accuracy: 0.8835 - lr: 0.0010
    Epoch 5/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.2476 - sparse_categorical_accuracy: 0.9062 - val_loss: 0.2536 - val_sparse_categorical_accuracy: 0.8863 - lr: 0.0010
    Epoch 6/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.2162 - sparse_categorical_accuracy: 0.9135 - val_loss: 0.2366 - val_sparse_categorical_accuracy: 0.9071 - lr: 0.0010
    Epoch 7/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.1986 - sparse_categorical_accuracy: 0.9240 - val_loss: 0.3842 - val_sparse_categorical_accuracy: 0.8336 - lr: 0.0010
    Epoch 8/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.1970 - sparse_categorical_accuracy: 0.9219 - val_loss: 0.2794 - val_sparse_categorical_accuracy: 0.8904 - lr: 0.0010
    Epoch 9/500
    90/90 [==============================] - 1s 12ms/step - loss: 0.1869 - sparse_categorical_accuracy: 0.9240 - val_loss: 0.1988 - val_sparse_categorical_accuracy: 0.9279 - lr: 0.0010
    Epoch 10/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.1835 - sparse_categorical_accuracy: 0.9267 - val_loss: 0.2326 - val_sparse_categorical_accuracy: 0.9001 - lr: 0.0010
    Epoch 11/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.1794 - sparse_categorical_accuracy: 0.9323 - val_loss: 0.2173 - val_sparse_categorical_accuracy: 0.9168 - lr: 0.0010
    Epoch 12/500
    90/90 [==============================] - 1s 12ms/step - loss: 0.1654 - sparse_categorical_accuracy: 0.9337 - val_loss: 0.1764 - val_sparse_categorical_accuracy: 0.9362 - lr: 0.0010
    Epoch 13/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.1584 - sparse_categorical_accuracy: 0.9413 - val_loss: 0.1750 - val_sparse_categorical_accuracy: 0.9307 - lr: 0.0010
    Epoch 14/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.1565 - sparse_categorical_accuracy: 0.9399 - val_loss: 0.1992 - val_sparse_categorical_accuracy: 0.9168 - lr: 0.0010
    Epoch 15/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.1615 - sparse_categorical_accuracy: 0.9326 - val_loss: 0.1755 - val_sparse_categorical_accuracy: 0.9348 - lr: 0.0010
    Epoch 16/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.1412 - sparse_categorical_accuracy: 0.9438 - val_loss: 0.1823 - val_sparse_categorical_accuracy: 0.9390 - lr: 0.0010
    Epoch 17/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.1432 - sparse_categorical_accuracy: 0.9441 - val_loss: 0.1717 - val_sparse_categorical_accuracy: 0.9417 - lr: 0.0010
    Epoch 18/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.1354 - sparse_categorical_accuracy: 0.9455 - val_loss: 0.1762 - val_sparse_categorical_accuracy: 0.9390 - lr: 0.0010
    Epoch 19/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.1410 - sparse_categorical_accuracy: 0.9431 - val_loss: 0.2740 - val_sparse_categorical_accuracy: 0.8932 - lr: 0.0010
    Epoch 20/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.1378 - sparse_categorical_accuracy: 0.9507 - val_loss: 0.1701 - val_sparse_categorical_accuracy: 0.9390 - lr: 0.0010
    Epoch 21/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.1260 - sparse_categorical_accuracy: 0.9556 - val_loss: 0.1852 - val_sparse_categorical_accuracy: 0.9334 - lr: 0.0010
    Epoch 22/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.1253 - sparse_categorical_accuracy: 0.9500 - val_loss: 0.1607 - val_sparse_categorical_accuracy: 0.9473 - lr: 0.0010
    Epoch 23/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.1178 - sparse_categorical_accuracy: 0.9552 - val_loss: 0.2113 - val_sparse_categorical_accuracy: 0.9112 - lr: 0.0010
    Epoch 24/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.1159 - sparse_categorical_accuracy: 0.9569 - val_loss: 0.2236 - val_sparse_categorical_accuracy: 0.9196 - lr: 0.0010
    Epoch 25/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.1184 - sparse_categorical_accuracy: 0.9563 - val_loss: 0.2414 - val_sparse_categorical_accuracy: 0.8988 - lr: 0.0010
    Epoch 26/500
    90/90 [==============================] - 1s 12ms/step - loss: 0.1107 - sparse_categorical_accuracy: 0.9590 - val_loss: 0.1591 - val_sparse_categorical_accuracy: 0.9390 - lr: 0.0010
    Epoch 27/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0956 - sparse_categorical_accuracy: 0.9660 - val_loss: 0.1837 - val_sparse_categorical_accuracy: 0.9334 - lr: 0.0010
    Epoch 28/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0930 - sparse_categorical_accuracy: 0.9663 - val_loss: 0.3239 - val_sparse_categorical_accuracy: 0.8696 - lr: 0.0010
    Epoch 29/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.1013 - sparse_categorical_accuracy: 0.9576 - val_loss: 0.2096 - val_sparse_categorical_accuracy: 0.9196 - lr: 0.0010
    Epoch 30/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0934 - sparse_categorical_accuracy: 0.9667 - val_loss: 0.1962 - val_sparse_categorical_accuracy: 0.9293 - lr: 0.0010
    Epoch 31/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0909 - sparse_categorical_accuracy: 0.9674 - val_loss: 0.2964 - val_sparse_categorical_accuracy: 0.8932 - lr: 0.0010
    Epoch 32/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0758 - sparse_categorical_accuracy: 0.9764 - val_loss: 0.2026 - val_sparse_categorical_accuracy: 0.9223 - lr: 0.0010
    Epoch 33/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0796 - sparse_categorical_accuracy: 0.9701 - val_loss: 0.2081 - val_sparse_categorical_accuracy: 0.9196 - lr: 0.0010
    Epoch 34/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0787 - sparse_categorical_accuracy: 0.9701 - val_loss: 0.2105 - val_sparse_categorical_accuracy: 0.9209 - lr: 0.0010
    Epoch 35/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0729 - sparse_categorical_accuracy: 0.9729 - val_loss: 0.2151 - val_sparse_categorical_accuracy: 0.9279 - lr: 0.0010
    Epoch 36/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0653 - sparse_categorical_accuracy: 0.9785 - val_loss: 0.1987 - val_sparse_categorical_accuracy: 0.9209 - lr: 0.0010
    Epoch 37/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0704 - sparse_categorical_accuracy: 0.9726 - val_loss: 0.2914 - val_sparse_categorical_accuracy: 0.9029 - lr: 0.0010
    Epoch 38/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0908 - sparse_categorical_accuracy: 0.9660 - val_loss: 0.2507 - val_sparse_categorical_accuracy: 0.9015 - lr: 0.0010
    Epoch 39/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0830 - sparse_categorical_accuracy: 0.9663 - val_loss: 0.2209 - val_sparse_categorical_accuracy: 0.9140 - lr: 0.0010
    Epoch 40/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0590 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.2645 - val_sparse_categorical_accuracy: 0.9098 - lr: 0.0010
    Epoch 41/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0592 - sparse_categorical_accuracy: 0.9767 - val_loss: 0.2335 - val_sparse_categorical_accuracy: 0.9154 - lr: 0.0010
    Epoch 42/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0582 - sparse_categorical_accuracy: 0.9785 - val_loss: 0.2530 - val_sparse_categorical_accuracy: 0.9126 - lr: 0.0010
    Epoch 43/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0580 - sparse_categorical_accuracy: 0.9781 - val_loss: 0.2452 - val_sparse_categorical_accuracy: 0.9015 - lr: 0.0010
    Epoch 44/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0486 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.2654 - val_sparse_categorical_accuracy: 0.9112 - lr: 0.0010
    Epoch 45/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0534 - sparse_categorical_accuracy: 0.9809 - val_loss: 0.2631 - val_sparse_categorical_accuracy: 0.9112 - lr: 0.0010
    Epoch 46/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0519 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.2714 - val_sparse_categorical_accuracy: 0.9154 - lr: 0.0010
    Epoch 47/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0477 - sparse_categorical_accuracy: 0.9837 - val_loss: 0.2677 - val_sparse_categorical_accuracy: 0.9015 - lr: 5.0000e-04
    Epoch 48/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0396 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.2757 - val_sparse_categorical_accuracy: 0.9085 - lr: 5.0000e-04
    Epoch 49/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0340 - sparse_categorical_accuracy: 0.9917 - val_loss: 0.2726 - val_sparse_categorical_accuracy: 0.9085 - lr: 5.0000e-04
    Epoch 50/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0429 - sparse_categorical_accuracy: 0.9844 - val_loss: 0.2937 - val_sparse_categorical_accuracy: 0.9154 - lr: 5.0000e-04
    Epoch 51/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0377 - sparse_categorical_accuracy: 0.9889 - val_loss: 0.3136 - val_sparse_categorical_accuracy: 0.8974 - lr: 5.0000e-04
    Epoch 52/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0389 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.2947 - val_sparse_categorical_accuracy: 0.9112 - lr: 5.0000e-04
    Epoch 53/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0313 - sparse_categorical_accuracy: 0.9917 - val_loss: 0.2875 - val_sparse_categorical_accuracy: 0.9168 - lr: 5.0000e-04
    Epoch 54/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0363 - sparse_categorical_accuracy: 0.9896 - val_loss: 0.2955 - val_sparse_categorical_accuracy: 0.9154 - lr: 5.0000e-04
    Epoch 55/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0308 - sparse_categorical_accuracy: 0.9941 - val_loss: 0.2884 - val_sparse_categorical_accuracy: 0.9098 - lr: 5.0000e-04
    Epoch 56/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0231 - sparse_categorical_accuracy: 0.9941 - val_loss: 0.2833 - val_sparse_categorical_accuracy: 0.9112 - lr: 5.0000e-04
    Epoch 57/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0265 - sparse_categorical_accuracy: 0.9937 - val_loss: 0.3043 - val_sparse_categorical_accuracy: 0.9085 - lr: 5.0000e-04
    Epoch 58/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0221 - sparse_categorical_accuracy: 0.9958 - val_loss: 0.3091 - val_sparse_categorical_accuracy: 0.9043 - lr: 5.0000e-04
    Epoch 59/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0275 - sparse_categorical_accuracy: 0.9924 - val_loss: 0.3190 - val_sparse_categorical_accuracy: 0.9043 - lr: 5.0000e-04
    Epoch 60/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0250 - sparse_categorical_accuracy: 0.9910 - val_loss: 0.3475 - val_sparse_categorical_accuracy: 0.9001 - lr: 5.0000e-04
    Epoch 61/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0341 - sparse_categorical_accuracy: 0.9892 - val_loss: 0.3021 - val_sparse_categorical_accuracy: 0.9098 - lr: 5.0000e-04
    Epoch 62/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0216 - sparse_categorical_accuracy: 0.9941 - val_loss: 0.3185 - val_sparse_categorical_accuracy: 0.9154 - lr: 5.0000e-04
    Epoch 63/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0300 - sparse_categorical_accuracy: 0.9917 - val_loss: 0.3113 - val_sparse_categorical_accuracy: 0.9071 - lr: 5.0000e-04
    Epoch 64/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0214 - sparse_categorical_accuracy: 0.9944 - val_loss: 0.3153 - val_sparse_categorical_accuracy: 0.9071 - lr: 5.0000e-04
    Epoch 65/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0244 - sparse_categorical_accuracy: 0.9934 - val_loss: 0.3392 - val_sparse_categorical_accuracy: 0.9085 - lr: 5.0000e-04
    Epoch 66/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0260 - sparse_categorical_accuracy: 0.9934 - val_loss: 0.3157 - val_sparse_categorical_accuracy: 0.9140 - lr: 5.0000e-04
    Epoch 67/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0179 - sparse_categorical_accuracy: 0.9979 - val_loss: 0.3271 - val_sparse_categorical_accuracy: 0.9140 - lr: 2.5000e-04
    Epoch 68/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0221 - sparse_categorical_accuracy: 0.9937 - val_loss: 0.3323 - val_sparse_categorical_accuracy: 0.9140 - lr: 2.5000e-04
    Epoch 69/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0173 - sparse_categorical_accuracy: 0.9965 - val_loss: 0.3470 - val_sparse_categorical_accuracy: 0.9043 - lr: 2.5000e-04
    Epoch 70/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0131 - sparse_categorical_accuracy: 0.9986 - val_loss: 0.3312 - val_sparse_categorical_accuracy: 0.9098 - lr: 2.5000e-04
    Epoch 71/500
    90/90 [==============================] - 1s 13ms/step - loss: 0.0187 - sparse_categorical_accuracy: 0.9962 - val_loss: 0.3522 - val_sparse_categorical_accuracy: 0.9029 - lr: 2.5000e-04
    Epoch 72/500
    90/90 [==============================] - 1s 13ms/step - loss: 0.0255 - sparse_categorical_accuracy: 0.9927 - val_loss: 0.3590 - val_sparse_categorical_accuracy: 0.9098 - lr: 2.5000e-04
    Epoch 73/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0212 - sparse_categorical_accuracy: 0.9948 - val_loss: 0.3305 - val_sparse_categorical_accuracy: 0.9223 - lr: 2.5000e-04
    Epoch 74/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0162 - sparse_categorical_accuracy: 0.9969 - val_loss: 0.3392 - val_sparse_categorical_accuracy: 0.9098 - lr: 2.5000e-04
    Epoch 75/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0190 - sparse_categorical_accuracy: 0.9948 - val_loss: 0.3368 - val_sparse_categorical_accuracy: 0.9085 - lr: 2.5000e-04
    Epoch 76/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0171 - sparse_categorical_accuracy: 0.9969 - val_loss: 0.3364 - val_sparse_categorical_accuracy: 0.9043 - lr: 2.5000e-04
    Epoch 76: early stopping


We can see that our model does quite well, having a final validation accuracy of around 91%, looking at the training and validation accuracy graph we can see that it very quickly gets up to this 91% and then doesn't improve


```python
metric = "sparse_categorical_accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.show()
plt.close()
```


![png](./Untitled68_9_0.png)


Lets try increasing the number of filters the network has to see if that improves the performance.


```python
TCN_model_2 = TCN(nb_classes = 2,filt=15)
TCN_model_2.summary()
epochs = 500
batch_size = 32

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model_2.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]
TCN_model_2.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

```

    Model: "model_1"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_2 (InputLayer)           [(None, 500, 1)]     0           []                               
                                                                                                      
     conv1d_11 (Conv1D)             (None, 500, 15)      165         ['input_2[0][0]']                
                                                                                                      
     batch_normalization_10 (BatchN  (None, 500, 15)     60          ['conv1d_11[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     dropout_10 (Dropout)           (None, 500, 15)      0           ['batch_normalization_10[0][0]'] 
                                                                                                      
     conv1d_12 (Conv1D)             (None, 500, 15)      2265        ['dropout_10[0][0]']             
                                                                                                      
     batch_normalization_11 (BatchN  (None, 500, 15)     60          ['conv1d_12[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     dropout_11 (Dropout)           (None, 500, 15)      0           ['batch_normalization_11[0][0]'] 
                                                                                                      
     conv1d_13 (Conv1D)             (None, 500, 15)      30          ['input_2[0][0]']                
                                                                                                      
     add_5 (Add)                    (None, 500, 15)      0           ['dropout_11[0][0]',             
                                                                      'conv1d_13[0][0]']              
                                                                                                      
     activation_5 (Activation)      (None, 500, 15)      0           ['add_5[0][0]']                  
                                                                                                      
     conv1d_14 (Conv1D)             (None, 500, 15)      2265        ['activation_5[0][0]']           
                                                                                                      
     batch_normalization_12 (BatchN  (None, 500, 15)     60          ['conv1d_14[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     dropout_12 (Dropout)           (None, 500, 15)      0           ['batch_normalization_12[0][0]'] 
                                                                                                      
     conv1d_15 (Conv1D)             (None, 500, 15)      2265        ['dropout_12[0][0]']             
                                                                                                      
     batch_normalization_13 (BatchN  (None, 500, 15)     60          ['conv1d_15[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     dropout_13 (Dropout)           (None, 500, 15)      0           ['batch_normalization_13[0][0]'] 
                                                                                                      
     add_6 (Add)                    (None, 500, 15)      0           ['dropout_13[0][0]',             
                                                                      'activation_5[0][0]']           
                                                                                                      
     activation_6 (Activation)      (None, 500, 15)      0           ['add_6[0][0]']                  
                                                                                                      
     conv1d_16 (Conv1D)             (None, 500, 15)      2265        ['activation_6[0][0]']           
                                                                                                      
     batch_normalization_14 (BatchN  (None, 500, 15)     60          ['conv1d_16[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     dropout_14 (Dropout)           (None, 500, 15)      0           ['batch_normalization_14[0][0]'] 
                                                                                                      
     conv1d_17 (Conv1D)             (None, 500, 15)      2265        ['dropout_14[0][0]']             
                                                                                                      
     batch_normalization_15 (BatchN  (None, 500, 15)     60          ['conv1d_17[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     dropout_15 (Dropout)           (None, 500, 15)      0           ['batch_normalization_15[0][0]'] 
                                                                                                      
     add_7 (Add)                    (None, 500, 15)      0           ['dropout_15[0][0]',             
                                                                      'activation_6[0][0]']           
                                                                                                      
     activation_7 (Activation)      (None, 500, 15)      0           ['add_7[0][0]']                  
                                                                                                      
     conv1d_18 (Conv1D)             (None, 500, 15)      2265        ['activation_7[0][0]']           
                                                                                                      
     batch_normalization_16 (BatchN  (None, 500, 15)     60          ['conv1d_18[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     dropout_16 (Dropout)           (None, 500, 15)      0           ['batch_normalization_16[0][0]'] 
                                                                                                      
     conv1d_19 (Conv1D)             (None, 500, 15)      2265        ['dropout_16[0][0]']             
                                                                                                      
     batch_normalization_17 (BatchN  (None, 500, 15)     60          ['conv1d_19[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     dropout_17 (Dropout)           (None, 500, 15)      0           ['batch_normalization_17[0][0]'] 
                                                                                                      
     add_8 (Add)                    (None, 500, 15)      0           ['dropout_17[0][0]',             
                                                                      'activation_7[0][0]']           
                                                                                                      
     activation_8 (Activation)      (None, 500, 15)      0           ['add_8[0][0]']                  
                                                                                                      
     conv1d_20 (Conv1D)             (None, 500, 15)      2265        ['activation_8[0][0]']           
                                                                                                      
     batch_normalization_18 (BatchN  (None, 500, 15)     60          ['conv1d_20[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     dropout_18 (Dropout)           (None, 500, 15)      0           ['batch_normalization_18[0][0]'] 
                                                                                                      
     conv1d_21 (Conv1D)             (None, 500, 15)      2265        ['dropout_18[0][0]']             
                                                                                                      
     batch_normalization_19 (BatchN  (None, 500, 15)     60          ['conv1d_21[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     dropout_19 (Dropout)           (None, 500, 15)      0           ['batch_normalization_19[0][0]'] 
                                                                                                      
     add_9 (Add)                    (None, 500, 15)      0           ['dropout_19[0][0]',             
                                                                      'activation_8[0][0]']           
                                                                                                      
     activation_9 (Activation)      (None, 500, 15)      0           ['add_9[0][0]']                  
                                                                                                      
     lambda_1 (Lambda)              (None, 15)           0           ['activation_9[0][0]']           
                                                                                                      
     dense (Dense)                  (None, 2)            32          ['lambda_1[0][0]']               
                                                                                                      
     softmax (Activation)           (None, 2)            0           ['dense[0][0]']                  
                                                                                                      
    ==================================================================================================
    Total params: 21,212
    Trainable params: 20,912
    Non-trainable params: 300
    __________________________________________________________________________________________________



```python
history = TCN_model_2.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1,
)
```

    Epoch 1/500
    90/90 [==============================] - 5s 20ms/step - loss: 0.7026 - sparse_categorical_accuracy: 0.6823 - val_loss: 2.2428 - val_sparse_categorical_accuracy: 0.5368 - lr: 0.0010
    Epoch 2/500
    90/90 [==============================] - 1s 12ms/step - loss: 0.2933 - sparse_categorical_accuracy: 0.8691 - val_loss: 0.5623 - val_sparse_categorical_accuracy: 0.7933 - lr: 0.0010
    Epoch 3/500
    90/90 [==============================] - 1s 14ms/step - loss: 0.2257 - sparse_categorical_accuracy: 0.9080 - val_loss: 0.3179 - val_sparse_categorical_accuracy: 0.8682 - lr: 0.0010
    Epoch 4/500
    90/90 [==============================] - 1s 17ms/step - loss: 0.1778 - sparse_categorical_accuracy: 0.9260 - val_loss: 0.1919 - val_sparse_categorical_accuracy: 0.9071 - lr: 0.0010
    Epoch 5/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.1535 - sparse_categorical_accuracy: 0.9438 - val_loss: 0.2212 - val_sparse_categorical_accuracy: 0.9015 - lr: 0.0010
    Epoch 6/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.1351 - sparse_categorical_accuracy: 0.9493 - val_loss: 0.2222 - val_sparse_categorical_accuracy: 0.9029 - lr: 0.0010
    Epoch 7/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.1016 - sparse_categorical_accuracy: 0.9625 - val_loss: 0.2292 - val_sparse_categorical_accuracy: 0.9029 - lr: 0.0010
    Epoch 8/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.1056 - sparse_categorical_accuracy: 0.9601 - val_loss: 0.2194 - val_sparse_categorical_accuracy: 0.9015 - lr: 0.0010
    Epoch 9/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0770 - sparse_categorical_accuracy: 0.9740 - val_loss: 0.2100 - val_sparse_categorical_accuracy: 0.9043 - lr: 0.0010
    Epoch 10/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0673 - sparse_categorical_accuracy: 0.9792 - val_loss: 0.2207 - val_sparse_categorical_accuracy: 0.9015 - lr: 0.0010
    Epoch 11/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0683 - sparse_categorical_accuracy: 0.9747 - val_loss: 0.4673 - val_sparse_categorical_accuracy: 0.8530 - lr: 0.0010
    Epoch 12/500
    90/90 [==============================] - 1s 12ms/step - loss: 0.0579 - sparse_categorical_accuracy: 0.9799 - val_loss: 0.2410 - val_sparse_categorical_accuracy: 0.9015 - lr: 0.0010
    Epoch 13/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0393 - sparse_categorical_accuracy: 0.9885 - val_loss: 0.2252 - val_sparse_categorical_accuracy: 0.9057 - lr: 0.0010
    Epoch 14/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0351 - sparse_categorical_accuracy: 0.9917 - val_loss: 0.3696 - val_sparse_categorical_accuracy: 0.8835 - lr: 0.0010
    Epoch 15/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0310 - sparse_categorical_accuracy: 0.9948 - val_loss: 0.3575 - val_sparse_categorical_accuracy: 0.8835 - lr: 0.0010
    Epoch 16/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0279 - sparse_categorical_accuracy: 0.9924 - val_loss: 0.2712 - val_sparse_categorical_accuracy: 0.9057 - lr: 0.0010
    Epoch 17/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0322 - sparse_categorical_accuracy: 0.9906 - val_loss: 0.2862 - val_sparse_categorical_accuracy: 0.8946 - lr: 0.0010
    Epoch 18/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0186 - sparse_categorical_accuracy: 0.9948 - val_loss: 0.2878 - val_sparse_categorical_accuracy: 0.9043 - lr: 0.0010
    Epoch 19/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0209 - sparse_categorical_accuracy: 0.9951 - val_loss: 0.2709 - val_sparse_categorical_accuracy: 0.9071 - lr: 0.0010
    Epoch 20/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0344 - sparse_categorical_accuracy: 0.9903 - val_loss: 0.3604 - val_sparse_categorical_accuracy: 0.8890 - lr: 0.0010
    Epoch 21/500
    90/90 [==============================] - 1s 12ms/step - loss: 0.0341 - sparse_categorical_accuracy: 0.9889 - val_loss: 0.3389 - val_sparse_categorical_accuracy: 0.8918 - lr: 0.0010
    Epoch 22/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0139 - sparse_categorical_accuracy: 0.9976 - val_loss: 0.3452 - val_sparse_categorical_accuracy: 0.9029 - lr: 0.0010
    Epoch 23/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0109 - sparse_categorical_accuracy: 0.9979 - val_loss: 0.3448 - val_sparse_categorical_accuracy: 0.9043 - lr: 0.0010
    Epoch 24/500
    90/90 [==============================] - 1s 12ms/step - loss: 0.0131 - sparse_categorical_accuracy: 0.9976 - val_loss: 0.3958 - val_sparse_categorical_accuracy: 0.8877 - lr: 0.0010
    Epoch 25/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0074 - sparse_categorical_accuracy: 0.9993 - val_loss: 0.2737 - val_sparse_categorical_accuracy: 0.9112 - lr: 5.0000e-04
    Epoch 26/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0044 - sparse_categorical_accuracy: 0.9997 - val_loss: 0.2896 - val_sparse_categorical_accuracy: 0.9112 - lr: 5.0000e-04
    Epoch 27/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0032 - sparse_categorical_accuracy: 0.9997 - val_loss: 0.2763 - val_sparse_categorical_accuracy: 0.9098 - lr: 5.0000e-04
    Epoch 28/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0030 - sparse_categorical_accuracy: 1.0000 - val_loss: 0.2728 - val_sparse_categorical_accuracy: 0.9112 - lr: 5.0000e-04
    Epoch 29/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0038 - sparse_categorical_accuracy: 0.9997 - val_loss: 0.2767 - val_sparse_categorical_accuracy: 0.9140 - lr: 5.0000e-04
    Epoch 30/500
    90/90 [==============================] - 1s 12ms/step - loss: 0.0025 - sparse_categorical_accuracy: 1.0000 - val_loss: 0.2964 - val_sparse_categorical_accuracy: 0.8988 - lr: 5.0000e-04
    Epoch 31/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0028 - sparse_categorical_accuracy: 0.9997 - val_loss: 0.2838 - val_sparse_categorical_accuracy: 0.9112 - lr: 5.0000e-04
    Epoch 32/500
    90/90 [==============================] - 1s 12ms/step - loss: 0.0036 - sparse_categorical_accuracy: 0.9997 - val_loss: 0.2889 - val_sparse_categorical_accuracy: 0.9154 - lr: 5.0000e-04
    Epoch 33/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0020 - sparse_categorical_accuracy: 1.0000 - val_loss: 0.2818 - val_sparse_categorical_accuracy: 0.9126 - lr: 5.0000e-04
    Epoch 34/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0021 - sparse_categorical_accuracy: 1.0000 - val_loss: 0.3045 - val_sparse_categorical_accuracy: 0.9098 - lr: 5.0000e-04
    Epoch 35/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0018 - sparse_categorical_accuracy: 1.0000 - val_loss: 0.3019 - val_sparse_categorical_accuracy: 0.9001 - lr: 5.0000e-04
    Epoch 36/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0035 - sparse_categorical_accuracy: 0.9993 - val_loss: 0.2918 - val_sparse_categorical_accuracy: 0.9126 - lr: 5.0000e-04
    Epoch 37/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0025 - sparse_categorical_accuracy: 0.9993 - val_loss: 0.2996 - val_sparse_categorical_accuracy: 0.9071 - lr: 5.0000e-04
    Epoch 38/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0019 - sparse_categorical_accuracy: 1.0000 - val_loss: 0.2858 - val_sparse_categorical_accuracy: 0.9140 - lr: 5.0000e-04
    Epoch 39/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0013 - sparse_categorical_accuracy: 1.0000 - val_loss: 0.2985 - val_sparse_categorical_accuracy: 0.9112 - lr: 5.0000e-04
    Epoch 40/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0010 - sparse_categorical_accuracy: 1.0000 - val_loss: 0.3036 - val_sparse_categorical_accuracy: 0.9126 - lr: 5.0000e-04
    Epoch 41/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0091 - sparse_categorical_accuracy: 0.9962 - val_loss: 0.4849 - val_sparse_categorical_accuracy: 0.8835 - lr: 5.0000e-04
    Epoch 42/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0082 - sparse_categorical_accuracy: 0.9972 - val_loss: 0.3850 - val_sparse_categorical_accuracy: 0.9057 - lr: 5.0000e-04
    Epoch 43/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0099 - sparse_categorical_accuracy: 0.9955 - val_loss: 0.3593 - val_sparse_categorical_accuracy: 0.8974 - lr: 5.0000e-04
    Epoch 44/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0066 - sparse_categorical_accuracy: 0.9986 - val_loss: 0.4029 - val_sparse_categorical_accuracy: 0.9071 - lr: 5.0000e-04
    Epoch 45/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0023 - sparse_categorical_accuracy: 1.0000 - val_loss: 0.3584 - val_sparse_categorical_accuracy: 0.9029 - lr: 2.5000e-04
    Epoch 46/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0014 - sparse_categorical_accuracy: 1.0000 - val_loss: 0.3535 - val_sparse_categorical_accuracy: 0.9043 - lr: 2.5000e-04
    Epoch 47/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0016 - sparse_categorical_accuracy: 0.9997 - val_loss: 0.3562 - val_sparse_categorical_accuracy: 0.9085 - lr: 2.5000e-04
    Epoch 48/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0017 - sparse_categorical_accuracy: 1.0000 - val_loss: 0.3590 - val_sparse_categorical_accuracy: 0.9098 - lr: 2.5000e-04
    Epoch 49/500
    90/90 [==============================] - 1s 11ms/step - loss: 7.9658e-04 - sparse_categorical_accuracy: 1.0000 - val_loss: 0.3499 - val_sparse_categorical_accuracy: 0.9085 - lr: 2.5000e-04
    Epoch 50/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0022 - sparse_categorical_accuracy: 0.9993 - val_loss: 0.3977 - val_sparse_categorical_accuracy: 0.8904 - lr: 2.5000e-04
    Epoch 51/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0019 - sparse_categorical_accuracy: 0.9997 - val_loss: 0.3606 - val_sparse_categorical_accuracy: 0.9015 - lr: 2.5000e-04
    Epoch 52/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0014 - sparse_categorical_accuracy: 0.9997 - val_loss: 0.3612 - val_sparse_categorical_accuracy: 0.9071 - lr: 2.5000e-04
    Epoch 53/500
    90/90 [==============================] - 1s 11ms/step - loss: 9.6241e-04 - sparse_categorical_accuracy: 1.0000 - val_loss: 0.3539 - val_sparse_categorical_accuracy: 0.9071 - lr: 2.5000e-04
    Epoch 54/500
    90/90 [==============================] - 1s 12ms/step - loss: 8.3294e-04 - sparse_categorical_accuracy: 1.0000 - val_loss: 0.3502 - val_sparse_categorical_accuracy: 0.9071 - lr: 2.5000e-04
    Epoch 54: early stopping



```python
metric = "sparse_categorical_accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.show()
plt.close()
```


![png](./Untitled68_13_0.png)


Another trick we can try is to change the source code of the TCN model and instead of only looking at the last sample of each filter (the out layer) we instead flatten the network, this increases the number of parameters in network so let's again decrease the number of filters we use.


```python
def TCN(nb_classes,Chans=1, Samples=500, layers=5, kernel_s=10,filt=10, dropout=0,activation='elu'):
    regRate=.25
    input1 = Input(shape = (Samples, Chans))
    x1 = Conv1D(filt,kernel_size=kernel_s,dilation_rate=1,activation=activation, padding = 'causal',kernel_initializer='he_uniform')(input1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(dropout)(x1)
    x1 = Conv1D(filt,kernel_size=kernel_s,dilation_rate=1,activation=activation, padding = 'causal',kernel_initializer='he_uniform')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(dropout)(x1)
    conv = Conv1D(filt,kernel_size=1,padding='same')(input1)
    added_1 = Add()([x1, conv])
    out = Activation(activation)(added_1)

    
    for i in range(layers-1):
        x = Conv1D(filt,kernel_size=kernel_s,dilation_rate=2**(i+1),activation=activation, padding = 'causal',kernel_initializer='he_uniform')(out)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Conv1D(filt,kernel_size=kernel_s,dilation_rate=2**(i+1),activation=activation, padding = 'causal',kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

        added = Add()([x, out])
        out = Activation(activation)(added)
    out = Flatten()(out)
    dense        = Dense(nb_classes, name = 'dense')(out)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1,outputs=softmax)

TCN_model_3 = TCN(nb_classes = 2,filt=5)
TCN_model_3.summary()
```

    Model: "model_2"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_3 (InputLayer)           [(None, 500, 1)]     0           []                               
                                                                                                      
     conv1d_22 (Conv1D)             (None, 500, 5)       55          ['input_3[0][0]']                
                                                                                                      
     batch_normalization_20 (BatchN  (None, 500, 5)      20          ['conv1d_22[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     dropout_20 (Dropout)           (None, 500, 5)       0           ['batch_normalization_20[0][0]'] 
                                                                                                      
     conv1d_23 (Conv1D)             (None, 500, 5)       255         ['dropout_20[0][0]']             
                                                                                                      
     batch_normalization_21 (BatchN  (None, 500, 5)      20          ['conv1d_23[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     dropout_21 (Dropout)           (None, 500, 5)       0           ['batch_normalization_21[0][0]'] 
                                                                                                      
     conv1d_24 (Conv1D)             (None, 500, 5)       10          ['input_3[0][0]']                
                                                                                                      
     add_10 (Add)                   (None, 500, 5)       0           ['dropout_21[0][0]',             
                                                                      'conv1d_24[0][0]']              
                                                                                                      
     activation_10 (Activation)     (None, 500, 5)       0           ['add_10[0][0]']                 
                                                                                                      
     conv1d_25 (Conv1D)             (None, 500, 5)       255         ['activation_10[0][0]']          
                                                                                                      
     batch_normalization_22 (BatchN  (None, 500, 5)      20          ['conv1d_25[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     dropout_22 (Dropout)           (None, 500, 5)       0           ['batch_normalization_22[0][0]'] 
                                                                                                      
     conv1d_26 (Conv1D)             (None, 500, 5)       255         ['dropout_22[0][0]']             
                                                                                                      
     batch_normalization_23 (BatchN  (None, 500, 5)      20          ['conv1d_26[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     dropout_23 (Dropout)           (None, 500, 5)       0           ['batch_normalization_23[0][0]'] 
                                                                                                      
     add_11 (Add)                   (None, 500, 5)       0           ['dropout_23[0][0]',             
                                                                      'activation_10[0][0]']          
                                                                                                      
     activation_11 (Activation)     (None, 500, 5)       0           ['add_11[0][0]']                 
                                                                                                      
     conv1d_27 (Conv1D)             (None, 500, 5)       255         ['activation_11[0][0]']          
                                                                                                      
     batch_normalization_24 (BatchN  (None, 500, 5)      20          ['conv1d_27[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     dropout_24 (Dropout)           (None, 500, 5)       0           ['batch_normalization_24[0][0]'] 
                                                                                                      
     conv1d_28 (Conv1D)             (None, 500, 5)       255         ['dropout_24[0][0]']             
                                                                                                      
     batch_normalization_25 (BatchN  (None, 500, 5)      20          ['conv1d_28[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     dropout_25 (Dropout)           (None, 500, 5)       0           ['batch_normalization_25[0][0]'] 
                                                                                                      
     add_12 (Add)                   (None, 500, 5)       0           ['dropout_25[0][0]',             
                                                                      'activation_11[0][0]']          
                                                                                                      
     activation_12 (Activation)     (None, 500, 5)       0           ['add_12[0][0]']                 
                                                                                                      
     conv1d_29 (Conv1D)             (None, 500, 5)       255         ['activation_12[0][0]']          
                                                                                                      
     batch_normalization_26 (BatchN  (None, 500, 5)      20          ['conv1d_29[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     dropout_26 (Dropout)           (None, 500, 5)       0           ['batch_normalization_26[0][0]'] 
                                                                                                      
     conv1d_30 (Conv1D)             (None, 500, 5)       255         ['dropout_26[0][0]']             
                                                                                                      
     batch_normalization_27 (BatchN  (None, 500, 5)      20          ['conv1d_30[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     dropout_27 (Dropout)           (None, 500, 5)       0           ['batch_normalization_27[0][0]'] 
                                                                                                      
     add_13 (Add)                   (None, 500, 5)       0           ['dropout_27[0][0]',             
                                                                      'activation_12[0][0]']          
                                                                                                      
     activation_13 (Activation)     (None, 500, 5)       0           ['add_13[0][0]']                 
                                                                                                      
     conv1d_31 (Conv1D)             (None, 500, 5)       255         ['activation_13[0][0]']          
                                                                                                      
     batch_normalization_28 (BatchN  (None, 500, 5)      20          ['conv1d_31[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     dropout_28 (Dropout)           (None, 500, 5)       0           ['batch_normalization_28[0][0]'] 
                                                                                                      
     conv1d_32 (Conv1D)             (None, 500, 5)       255         ['dropout_28[0][0]']             
                                                                                                      
     batch_normalization_29 (BatchN  (None, 500, 5)      20          ['conv1d_32[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     dropout_29 (Dropout)           (None, 500, 5)       0           ['batch_normalization_29[0][0]'] 
                                                                                                      
     add_14 (Add)                   (None, 500, 5)       0           ['dropout_29[0][0]',             
                                                                      'activation_13[0][0]']          
                                                                                                      
     activation_14 (Activation)     (None, 500, 5)       0           ['add_14[0][0]']                 
                                                                                                      
     flatten (Flatten)              (None, 2500)         0           ['activation_14[0][0]']          
                                                                                                      
     dense (Dense)                  (None, 2)            5002        ['flatten[0][0]']                
                                                                                                      
     softmax (Activation)           (None, 2)            0           ['dense[0][0]']                  
                                                                                                      
    ==================================================================================================
    Total params: 7,562
    Trainable params: 7,462
    Non-trainable params: 100
    __________________________________________________________________________________________________



```python
epochs = 500
batch_size = 32

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model_3.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]
TCN_model_3.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
history = TCN_model_3.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1,
)
```

    Epoch 1/500
    90/90 [==============================] - 4s 18ms/step - loss: 0.8511 - sparse_categorical_accuracy: 0.6469 - val_loss: 3.3143 - val_sparse_categorical_accuracy: 0.5007 - lr: 0.0010
    Epoch 2/500
    90/90 [==============================] - 1s 12ms/step - loss: 0.3462 - sparse_categorical_accuracy: 0.8677 - val_loss: 0.5212 - val_sparse_categorical_accuracy: 0.8114 - lr: 0.0010
    Epoch 3/500
    90/90 [==============================] - 1s 12ms/step - loss: 0.2373 - sparse_categorical_accuracy: 0.9125 - val_loss: 0.2840 - val_sparse_categorical_accuracy: 0.8793 - lr: 0.0010
    Epoch 4/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.2195 - sparse_categorical_accuracy: 0.9146 - val_loss: 0.3031 - val_sparse_categorical_accuracy: 0.8974 - lr: 0.0010
    Epoch 5/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.2022 - sparse_categorical_accuracy: 0.9233 - val_loss: 0.3261 - val_sparse_categorical_accuracy: 0.8863 - lr: 0.0010
    Epoch 6/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.1752 - sparse_categorical_accuracy: 0.9288 - val_loss: 0.5190 - val_sparse_categorical_accuracy: 0.8363 - lr: 0.0010
    Epoch 7/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.1465 - sparse_categorical_accuracy: 0.9410 - val_loss: 0.3399 - val_sparse_categorical_accuracy: 0.8807 - lr: 0.0010
    Epoch 8/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.1204 - sparse_categorical_accuracy: 0.9552 - val_loss: 0.3331 - val_sparse_categorical_accuracy: 0.8863 - lr: 0.0010
    Epoch 9/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.1204 - sparse_categorical_accuracy: 0.9497 - val_loss: 0.3068 - val_sparse_categorical_accuracy: 0.8988 - lr: 0.0010
    Epoch 10/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.1112 - sparse_categorical_accuracy: 0.9573 - val_loss: 0.4133 - val_sparse_categorical_accuracy: 0.8585 - lr: 0.0010
    Epoch 11/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.1313 - sparse_categorical_accuracy: 0.9503 - val_loss: 0.8094 - val_sparse_categorical_accuracy: 0.8239 - lr: 0.0010
    Epoch 12/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.1284 - sparse_categorical_accuracy: 0.9510 - val_loss: 0.3401 - val_sparse_categorical_accuracy: 0.8863 - lr: 0.0010
    Epoch 13/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0947 - sparse_categorical_accuracy: 0.9635 - val_loss: 0.3506 - val_sparse_categorical_accuracy: 0.8918 - lr: 0.0010
    Epoch 14/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0781 - sparse_categorical_accuracy: 0.9715 - val_loss: 0.3986 - val_sparse_categorical_accuracy: 0.8821 - lr: 0.0010
    Epoch 15/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0800 - sparse_categorical_accuracy: 0.9677 - val_loss: 0.3956 - val_sparse_categorical_accuracy: 0.8835 - lr: 0.0010
    Epoch 16/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0860 - sparse_categorical_accuracy: 0.9649 - val_loss: 0.4759 - val_sparse_categorical_accuracy: 0.8724 - lr: 0.0010
    Epoch 17/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0737 - sparse_categorical_accuracy: 0.9708 - val_loss: 0.4114 - val_sparse_categorical_accuracy: 0.8807 - lr: 0.0010
    Epoch 18/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0814 - sparse_categorical_accuracy: 0.9670 - val_loss: 0.4156 - val_sparse_categorical_accuracy: 0.8988 - lr: 0.0010
    Epoch 19/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0514 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.3724 - val_sparse_categorical_accuracy: 0.8932 - lr: 0.0010
    Epoch 20/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0544 - sparse_categorical_accuracy: 0.9785 - val_loss: 0.4212 - val_sparse_categorical_accuracy: 0.8918 - lr: 0.0010
    Epoch 21/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0449 - sparse_categorical_accuracy: 0.9858 - val_loss: 0.4394 - val_sparse_categorical_accuracy: 0.8779 - lr: 0.0010
    Epoch 22/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0473 - sparse_categorical_accuracy: 0.9816 - val_loss: 0.4151 - val_sparse_categorical_accuracy: 0.8849 - lr: 0.0010
    Epoch 23/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0578 - sparse_categorical_accuracy: 0.9812 - val_loss: 0.4482 - val_sparse_categorical_accuracy: 0.8904 - lr: 0.0010
    Epoch 24/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0314 - sparse_categorical_accuracy: 0.9924 - val_loss: 0.4182 - val_sparse_categorical_accuracy: 0.8932 - lr: 5.0000e-04
    Epoch 25/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0209 - sparse_categorical_accuracy: 0.9937 - val_loss: 0.4111 - val_sparse_categorical_accuracy: 0.8946 - lr: 5.0000e-04
    Epoch 26/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0191 - sparse_categorical_accuracy: 0.9962 - val_loss: 0.3970 - val_sparse_categorical_accuracy: 0.9001 - lr: 5.0000e-04
    Epoch 27/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0156 - sparse_categorical_accuracy: 0.9965 - val_loss: 0.4019 - val_sparse_categorical_accuracy: 0.8974 - lr: 5.0000e-04
    Epoch 28/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0225 - sparse_categorical_accuracy: 0.9931 - val_loss: 0.4273 - val_sparse_categorical_accuracy: 0.8904 - lr: 5.0000e-04
    Epoch 29/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0168 - sparse_categorical_accuracy: 0.9969 - val_loss: 0.4179 - val_sparse_categorical_accuracy: 0.8988 - lr: 5.0000e-04
    Epoch 30/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0102 - sparse_categorical_accuracy: 0.9979 - val_loss: 0.4302 - val_sparse_categorical_accuracy: 0.8946 - lr: 5.0000e-04
    Epoch 31/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0093 - sparse_categorical_accuracy: 0.9990 - val_loss: 0.4508 - val_sparse_categorical_accuracy: 0.8932 - lr: 5.0000e-04
    Epoch 32/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0157 - sparse_categorical_accuracy: 0.9962 - val_loss: 0.4591 - val_sparse_categorical_accuracy: 0.8807 - lr: 5.0000e-04
    Epoch 33/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0170 - sparse_categorical_accuracy: 0.9951 - val_loss: 0.4427 - val_sparse_categorical_accuracy: 0.8807 - lr: 5.0000e-04
    Epoch 34/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0148 - sparse_categorical_accuracy: 0.9955 - val_loss: 0.5704 - val_sparse_categorical_accuracy: 0.8738 - lr: 5.0000e-04
    Epoch 35/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0142 - sparse_categorical_accuracy: 0.9958 - val_loss: 0.4625 - val_sparse_categorical_accuracy: 0.8960 - lr: 5.0000e-04
    Epoch 36/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0128 - sparse_categorical_accuracy: 0.9965 - val_loss: 0.5169 - val_sparse_categorical_accuracy: 0.8890 - lr: 5.0000e-04
    Epoch 37/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0082 - sparse_categorical_accuracy: 0.9990 - val_loss: 0.4603 - val_sparse_categorical_accuracy: 0.8974 - lr: 5.0000e-04
    Epoch 38/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0153 - sparse_categorical_accuracy: 0.9962 - val_loss: 0.4560 - val_sparse_categorical_accuracy: 0.8988 - lr: 5.0000e-04
    Epoch 39/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0132 - sparse_categorical_accuracy: 0.9965 - val_loss: 0.5157 - val_sparse_categorical_accuracy: 0.8738 - lr: 5.0000e-04
    Epoch 40/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0103 - sparse_categorical_accuracy: 0.9979 - val_loss: 0.5172 - val_sparse_categorical_accuracy: 0.8918 - lr: 5.0000e-04
    Epoch 41/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0100 - sparse_categorical_accuracy: 0.9983 - val_loss: 0.4915 - val_sparse_categorical_accuracy: 0.8877 - lr: 5.0000e-04
    Epoch 42/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0096 - sparse_categorical_accuracy: 0.9976 - val_loss: 0.4898 - val_sparse_categorical_accuracy: 0.8904 - lr: 5.0000e-04
    Epoch 43/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0122 - sparse_categorical_accuracy: 0.9962 - val_loss: 0.4642 - val_sparse_categorical_accuracy: 0.8946 - lr: 5.0000e-04
    Epoch 44/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0100 - sparse_categorical_accuracy: 0.9969 - val_loss: 0.4602 - val_sparse_categorical_accuracy: 0.8835 - lr: 2.5000e-04
    Epoch 45/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0059 - sparse_categorical_accuracy: 0.9997 - val_loss: 0.4580 - val_sparse_categorical_accuracy: 0.8932 - lr: 2.5000e-04
    Epoch 46/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0120 - sparse_categorical_accuracy: 0.9972 - val_loss: 0.4703 - val_sparse_categorical_accuracy: 0.8863 - lr: 2.5000e-04
    Epoch 47/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0085 - sparse_categorical_accuracy: 0.9986 - val_loss: 0.4532 - val_sparse_categorical_accuracy: 0.9001 - lr: 2.5000e-04
    Epoch 48/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0062 - sparse_categorical_accuracy: 0.9993 - val_loss: 0.4753 - val_sparse_categorical_accuracy: 0.8932 - lr: 2.5000e-04
    Epoch 49/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0063 - sparse_categorical_accuracy: 0.9990 - val_loss: 0.4990 - val_sparse_categorical_accuracy: 0.8890 - lr: 2.5000e-04
    Epoch 50/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0061 - sparse_categorical_accuracy: 0.9997 - val_loss: 0.4871 - val_sparse_categorical_accuracy: 0.8877 - lr: 2.5000e-04
    Epoch 51/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0062 - sparse_categorical_accuracy: 0.9993 - val_loss: 0.4996 - val_sparse_categorical_accuracy: 0.8904 - lr: 2.5000e-04
    Epoch 52/500
    90/90 [==============================] - 1s 11ms/step - loss: 0.0079 - sparse_categorical_accuracy: 0.9983 - val_loss: 0.5458 - val_sparse_categorical_accuracy: 0.8793 - lr: 2.5000e-04
    Epoch 53/500
    90/90 [==============================] - 1s 10ms/step - loss: 0.0065 - sparse_categorical_accuracy: 0.9990 - val_loss: 0.4834 - val_sparse_categorical_accuracy: 0.8877 - lr: 2.5000e-04
    Epoch 53: early stopping



```python
metric = "sparse_categorical_accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.show()
plt.close()
```


![png](./Untitled68_17_0.png)


But as always, the best thing is to do cross validated hyperparameter search to see how many filters you need to get a good performing network, for our case I believe that the second network we tested was the best one so let's look at the final test accuracy of that network:




```python
TCN_model_2 = keras.models.load_model("best_model_2.h5")

test_loss, test_acc = TCN_model_2.evaluate(x_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)
```

    42/42 [==============================] - 1s 6ms/step - loss: 0.2183 - sparse_categorical_accuracy: 0.9182
    Test accuracy 0.918181836605072
    Test loss 0.21826055645942688