from typing import Any, Tuple
import tensorflow as tf
import numpy as np
import scipy

"""
Network component wrappers
"""

def load_arch(name:str, **kwargs:Any) -> tf.keras.models.Model:
    """ Loads a keras-supported pre-defined architecture """
    try:
        p = {k:v for k,v in locals().items() if k != 'name'}['kwargs']
        arch  = tf.keras.applications.__dict__[name.lower()].__dict__[name]
        model = arch(**p)
        return model
    except:
        print('[ERROR] Could not load model.')

def dense(x, neurons, activation='linear', dropout=.0, **kwargs):
    y = tf.keras.layers.Dropout(dropout)(x) if dropout > 0 else x
    y = tf.keras.layers.Dense(neurons, activation, **kwargs)(y)
    return y

def conv3x3(x, filters, kernel_size=(3,3), batch_norm=False, **kwargs):
    """ 2D convolutional layer with 3x3 kernel. """
    y = tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', padding='same', **kwargs)(x)
    if batch_norm:
        y = tf.keras.layers.BatchNormalization()(y)
    return y

def resbneck(x, filters, kernel=(3,3), downsampling=None, batch_norm=True, name=None):
    """ Residual bottleneck with pre-activation. """
    _bn   = lambda x: tf.keras.layers.BatchNormalization()(x)
    _relu = lambda x: tf.keras.layers.Activation('relu')(x)
    _add  = lambda x: tf.keras.layers.Add()(x)
    _conv = lambda x,f,k,s: tf.keras.layers.Conv2D(f,k,s,'same')(x)
    _name = lambda x,n: tf.keras.layers.Lambda(lambda x: x, name=n)(x)

    # Pre-activation
    y =  _relu(_bn(x)) if batch_norm else _relu(x)

    # Shortcut projection
    if downsampling == True:
        skip = _conv(y, 4*filters, 1, 2)
    elif downsampling == 'blurpool':
        skip = blurpool(_conv(y, 4*filters, 1, 1))
    else:
        skip = _conv(y, 4*filters, 1, 1)

    y =  _relu(_bn(_conv(y, filters, 1, 1))) if batch_norm else \
         _relu(_conv(y, filters, 1, 1))
    y =  _relu(_bn(_conv(y, filters, kernel, 1))) if batch_norm else \
         _relu(_conv(y, filters, kernel, 1))

    if downsampling == True:
        y = _conv(y, 4*filters, 1, 2)
    elif downsampling == 'blurpool':
        y = blurpool(_conv(y, 4*filters, 1, 1))
    else:
        y = _conv(y, 4*filters, 1, 1)
    y = _add([skip,y])

    if name:
        y = _name(y, name) 
    return y

def conv1x1(x, filters, kernel_size=(1,1), activation='relu', batch_norm=False, name=None):
    """ 2D convolutional layer with 1x1 kernel, i.e., channel-wise convolution. """
    _name = lambda x,n: tf.keras.layers.Lambda(lambda x: x, name=n)(x)

    if activation == 'leakyrelu':
        y = tf.keras.layers.Conv2D(filters, kernel_size, activation='linear', padding='same')(x)
        y = tf.keras.layers.LeakyReLU(alpha=0.1)(y)
    else: 
        y = tf.keras.layers.Conv2D(filters, kernel_size, activation=activation, padding='same')(x)
    if batch_norm:
        y = tf.keras.layers.BatchNormalization()(y)
    if name:
        y = _name(y, name) 
    return y

def deconv(x, filters, kernel_size=(4,4), strides=(2,2), batch_norm=False, **kwargs):
    """ Transposed 2D convolution, aka 'deconvolution' """
    _deconv = lambda x,f,k,s: tf.keras.layers.Conv2DTranspose(f,k,s,'same')(x)
    _lrelu  = lambda x: tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    _bn     = lambda x: tf.keras.layers.BatchNormalization()(x)
    _kwargs = lambda x: tf.keras.layers.Lambda(lambda x: x, **kwargs)(x)

    y = _lrelu(_deconv(x, filters, kernel_size, strides))
    y = _bn(y) if batch_norm else y

    # Allows the layer to be named
    y = _kwargs(y, **kwargs)  
    return y

def upsampling_zhang(x, upscale_factor=2, batch_norm=True):
    """ Upsampling block (loosely) based on Joint Task-Recursive Learning for
    Semantic Segmentation and Depth Estimation, Zhang et al., ECCV 2018. In
    their task, it was shown to be more effective than standard deconv.
    """
    _lrelu = lambda x: tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    _bn    = lambda x: tf.keras.layers.BatchNormalization()(x)
    _conv1 = lambda x,f,k: tf.keras.layers.Conv2D(f,k,1,'same')(x)
    _conv2 = lambda x,f: tf.keras.layers.Conv2D(f,3,1,'same',None,(2,2))(x)

    # Parallel conv. branches for multiscale receptive fields
    channels = x.shape[-1]
    d = 4
    c1 = _bn(_lrelu(_conv1(x, channels//d, 1))) if batch_norm else\
         _lrelu(_conv1(x, channels//d, 1))
    c2 = _bn(_lrelu(_conv1(x, channels//2, 3))) if batch_norm else\
         _lrelu(_conv1(x, channels//d, 3))
    c3 = _bn(_lrelu(_conv1(x, channels//2, 5))) if batch_norm else\
         _lrelu(_conv1(x, channels//d, 5))
    c4 = _bn(_lrelu(_conv2(x, channels//2))) if batch_norm else\
         _lrelu(_conv2(x, channels//d))
    y  = tf.keras.layers.Concatenate()([c1,c2,c3,c4]) 
    
    # Upsampling with sub-pixel convolution
    y = _lrelu(_conv1(y, channels//d*(upscale_factor**2), 3))
    y = tf.nn.depth_to_space(y, upscale_factor)
    return y

def decoder_block(x, out_depth, upscale_factor=1, batch_norm=True):
    """ Inception-like block with bilinear additive upsampling """
    _lrelu = lambda x: tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    _bn    = lambda x: tf.keras.layers.BatchNormalization()(x)
    _conv1 = lambda x,f,k: tf.keras.layers.Conv2D(f,k,1,'same')(x)
    _conv2 = lambda x,f:   tf.keras.layers.Conv2D(f,3,1,'same',None,(2,2))(x)

    # Parallel conv. branches for multiscale receptive fields
    c1 = _bn(_lrelu(_conv1(x, out_depth//2, 1))) if batch_norm else\
         _lrelu(_conv1(x, out_depth//2, 1))
    c2 = _bn(_lrelu(_conv1(x, out_depth//2, 3))) if batch_norm else\
         _lrelu(_conv1(x, out_depth//2, 3))
    c3 = _bn(_lrelu(_conv1(x, out_depth//2, 5))) if batch_norm else\
         _lrelu(_conv1(x, out_depth//2, 5))
    c4 = _bn(_lrelu(_conv2(x, out_depth//2))) if batch_norm else\
         _lrelu(_conv2(x, out_depth//2))
    y  = tf.keras.layers.Concatenate()([c1,c2,c3,c4]) 

    # Bilinear additive upsampling for parameter-free upscaling/depth reduction
    in_depth = x.shape[-1]
    step = in_depth//out_depth
    y = tf.keras.layers.UpSampling2D(upscale_factor, interpolation='bilinear')(y) 
    y_list = [tf.reduce_sum(y[:,:,:,i*step:(i+1)*step], axis=-1) 
              for i in range(out_depth)]
    y = tf.stack(y_list, axis=-1) 
    return y

def decoder_block_subpixel(x, out_depth, upscale_factor=1, batch_norm=True, name=None):
    """ Inception-like block with bilinear additive upsampling 
        (depthwiseconv version)
    """
    _lrelu = lambda x: tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    _bn    = lambda x: tf.keras.layers.BatchNormalization()(x)
    _conv1 = lambda x,f,k: tf.keras.layers.SeparableConv2D(f,k,1,'same')(x)
    _conv2 = lambda x,f:   tf.keras.layers.SeparableConv2D(f,3,1,'same',None,(2,2))(x)
    _name  = lambda x,n: tf.keras.layers.Lambda(lambda x: x, name=n)(x)

    # Parallel conv. branches for multiscale receptive fields
    c1 = _bn(_lrelu(_conv1(x, out_depth//2, 1))) if batch_norm else\
         _lrelu(_conv1(x, out_depth//2, 1))
    c2 = _bn(_lrelu(_conv1(x, out_depth//2, 3))) if batch_norm else\
         _lrelu(_conv1(x, out_depth//2, 3))
    c3 = _bn(_lrelu(_conv1(x, out_depth//2, 5))) if batch_norm else\
         _lrelu(_conv1(x, out_depth//2, 5))
    c4 = _bn(_lrelu(_conv2(x, out_depth//2))) if batch_norm else\
         _lrelu(_conv2(x, out_depth//2))
    y  = tf.keras.layers.Concatenate()([c1,c2,c3,c4]) 

    # Sub-pixel convolution
    y = _lrelu(_conv1(y, out_depth*upscale_factor**2, 3))
    y = tf.nn.depth_to_space(y, upscale_factor)

    if name:
        y = _name(y, name)
    return y

def upsampling_subpixel(x, upscale_factor=2, filter_num=16):
    _lrelu = lambda x: tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    _conv  = lambda x,f,k: tf.keras.layers.Conv2D(f,k,1,'same')(x)
    y = _lrelu(_conv(x, filter_num*upscale_factor**2, 3))
    y = tf.nn.depth_to_space(y, upscale_factor)
    return y

def bilinear_addupsampling(x, upscale_factor=2, out_depth=32):
    """ Bilinear Additive Upsampling. Follows the general idea introduced in:
        'The Devil is in the Decoder'. Wojna et al., IJCV 2019.
    """
    in_depth = x.shape[-1]
    step = in_depth//out_depth
    y = tf.keras.layers.UpSampling2D(upscale_factor, interpolation='bilinear')(x) 
    y_list = [tf.reduce_sum(y[:,:,:,i*step:(i+1)*step], axis=-1) 
              for i in range(out_depth)]
    y = tf.stack(y_list, axis=-1) 
    return y

def upconv(x, filters, size=(2,2), interpolation='nearest', batch_norm=False):
    y = tf.keras.layers.Conv2D(filters,(3,3),1,'same')(x)
    if batch_norm:
        y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.UpSampling2D(size, interpolation=interpolation)(y)    
    return y

def leaky_relu(x, alpha=0.1, **kwargs):
    return tf.keras.layers.LeakyReLU(alpha=0.1, **kwargs)(x)

def globalavgpool(x, **kwargs):
    """ Global average pooling """
    y = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last', **kwargs)(x)
    return y

def maxpool(x, pool_size=(2,2)):
    """ Max pooling """
    y = tf.keras.layers.MaxPooling2D(pool_size, padding='same')(x)
    return y

def blurpool(x):
    """ Blur pool w/ strided convolution"""
    def _gaussian_kernel(depth,ksz=5):
        """ Discrete approximation of Gaussian kernel with binomial coefs. """
        kernel_1d = scipy.special.binom(ksz-1,np.arange(ksz))
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        kernel_2d /= kernel_2d.sum()
        kernel_3d = np.zeros((5,5,depth,1), dtype='float32')
        for i in range(depth):
            kernel_3d[:,:,i,0] = kernel_2d
        return tf.constant(kernel_3d)
    
    x_padded = tf.pad(x, [[0,0],[2,2],[2,2],[0,0]], 'reflect')
    y = tf.nn.depthwise_conv2d(x_padded, _gaussian_kernel(x.shape[-1]), strides=[1,2,2,1], padding='VALID')
    return y

def maxblurpool(x):
    """ Blur pool w/ max pooling"""
    def _gaussian_kernel(depth,ksz=5):
        """ Discrete approximation of Gaussian kernel with binomial coefs. """
        kernel_1d = scipy.special.binom(ksz-1,np.arange(ksz))
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        kernel_2d /= kernel_2d.sum()
        kernel_3d = np.zeros((5,5,depth,1), dtype='float32')
        for i in range(depth):
            kernel_3d[:,:,i,0] = kernel_2d
        return tf.constant(kernel_3d)
    
    x_padded = tf.pad(x, [[0,0],[2,2],[2,2],[0,0]], 'reflect')
    y = tf.nn.depthwise_conv2d(x_padded, _gaussian_kernel(x.shape[-1]), strides=[1,1,1,1], padding='VALID')
    y = tf.keras.layers.MaxPooling2D((2,2), padding='same')(x)
    return y

def avgblurpool(x):
    """ Blur pool w/ average pooling"""
    def _gaussian_kernel(depth,ksz=5):
        """ Discrete approximation of Gaussian kernel with binomial coefs. """
        kernel_1d = scipy.special.binom(ksz-1,np.arange(ksz))
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        kernel_2d /= kernel_2d.sum()
        kernel_3d = np.zeros((5,5,depth,1), dtype='float32')
        for i in range(depth):
            kernel_3d[:,:,i,0] = kernel_2d
        return tf.constant(kernel_3d)
    
    x_padded = tf.pad(x, [[0,0],[2,2],[2,2],[0,0]], 'reflect')
    y = tf.nn.depthwise_conv2d(x_padded, _gaussian_kernel(x.shape[-1]), strides=[1,1,1,1], padding='VALID')
    y = tf.keras.layers.AveragePooling2D((2,2), padding='same')(x)
    return y

def flatten(x):
    y = tf.keras.layers.Flatten()(x)
    return y

def resize(x, out_shape, method='nearest'):
    y = tf.image.resize(x, out_shape, method)
    return y

def upsample(x, size, interpolation='nearest'):
    y = tf.keras.layers.UpSampling2D(size, interpolation=interpolation)(x)
    return y

def rename(x, name):
    return tf.keras.layers.Lambda(lambda x: x, name=name)(x)

def model(inputs, outputs):
    return tf.keras.Model(inputs, outputs)

def input(shape=None):
    return tf.keras.layers.Input(shape)

def shape(x):
    y = tf.keras.backend.shape(x)
    return y

def spatial_dropout(x, rate):
    y = tf.keras.layers.SpatialDropout2D(rate, data_format='channels_last')(x)
    return y

def concatenate(tensor_list):
    y = tf.keras.layers.Concatenate()(tensor_list)
    return y

def add(tensor_list):
    y = tf.keras.layers.Add()(tensor_list)
    return y
