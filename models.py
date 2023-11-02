import nn
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from sklearn.cross_decomposition import PLSRegression

# ---------------------------------------------------------------------------
def encoder_decoder_landmarks(filter_num):
        return singletask_encoder_decoder(filter_num, task='landmarks')

def encoder_decoder_shape(filter_num):
        return singletask_encoder_decoder(filter_num, task='shape')

@tf.keras.utils.register_keras_serializable()
class ResidualBottleneck(tf.keras.layers.Layer):
    def __init__(self, filters, kernel=(3,3), downsampling=None, batch_norm=True, **kwargs):
        #super(ResidualBottleneck, self).__init__(**kwargs)
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel  = kernel
        self.downsampling = downsampling
        self.batch_norm   = batch_norm

        self.relu = tf.keras.layers.Activation('relu')
        self.add  = tf.keras.layers.Add()
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.bn_3 = tf.keras.layers.BatchNormalization()
        if downsampling == True:
            self.conv2_1 = tf.keras.layers.Conv2D(4*self.filters, 1 ,2, 'same')
        else:
            self.conv2_1 = tf.keras.layers.Conv2D(4*self.filters, 1, 1, 'same')
        self.conv2_2 = tf.keras.layers.Conv2D(self.filters, 1, 1, 'same')
        self.conv2_3 = tf.keras.layers.Conv2D(self.filters, self.kernel, 1, 'same')
        if downsampling == True:
            self.conv2_4 = tf.keras.layers.Conv2D(4*self.filters, 1 ,2, 'same')
        else:
            self.conv2_4 = tf.keras.layers.Conv2D(4*self.filters, 1, 1, 'same')

    def call(self, inp, training=False):
        # Pre-activation
        y = self.relu(self.bn_1(inp, training=training)) if self.batch_norm else self.relu(inp)

        # Shortcut projection
        if self.downsampling == 'blurpool':
            skip = nn.blurpool(self.conv2_1(y))

        y = self.relu(self.bn_2(self.conv2_2(y), training=training)) if self.batch_norm else \
            self.relu(self.conv2_2(y))
        y = self.relu(self.bn_3(self.conv2_3(y), training=training)) if self.batch_norm else \
            self.relu(self.conv2_3(y))

        if self.downsampling == 'blurpool':
            y = nn.blurpool(self.conv2_4(y))
        y = self.add([skip,y])
        return y

    def get_config(self):
        config = super(ResidualBottleneck, self).get_config()
        config.update({'name':    self.name,
                'filters': self.filters,
                'kernel':  self.kernel,
                'downsampling': self.downsampling,
                'batch_norm':   self.batch_norm,})
        return config

@tf.keras.utils.register_keras_serializable()
class DecoderSubpixelBlock(tf.keras.layers.Layer):
    def __init__(self, out_depth, upscale_factor=1, batch_norm=True, **kwargs):
        #super(DecoderSubpixelBlock, self).__init__(**kwargs)
        super().__init__(**kwargs)
        self.out_depth      = out_depth
        self.upscale_factor = upscale_factor
        self.batch_norm     = batch_norm

        self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.cat   = tf.keras.layers.Concatenate()
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.bn_4 = tf.keras.layers.BatchNormalization()
        self.conv2_1 = tf.keras.layers.SeparableConv2D(self.out_depth//2, 1 ,1,'same')
        self.conv2_2 = tf.keras.layers.SeparableConv2D(self.out_depth//2, 3 ,1,'same')
        self.conv2_3 = tf.keras.layers.SeparableConv2D(self.out_depth//2, 5 ,1,'same')
        self.conv2_4 = tf.keras.layers.SeparableConv2D(self.out_depth//2, 3, 1, 'same', None, (2,2))
        self.conv2_5 = tf.keras.layers.SeparableConv2D(self.out_depth*self.upscale_factor**2, 3 ,1,'same')

    def call(self, inp, training=False):
        # Parallel conv. branches for multiscale receptive fields
        c1 = self.bn_1(self.lrelu(self.conv2_1(inp)), training=training) if self.batch_norm else\
             self.lrelu(self.conv2_1(inp))
        c2 = self.bn_2(self.lrelu(self.conv2_2(inp)), training=training) if self.batch_norm else\
             self.lrelu(self.conv2_2(inp))
        c3 = self.bn_3(self.lrelu(self.conv2_3(inp)), training=training) if self.batch_norm else\
             self.lrelu(self.conv2_3(inp))
        c4 = self.bn_4(self.lrelu(self.conv2_4(inp)), training=training) if self.batch_norm else\
             self.lrelu(self.conv2_4(inp))
        y  = self.cat([c1,c2,c3,c4]) 

        y = self.lrelu(self.conv2_5(y))
        y = tf.nn.depth_to_space(y, self.upscale_factor)
        return y

    def get_config(self):
        config = super(DecoderSubpixelBlock, self).get_config()
        config.update({'name':           self.name,
                       'out_depth':      self.out_depth,
                       'upscale_factor': self.upscale_factor,
                       'batch_norm':     self.batch_norm,})
        return config

def singletask_encoder_decoder(filter_num=128, input_depth=3, task='landmarks'):
    inp = nn.input(shape=(None,None,input_depth,))

    # Input projection (to match the depth between input and encoder)
    d0 = tf.keras.layers.Conv2D(filter_num//8,
             kernel_size=1,
             activation='relu', 
             padding='same',
             name='input_proj_'+task)(inp)

    # Encoder ----------------------------------------------------------------
    d1 = ResidualBottleneck(filter_num//8,
             batch_norm=True,
             downsampling='blurpool',
             name='encoder_blk_1_' + task)(d0)
    d2 = ResidualBottleneck(filter_num//4,
             batch_norm=True,
             downsampling='blurpool',
             name='encoder_blk_2_' + task)(d1)
    d3 = ResidualBottleneck(filter_num//2,
             batch_norm=True,
             downsampling='blurpool',
             name='encoder_blk_3_' + task)(d2)
    d4 = ResidualBottleneck(filter_num,
             batch_norm=True,
             downsampling='blurpool',
             name='encoder_blk_4_' + task)(d3)

    # Decoder ----------------------------------------------------------------
    u0 = DecoderSubpixelBlock(filter_num//2,
             upscale_factor=2,
             batch_norm=False,
             name='decoder_blk_1_' + task)(d4)
    u1 = DecoderSubpixelBlock(filter_num//4, 
             upscale_factor=2,
             batch_norm=False,
             name='decoder_blk_2_' + task)(nn.concatenate([u0,d3]))
    u2 = DecoderSubpixelBlock(filter_num//8, 
             upscale_factor=2,
             batch_norm=False,
             name='decoder_blk_3_' + task)(nn.concatenate([u1,d2]))
    u3 = DecoderSubpixelBlock(filter_num//8,
             upscale_factor=2,
             batch_norm=False,
             name='decoder_blk_4_' + task)(nn.concatenate([u2,d1]))

    # Task-dependent prediction head
    y = nn.spatial_dropout(u3, 0.25)
    out_filters = 55 if task == 'landmarks' else 1
    y = nn.conv1x1(y,
            filters=out_filters,
            activation='linear',
            batch_norm=False,
            name='projection_' + task)
    y = nn.leaky_relu(y, alpha=0.1, name=task)
    return nn.model(inputs=inp, outputs=y)