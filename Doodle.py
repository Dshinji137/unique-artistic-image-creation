
# coding: utf-8

# In[47]:

import theano
import theano.tensor as T 
from theano.tensor.nnet import relu,conv2d
import h5py
from theano.tensor.signal.pool import pool_2d
import numpy as np
import scipy.optimize
from scipy.optimize import minimize
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[2]:

class ConvLayer(object):
    """Convolutional layer"""

    def __init__(self, rng, input, filter_shape, image_shape, W=None, b=None, padding=(1,1),filter_flip=True):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type W: in accordance to filter_shape
        :param: use pretrained VGG weights

        :type b: (filter_shape[0],)
        :param: use pretrained vgg weights

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type padding: tuple or list of length 2
        :param padding: padding for conv
        """

        #assert image_shape[1] == filter_shape[1]
        self.input = input

        if W is None:
            # there are "num input feature maps * filter height * filter width"
            # inputs to each hidden unit
            fan_in = numpy.prod(filter_shape[1:])
            # each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /
            #  pooling size
            fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
            # initialize weights with random weights
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX), borrow=True)
        else:
            self.W = theano.shared(W,borrow=True)

        if b is None:
            pass
        else:
            self.b = theano.shared(b,borrow=True)
        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode=padding,
            filter_flip=filter_flip
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        if b is None:
            self.output = relu(conv_out)
        else:
            self.output = relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))


# In[36]:

# Transfer image to 4-D tensor
MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))
def convertInput(im):
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    im = im[::-1, :, :]
    im = im - MEAN_VALUES
    return np.float32(im[np.newaxis])

def convertMap(im):
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    im = im[::-1, :, :]
    return np.float32(im[np.newaxis])


# In[37]:

# Load and transfer images
style_image = plt.imread('Renoir.jpg')
style_image = convertInput(style_image)
style_map = plt.imread('Renoir_sem.png')
style_map = convertMap(style_map)

content_image = plt.imread('Renoir.jpg')
shape = content_image.shape
content_image = convertInput(content_image)
content_map = plt.imread('Landscape_sem.png')
content_map = convertMap(content_map)

noise = np.random.uniform(-127,128,size=(1,3,style_image.shape[2],style_image.shape[3]))
output_image = theano.shared(np.asarray(noise,dtype='float32'),borrow=True)


# In[38]:

# parameters to adjust
style_weight = 1
content_weight = 1
semantic_weight = 100
regular_weight = 0


# In[39]:

x = T.tensor4('image')
mapImage = T.tensor4('map')
mode='average_exc_pad'

# load the trained VGG
vgg_weights = h5py.File("vgg19_weights.h5")

# build the model
rng = np.random.RandomState(23455)
g = vgg_weights['layer_1']
conv1_1 = ConvLayer(
    rng=rng, input=x, filter_shape=(64,3,3,3), image_shape=None, 
    W=g['param_0'].value ,b=g['param_1'].value, padding=(1,1)
)
g = vgg_weights['layer_3']
conv1_2 = ConvLayer(rng=rng, input=conv1_1.output,filter_shape=(64,64,3,3), image_shape=None,
    W=g['param_0'].value,b=g['param_1'].value, padding=(1,1))
conv1_pool = pool_2d(conv1_2.output,ds=(2,2),padding=(0,0),mode=mode,ignore_border=True)

g = vgg_weights['layer_6']
conv2_1 = ConvLayer(rng=rng, input=conv1_pool, filter_shape=(128,64,3,3), image_shape=None, 
    W=g['param_0'].value ,b=g['param_1'].value, padding=(1,1))
g = vgg_weights['layer_8']
conv2_2 = ConvLayer(rng=rng, input=conv2_1.output, filter_shape=(128,128,3,3), image_shape=None, 
    W=g['param_0'].value ,b=g['param_1'].value, padding=(1,1))
conv2_pool = pool_2d(conv2_2.output,ds=(2,2),padding=(0,0),mode=mode,ignore_border=True)

g = vgg_weights['layer_11']
conv3_1 = ConvLayer(rng=rng, input=conv2_pool, filter_shape=(256,128,3,3), image_shape=None, 
    W=g['param_0'].value ,b=g['param_1'].value, padding=(1,1))
g = vgg_weights['layer_13']
conv3_2 = ConvLayer(rng=rng, input=conv3_1.output, filter_shape=(256,256,3,3), image_shape=None, 
    W=g['param_0'].value ,b=g['param_1'].value, padding=(1,1))
g = vgg_weights['layer_15']
conv3_3 = ConvLayer(rng=rng, input=conv3_2.output, filter_shape=(256,256,3,3), image_shape=None, 
    W=g['param_0'].value ,b=g['param_1'].value, padding=(1,1))
g = vgg_weights['layer_17']
conv3_4 = ConvLayer(rng=rng, input=conv3_3.output, filter_shape=(256,256,3,3), image_shape=None, 
    W=g['param_0'].value ,b=g['param_1'].value, padding=(1,1))
conv3_pool = pool_2d(conv3_4.output,ds=(2,2),padding=(0,0),mode=mode,ignore_border=True)

# map image pass through pool layer
mapImage3 = pool_2d(mapImage,ds=(4,4),mode=mode,ignore_border=True)
g = vgg_weights['layer_20']
conv4_1 = ConvLayer(rng=rng, input=conv3_pool, filter_shape=(512,256,3,3), image_shape=None, 
    W=g['param_0'].value ,b=g['param_1'].value, padding=(1,1))
g = vgg_weights['layer_22']
conv4_2 = ConvLayer(rng=rng, input=conv4_1.output, filter_shape=(512,512,3,3), image_shape=None, 
    W=g['param_0'].value ,b=g['param_1'].value, padding=(1,1))
mapImage4 = pool_2d(mapImage,ds=(8,8),mode=mode,ignore_border=True)
# semantic layers
sem3_1 = T.concatenate([conv3_1.output,mapImage3],axis=1)
sem4_1 = T.concatenate([conv4_1.output,mapImage4],axis=1)


# In[40]:

# special layers
content_layers = [conv4_2]
style_layers = [conv3_1,conv4_1]
concat_layers = [sem3_1,sem4_1]


# In[41]:

# Normalization
def compute_norms(patches):
    ni = T.sqrt(T.sum(patches[:,:-3] ** 2.0, axis=(1,), keepdims=True))
    ns = T.sqrt(T.sum(patches[:,-3:] ** 2.0, axis=(1,), keepdims=True))
    return [ni] + [ns] 
def normalize_components(patches, norms):
    # normlize the concat layer, conv and semantic layers separately
    return T.concatenate([patches[:,:-3] / (norms[0]),
                          semantic_weight * patches[:,-3:] / (norms[1])],
                         axis=1)
def extract_patch(im):
    output = T.nnet.neighbours.images2neibs(im,(3,3),(1,1),mode='valid')
    output = output.reshape((-1,output.shape[0]//im.shape[1],3,3)).dimshuffle((1, 0, 2, 3))
    return output


# In[42]:

# style_loss
nn_outputs = []
# define a theano function to concact conv layer and semantic layer
concat = theano.function(
    inputs=[x,mapImage],
    outputs=[concat_layer for concat_layer in concat_layers]
)
# concatenate style image to style map
style_concats = concat(style_image,style_map)
# make it shared variables
style_concats = [theano.shared(element) for element in style_concats]
# extract concatenated image
style_patches = [extract_patch(element.get_value()) for element in style_concats]

# concatenate output image to content map
output_concats = concat(output_image.get_value(),content_map)
# extract concatenated image
output_patches = [extract_patch(element) for element in output_concats]

for output_patch, style_patch in zip(output_patches,style_patches):
    # normalization
    style_patch_norm = normalize_components(style_patch,compute_norms(style_patch))
    output_patch_norm = normalize_components(output_patch,compute_norms(output_patch))
    # flatten
    v1 = output_patch_norm.flatten(2)
    v2 = style_patch_norm.flatten(2)
    # for each patch in output image and content map combined, find the most suitable
    # patch in style image and style map combined,(with the maximal Fai function), 
    # record in nn_outputs
    index = T.argmax(T.dot(v1,v2.T),axis=1)
    nn_outputs.append(style_patch[index])

# style loss: the mean of difference between most correlated style patches and style layer
convlayer_patches = [extract_patch(element.output) for element in style_layers]
style_losses = [style_weight * T.mean((convlayer_patch - nn_output[:,:-3]) ** 2.0) 
                for convlayer_patch,nn_output in zip(convlayer_patches,nn_outputs)]


# In[43]:

# content loss
content_losses = []
if content_layers!=[]:
    f = theano.function(
        inputs=[x],
        outputs=[content_layers[0].output]
    )
    content_losses = [content_weight * T.mean((f(content_image)[0] - f(output_image.get_value())[0]) ** 2)]
# regularization
im = output_image.get_value()
regularization = (((im[:,:,:-1,:-1] - im[:,:,1:,:-1])**2 + (im[:,:,:-1,:-1] - im[:,:,:-1,1:])**2)**1.25).sum()


# In[44]:

# prepare to bfgs
cost = T.sum(content_losses + style_losses + [regular_weight * regularization])
grads = T.grad(cost, x)

# theano functions to generate total loss and grads
f_loss = theano.function([x], cost, allow_input_downcast=True)
f_grad = theano.function([x], grads, allow_input_downcast=True)

def eval_loss(x0):
    x0 = np.float32(x0.reshape((1, 3,style_image.shape[2],style_image.shape[3])))
    output_image.set_value(x0)
    return f_loss(output_image.get_value()).astype('float64')
def eval_grad(x0):
    x0 = np.float32(x0.reshape((1, 3, style_image.shape[2],style_image.shape[3])))
    output_image.set_value(x0)
    return np.array(f_grad(output_image.get_value())).flatten().astype('float64')


# In[45]:

# recover data to output as an image
def finalize_image(x):
    x = np.copy(x[0])
    x = x + MEAN_VALUES
    x = x[::-1]
    x = np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# In[50]:

# initialize synthesized image
noise = np.random.uniform(-127,128,size=(1,3,style_image.shape[2],style_image.shape[3]))
output_image = theano.shared(np.asarray(noise,dtype='float32'),borrow=True)
x0 = output_image.get_value().astype('float64')
result = []
result.append(x0)
# Doodle
data_bounds = np.zeros((np.product(x0.shape), 2), dtype=np.float64)
data_bounds[:] = (-128,128)
for i in range(3):
    scipy.optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, maxfun=40,
                                m=5,bounds=data_bounds,factr=0.0, pgtol=0.0)
    x0 = output_image.get_value().astype('float64')
    result.append(x0)


# In[34]:

# draw synthesized image
plt.figure(figsize=(12,12))
mimus = np.asarray([0,1,1]).reshape((3,1,1))
for i in range(len(result)):
    plt.subplot(3, 3, i+1)
    plt.gca().xaxis.set_visible(False)    
    plt.gca().yaxis.set_visible(False)
    im = finalize_image(result[i])
    plt.imshow(im)
plt.tight_layout()


# In[ ]:



