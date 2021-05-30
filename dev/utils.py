
import numpy as np
import jax.numpy as jnp
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.python.client import device_lib
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    return True

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    gag = [x.name for x in local_device_protos if x.device_type == 'GPU']
    print(gag)

    if tf.test.gpu_device_name(): 
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

def make_gen(x):
    def gen():
      i = 0
      while(i<len(x)):
        yield next(x)
        i +=1
    return gen

def add_noise(img,sigma=0.01):
    '''Add random noise to an image'''
    deviation = sigma*np.random.uniform(0,1)
    noise = np.random.normal(0, sigma, img.shape)
    img += noise
    np.clip(img, 0., 1.)
    return img

# def smear(data,axis,ds):
#     n_time = data.shape[axis]
#     nt = int(n_time//ds+1)
# #     sdata = np.array(np.split(data, nt, axis=axis))
#     sdata = np.array(np.array_split(data, nt, axis=axis))
#     return np.mean(sdata,axis=axis+1)

def smear(data,axis,ds):
    n_time = data.shape[axis]
    nt = int(n_time//ds+1)
#     sdata = np.array(np.split(data, nt, axis=axis))
    sdata = np.array_split(data, nt, axis=axis)
    sdata = [np.mean(sd,axis=axis) for sd in sdata]    
    sdata = np.array(sdata)
#     print(sdata.shape)
    return sdata

def jsmear(data,axis,ds):
    n_time = data.shape[axis]
    nt = int(n_time//ds)
    sdata = jnp.array(jnp.split(data, nt, axis=axis))
    return jnp.mean(sdata,axis=axis+1)

def get_patch(shape,axis,sigma0=0.7,muu=0,c=1):
#     nx,ny,nz = shape
    shape = list(shape)
    nr = shape[axis]
    shape.remove(nr)
    ny,nx = shape
    x, y = np.meshgrid(np.linspace(-1,1,nx), np.linspace(-1,1,ny))
    dst = np.sqrt(x*x+y*y)
    # Intializing sigma and muu
    gpatch = []
    for z in np.linspace(-.5,.5,nr):
        sigma = sigma0-np.abs(z)
        gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
        gpatch.append(gauss)

    gpatch = np.stack(gpatch,axis=axis)
    gpatch = gpatch-gpatch.min()
    gpatch = gpatch/gpatch.max()
    return c*gpatch

# def jget_patch(shape,axis,sigma0=0.7,muu=0,c=1):
# #     nx,ny,nz = shape
#     shape = list(shape)
#     nr = shape[axis]
#     shape.remove(nr)
#     ny,nx = shape
#     x, y = jnp.meshgrid(jnp.linspace(-1,1,nx), jnp.linspace(-1,1,ny))
#     dst = jnp.sqrt(x*x+y*y)
#     # Intializing sigma and muu
#     gpatch = []
#     for z in jnp.linspace(-.5,.5,nr):
#         sigma = sigma0-jnp.abs(z)
#         gauss = jnp.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
#         gpatch.append(gauss)

#     gpatch = jnp.stack(gpatch,axis=axis)
#     gpatch = gpatch-gpatch.min()
#     gpatch = gpatch/gpatch.max()
#     return c*gpatch


def get_slice(data,label,nx,ny,nz,data2=None):
    lx,ly,lz = data.shape  
    if nx==0 or nx==lx:
        slx = slice(0, lx)                
    else:
        idx = np.random.randint(0, lx - nx)            
        slx = slice(idx, (idx+nx))       
    if ny==0 or ny==ly:
        sly = slice(0, ly)                
    else:
        idy = np.random.randint(0, ly - ny)            
        sly = slice(idy, (idy+ny))
    if nz==0 or nz==ly:
        slz = slice(0, lz)                
    else:
        idz = np.random.randint(0, lz - nz)            
        slz = slice(idz, (idz+nz))
    if data2 is None:
        return data[slx, sly, slz],label[slx, sly, slz]
    else:
        return data[slx, sly, slz],data2[slx, sly, :],label[slx, sly, slz]

# def get_slice(data,label,nx,ny,nz):
#     lx,ly,lz = data.shape  
#     if nx==0 or nx==lx:
#         slx = slice(0, lx)                
#     else:
#         idx = np.random.randint(0, lx - nx)            
#         slx = slice(idx, (idx+nx))       
#     if ny==0 or ny==ly:
#         sly = slice(0, ly)                
#     else:
#         idy = np.random.randint(0, ly - ny)            
#         sly = slice(idy, (idy+ny))
#     if nz==0 or nz==ly:
#         slz = slice(0, lz)                
#     else:
#         idz = np.random.randint(0, lz - nz)            
#         slz = slice(idz, (idz+nz))
#     return data[slx, sly, slz],label[slx, sly, slz]

# def jget_slice(data,label,nx,ny,nz):
#     lx,ly,lz = data.shape  
#     if nx==0 or nx==lx:
#         slx = slice(0, lx)                
#     else:
#         idx = jnp.random.randint(0, lx - nx)            
#         slx = slice(idx, (idx+nx))       
#     if ny==0 or ny==ly:
#         sly = slice(0, ly)                
#     else:
#         idy = jnp.random.randint(0, ly - ny)            
#         sly = slice(idy, (idy+ny))
#     if nz==0 or nz==ly:
#         slz = slice(0, lz)                
#     else:
#         idz = jnp.random.randint(0, lz - nz)            
#         slz = slice(idz, (idz+nz))
#     return data[slx, sly, slz],label[slx, sly, slz]

def SimpleConv(shape=(28, 28, 1),n_class=1,fgrow=1,bnorm=False,lactivation='relu'):
    tf.keras.backend.clear_session()
    nd1,nd2,nch = shape
    input_img = keras.Input(shape=shape)

    nch *= fgrow
    x1 = layers.Conv2D(nch, (3, 3), activation='relu', padding='same')(input_img)
    if bnorm:
        x1 = layers.BatchNormalization()(x1)
    x2 = layers.Conv2D(nch, (3, 3), activation='relu', padding='same')(x1)
    if bnorm:
        x2 = layers.BatchNormalization()(x2)    
    nch *= fgrow
    x3 = layers.Conv2D(nch, (3, 3), activation='relu', padding='same')(x2)
    if bnorm:
        x3 = layers.BatchNormalization()(x3)
    nch *= fgrow
    x4 = layers.Conv2D(nch, (3, 3), activation='relu', padding='same')(x3)
    if bnorm:
        x4 = layers.BatchNormalization()(x4)
    nch /= fgrow
    x5 = layers.Conv2D(nch, (3, 3), activation='relu', padding='same')(x4)
    if bnorm:
        x5 = layers.BatchNormalization()(x5)
    x5 = x3+x5
    nch /= fgrow
    x6 = layers.Conv2D(nch, (3, 3), activation='relu', padding='same')(x5)
    if bnorm:
        x6 = layers.BatchNormalization()(x6)
    nch /= fgrow
    decoded = layers.Conv2D(nch, (3, 3), activation=lactivation, padding='same')(x6)
#     decoded = layers.Conv2D(n_class, (3, 3), activation='tanh', padding='same')(x6)
    model = keras.Model(input_img, decoded)
    return model

def SimpleConv_count(shape1=(28, 28, 1),shape2=(28, 28, 1),n_class=1,fgrow=1,bnorm=False,lactivation='relu'):
    tf.keras.backend.clear_session()
    nd1,nd2,nch = shape1
    input_img1 = keras.Input(shape=shape1)
    nch *= fgrow
    x1 = layers.Conv2D(nch, (3, 3), activation='relu', padding='same')(input_img1)
    if bnorm:
        x1 = layers.BatchNormalization()(x1)
    x2 = layers.Conv2D(nch, (3, 3), activation='relu', padding='same')(x1)
    if bnorm:
        x2 = layers.BatchNormalization()(x2)
    nch *= fgrow
    x3 = layers.Conv2D(nch, (3, 3), activation='relu', padding='same')(x2)
    if bnorm:
        x3 = layers.BatchNormalization()(x3)
    nch *= fgrow
    x4 = layers.Conv2D(nch, (3, 3), activation='relu', padding='same')(x3)
    if bnorm:
        x4 = layers.BatchNormalization()(x4)
    nch /= fgrow
    x5 = layers.Conv2D(nch, (3, 3), activation='relu', padding='same')(x4)
    if bnorm:
        x5 = layers.BatchNormalization()(x5)
    x5 = x3+x5
    nch /= fgrow
    x6p = layers.Conv2D(nch, (3, 3), activation='relu', padding='same')(x5)
    if bnorm:
        x6p = layers.BatchNormalization()(x6p)
    
    nd1,nd2,nch = shape1
    input_img2 = keras.Input(shape=shape2)
    nch *= fgrow
    x1 = layers.Conv2D(nch, (3, 3), activation='relu', padding='same')(input_img2)
    if bnorm:
        x1 = layers.BatchNormalization()(x1)
    x2 = layers.Conv2D(nch, (3, 3), activation='relu', padding='same')(x1)
    if bnorm:
        x2 = layers.BatchNormalization()(x2)
    nch *= fgrow
    x3 = layers.Conv2D(nch, (3, 3), activation='relu', padding='same')(x2)
    if bnorm:
        x3 = layers.BatchNormalization()(x3)
    nch *= fgrow
    x4 = layers.Conv2D(nch, (3, 3), activation='relu', padding='same')(x3)
    if bnorm:
        x4 = layers.BatchNormalization()(x4)
    nch /= fgrow
    x5 = layers.Conv2D(nch, (3, 3), activation='relu', padding='same')(x4)
    if bnorm:
        x5 = layers.BatchNormalization()(x5)
    x5 = x3+x5
    nch /= fgrow
    x6c = layers.Conv2D(nch, (3, 3), activation='relu', padding='same')(x5)
    if bnorm:
        x6c = layers.BatchNormalization()(x6c)
    
    x6 = x6p+x6c
    
    decoded = layers.Conv2D(n_class, (3, 3), activation=lactivation, padding='same')(x6)

    model = keras.Model([input_img1,input_img2], decoded)
    return model
