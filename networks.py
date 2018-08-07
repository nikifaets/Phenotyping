from keras.models import Model
from normalize import normalize
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv3D, MaxPooling3D, Conv2D, MaxPooling2D, Input, UpSampling2D, Activation, Conv2DTranspose
import numpy as np 
from keras import regularizers

stride = (2,2)

def ae1(inputs):
	#encoder
	encoded = Conv2D(2, (2,2), activation='relu',padding='same')(inputs)
	#encoded = Conv2D(16, (2,2), activation='relu',padding='same' )(encoded)
	encoded = Conv2D(16, (2,2), activation='relu',padding='same' )(encoded)
	#encoded = Conv2D(16, (2,2dd), activation='relu',padding='same' )(encoded)
	encoded = Conv2D(16, (2,2), activation='relu', strides=(2,2))(encoded)
	encoded = Conv2D(32, (2,2), activation='relu', padding='same' )(encoded)
	#encoded = Conv2D(64, (2,2), activation='relu', padding='same' )(encoded)
	encoded = Conv2D(64, (2,2), activation='relu', strides=(2,2))(encoded)	
	encoded = Conv2D(64, (2,2), activation='relu', padding='same' )(encoded)
	encoded = Conv2D(64, (2,2), activation='relu', strides=(2,2) )(encoded)
	
	#encoded = MaxPooling2D((2,2))(encoded)
	encoded = Conv2D(64, (2,2), activation='relu', padding='same' )(encoded)
	encoded = Conv2D(256, (2,2), activation='relu', strides=(4,4))(encoded)
	encoded = Conv2D(256, (2,2), activation='relu', padding='same', name='latent_space')(encoded)
	#encoded = Conv2D(128, (2,2), activation='relu', strides=(2,2))(encoded)
	#encoded = Conv2D(256, (2,2), activation='relu', padding='same')(encoded)
	#encoded = Conv2D(256, (2,2), activation='relu', strides=(2,2))(encoded)

	#decoder
	#decoded = Conv2D(256, (2,2), activation='relu', padding='same')(encoded)
	#decoded = Conv2DTranspose(256, (2,2), activation='relu',  strides=(2,2))(decoded)
	decoded = Conv2DTranspose(256, (4,4), activation='relu')(encoded)
	decoded = Conv2DTranspose(256, (2,2), activation='relu', strides=(2,2))(decoded)
	decoded = Conv2DTranspose(128, (2,2), activation='relu', strides=(2,2))(decoded)
	#decoded = Conv2D(32, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2DTranspose(64, (2,2), activation='relu', strides=(2,2))(decoded)
	decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	#decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	#decoded = UpSampling2D( (2,2))(decoded)
	#decoded = Conv2D(32, (2,2), activation='relu',padding='same' )(decoded)
	decoded = Conv2D(16, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2D(2, (2,2), activation='tanh', padding='same')(decoded)

	return decoded

def currBest(inputs):
	encoded = Conv2D(2, (2,2), activation='relu',padding='same')(inputs)
	encoded = Conv2D(16, (2,2), activation='relu',padding='same' )(encoded)
	encoded = Conv2D(16, (2,2), activation='relu', strides=(2,2))(encoded)
	encoded = Conv2D(32, (2,2), activation='relu', padding='same' )(encoded)
	encoded = Conv2D(64, (2,2), activation='relu', strides=(2,2))(encoded)	
	encoded = Conv2D(64, (2,2), activation='relu', padding='same' )(encoded)
	encoded = Conv2D(64, (2,2), activation='relu', strides=(2,2) )(encoded)
	
	encoded = Conv2D(64, (2,2), activation='relu', padding='same' )(encoded)
	encoded = Conv2D(256, (2,2), activation='relu', strides=(4,4))(encoded)
	encoded = Conv2D(256, (2,2), activation='relu', padding='same', name='latent_space')(encoded)

	#decoder

	decoded = Conv2DTranspose(256, (4,4), activation='relu')(encoded)
	decoded = Conv2D(128, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2DTranspose(128, (2,2), activation='relu', strides=(2,2))(decoded)
	decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2DTranspose(64, (2,2), activation='relu', strides=(2,2))(decoded)
	decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2DTranspose(64, (2,2), activation='relu', strides=(2,2))(decoded)
	decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2D(32, (2,2), activation='relu',padding='same' )(decoded)
	decoded = Conv2D(16, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2D(2, (2,2), activation='tanh', padding='same')(decoded)

	return decoded


def batViko(inputs):
	encoded = Conv2D(2, (2,2), activation='relu', padding='same')(inputs)
	encoded = Conv2D(16, (2,2), activation='relu', padding='same')(encoded)
	encoded = Conv2D(16, (2,2), activation='relu',  strides=(2,2))(encoded)
	encoded = Conv2D(32, (2,2), activation='relu',  padding='same')(encoded)
	encoded = Conv2D(64, (2,2), activation='relu',  strides=(2,2))(encoded)	
	encoded = Conv2D(64, (2,2), activation='relu',  padding='same')(encoded)
	encoded = Conv2D(64, (2,2), activation='relu',  strides=(2,2),)(encoded)
	
	encoded = Conv2D(64, (2,2), activation='relu', padding='same' )(encoded)
	encoded = Conv2D(64, (2,2), activation='relu', strides=(4,4))(encoded)
	encoded = Conv2D(4, (2,2), activation='relu', padding='same', name='latent_space')(encoded)

	#decoder

	decoded = Conv2DTranspose(256, (4,4), activation='relu')(encoded)
	decoded = Conv2D(128, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2DTranspose(128, (2,2), activation='relu', strides=(2,2))(decoded)
	decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2DTranspose(64, (2,2), activation='relu', strides=(2,2))(decoded)
	decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2DTranspose(64, (2,2), activation='relu', strides=(2,2))(decoded)
	decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2D(32, (2,2), activation='relu',padding='same' )(decoded)
	decoded = Conv2D(16, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2D(2, (2,2), activation='tanh', padding='same')(decoded)

	return decoded


def batVikoPadding(inputs):
	encoded = Conv2D(2, (2,2), activation='relu', padding='valid')(inputs)
	encoded = Conv2D(16, (2,2), activation='relu', padding='valid')(encoded)
	encoded = Conv2D(16, (2,2), activation='relu',  strides=(2,2))(encoded)
	encoded = Conv2D(32, (2,2), activation='relu',  padding='valid')(encoded)
	encoded = Conv2D(64, (2,2), activation='relu',  strides=(2,2))(encoded)	
	encoded = Conv2D(128, (2,2), activation='relu',  padding='valid')(encoded)
	encoded = Conv2D(128, (2,2), activation='relu',  strides=(2,2))(encoded)
	encoded = Conv2D(256, (2,2), activation='relu',  padding='valid')(encoded)
	
	#encoded = Conv2D(64, (2,2), activation='relu', padding='valid')(encoded)
	#encoded = Conv2D(256, (2,2), activation='relu', strides=(2,2))(encoded)
	encoded = Conv2D(256, (2,2), activation='relu', padding='valid', name='latent_space')(encoded)

	#decoderd

	decoded = Conv2DTranspose(256, (4,4), activation='relu')(encoded)
	decoded = Conv2D(128, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2DTranspose(128, (2,2), activation='relu', strides=(2,2))(decoded)
	decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2DTranspose(64, (2,2), activation='relu', strides=(2,2))(decoded)
	decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2DTranspose(64, (2,2), activation='relu', strides=(2,2))(decoded)
	#decoded = Conv2DTranspose(64, (2,2), activation='relu', strides=(2,2))(decoded)
	'''decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2D(32, (2,2), activation='relu',padding='same' )(decoded)
	decoded = Conv2D(16, (2,2), activation='relu', padding='same')(decoded)'''
	decoded = Conv2D(2, (2,2), activation='tanh', padding='same')(decoded)

	return decoded

def batVikoMix(inputs):
	encoded = Conv2D(2, (2,2), activation='relu', padding='same')(inputs)
	encoded = Conv2D(16, (4,4), activation='relu', padding='same')(encoded)
	encoded = MaxPooling2D((2,2))(encoded)
	encoded = Conv2D(32, (2,2), activation='relu',  padding='same')(encoded)
	encoded = MaxPooling2D((2,2))(encoded)
	encoded = Conv2D(64, (2,2), activation='relu',  padding='same')(encoded)
	encoded = MaxPooling2D((2,2))(encoded)
	
	encoded = Conv2D(64, (2,2), activation='relu', padding='same' )(encoded)
	encoded = MaxPooling2D((2,2))(encoded)
	encoded = MaxPooling2D((2,2))(encoded)
	encoded = Conv2D(128, (2,2), activation='relu', padding='same', name='latent_space')(encoded)

		#decoder

	decoded = Conv2DTranspose(256, (4,4), activation='relu')(encoded)
	decoded = Conv2D(128, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2DTranspose(128, (2,2), activation='relu', strides=(2,2))(decoded)
	decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2DTranspose(64, (2,2), activation='relu', strides=(2,2))(decoded)
	decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2DTranspose(64, (2,2), activation='relu', strides=(2,2))(decoded)
	decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2D(32, (2,2), activation='relu',padding='same' )(decoded)
	decoded = Conv2D(16, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2D(2, (2,2), activation='tanh', padding='same')(decoded)

	return decoded

def batVikoVanilla(inputs):
	encoded = Conv2D(2, (2,2), activation='relu', padding='same')(inputs)
	encoded = MaxPooling2D((2,2))(encoded)
	encoded = Conv2D(32, (2,2), activation='relu',  padding='same')(encoded)
	encoded = MaxPooling2D((2,2))(encoded)
	encoded = Conv2D(64, (2,2), activation='relu',  padding='same')(encoded)
	encoded = MaxPooling2D((2,2))(encoded)
				
	encoded = Conv2D(64, (2,2), activation='relu', padding='same' )(encoded)
	encoded = MaxPooling2D((2,2))(encoded)
	encoded = Conv2D(64, (2,2), activation='relu', padding='same', name='latent_space')(encoded)

				#decoder

	decoded = Conv2D(128, (2,2), activation='relu', padding='same')(encoded)
	decoded = UpSampling2D((2,2))(decoded)
	decoded = Conv2D(128, (2,2), activation='relu', padding='same')(decoded)
	decoded = UpSampling2D((2,2))(decoded)
	decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	decoded = UpSampling2D((2,2))(decoded)	
	decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	decoded = UpSampling2D((2,2))(decoded)
	decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	decoded = UpSampling2D((2,2))(decoded)
	decoded = Conv2D(16, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2D(2, (2,2), activation='tanh', padding='same')(decoded)

	return decoded

def cellChannel(inputs):

	encoded = Conv2D(1, (2,2), activation='relu', padding='same')(inputs)
	encoded = Conv2D(4,(2,2), activation='relu', padding='same')(encoded)
	encoded = Conv2D(16, (2,2), activation='relu', padding='valid')(encoded)
	encoded = Conv2D(16, (2,2), activation='relu',  strides=(2,2))(encoded)
	encoded = Conv2D(32, (2,2), activation='relu',  padding='valid')(encoded)
	encoded = Conv2D(64, (2,2), activation='relu',  strides=(2,2))(encoded)	
	encoded = Conv2D(64, (2,2), activation='relu',  padding='valid')(encoded)
	encoded = Conv2D(128, (2,2), activation='relu',  strides=(2,2))(encoded)
	encoded = Conv2D(128, (2,2), activation='relu',  padding='valid')(encoded)
	encoded = Conv2D(128, (2,2), activation='relu',  strides=(2,2))(encoded)
	#encoded = Conv2D(128, (2,2), activation='relu',  padding='same')(encoded)
	
	
	#encoded = Conv2D(64, (2,2), activation='relu', padding='valid')(encoded)
	#encoded = Conv2D(256, (2,2), activation='relu', strides=(2,2))(encoded)
	encoded = Conv2D(16, (2,2), activation='relu', padding='same', name='latent_space')(encoded)

	#decoderd


	#decoded = Conv2D(256, (2,2), activation='relu', padding='same')(encoded)
	decoded = Conv2DTranspose(256, (2,2), activation='relu', strides=(2,2))(encoded)
	decoded = Conv2D(128, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2DTranspose(128, (2,2), activation='relu', strides=(2,2))(decoded)
	decoded = Conv2D(128, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2DTranspose(128, (2,2), activation='relu', strides=(2,2))(decoded)
	decoded = Conv2D(128, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2DTranspose(64, (2,2), activation='relu', strides=(2,2))(decoded)
	decoded = Conv2DTranspose(64, (2,2), activation='relu', strides=(2,2))(decoded)
	#decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2D(32, (2,2), activation='relu',padding='same' )(decoded)
	decoded = Conv2D(16, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2D(1, (2,2), activation='tanh', padding='same')(decoded)

	return decoded

def cellChannel1(inputs):
	encoded = Conv2D(1, (2,2), activation='relu',padding='same')(inputs)
	encoded = Conv2D(16, (2,2), activation='relu',padding='same' )(encoded)
	encoded = Conv2D(16, (2,2), activation='relu', strides=(2,2))(encoded)
	encoded = Conv2D(32, (2,2), activation='relu', padding='same' )(encoded)
	encoded = Conv2D(64, (2,2), activation='relu', strides=(2,2))(encoded)	
	encoded = Conv2D(64, (2,2), activation='relu', padding='same' )(encoded)
	encoded = Conv2D(64, (2,2), activation='relu', strides=(2,2) )(encoded)
	
	encoded = Conv2D(64, (2,2), activation='relu', padding='same' )(encoded)
	encoded = Conv2D(256, (2,2), activation='relu', strides=(4,4))(encoded)
	encoded = Conv2D(1, (2,2), activation='relu', padding='same', name='latent_space')(encoded)

	#decoder

	decoded = Conv2DTranspose(256, (4,4), activation='relu')(encoded)
	decoded = Conv2D(128, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2DTranspose(128, (2,2), activation='relu', strides=(2,2))(decoded)
	decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2DTranspose(64, (2,2), activation='relu', strides=(2,2))(decoded)
	decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2DTranspose(64, (2,2), activation='relu', strides=(2,2))(decoded)
	decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2D(32, (2,2), activation='relu',padding='same' )(decoded)
	decoded = Conv2D(16, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2D(1, (2,2), activation='tanh', padding='same')(decoded)

	return decoded


def basicDenoiser(input_img):
    #encoder
	
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	conv2a = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
	pool2a = MaxPooling2D(pool_size=(2, 2))(conv2a)
	conv2b = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2a)
	pool2b = MaxPooling2D(pool_size=(2, 2))(conv2b)
	conv2c = Conv2D(16, (3, 3), activation='relu', padding='same')(pool2b)
	pool2c = MaxPooling2D(pool_size=(2, 2))(conv2c)
	conv3 = Conv2D(2, (3, 3), activation='relu', padding='same', name='latent_space')(pool2c)

    #decoder
    
	conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
	up1 = UpSampling2D((2,2))(conv4)
	conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
	up2 = UpSampling2D((2,2))(conv5)
	conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
	up2 = UpSampling2D((2,2))(conv6)
	conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
	up1a = UpSampling2D((2,2))(conv6)
	conv5a = Conv2D(32, (3, 3), activation='relu', padding='same')(up1a)
	up2 = UpSampling2D((2,2))(conv5a)
	decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)
	return decoded