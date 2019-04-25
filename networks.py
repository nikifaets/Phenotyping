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
	encoded = Conv2D(64, (2,2), activation='relu', strides = (2,2))(encoded)
	
	encoded = Conv2D(256, (2,2), activation='relu', padding='same' )(encoded)
	encoded = Conv2D(1, (2,2), activation='relu', padding='same', name='latent_space')(encoded)

	#decoder

	decoded = Conv2DTranspose(256, (2,2), activation='relu', strides=(2,2))(encoded)
	decoded = Conv2D(128, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2DTranspose(128, (2,2), activation='relu', strides=(2,2))(decoded)
	decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2DTranspose(64, (2,2), activation='relu', strides=(2,2))(decoded)
	decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2D(32, (2,2), activation='relu',padding='same' )(decoded)
	decoded = Conv2D(16, (2,2), activation='relu', padding='same')(decoded)
	decoded = Conv2D(1, (2,2), activation='sigmoid', padding='same')(decoded)

	return decoded


def basicDenoiser(input_img):
    #encoder
	
	encoded = Conv2D(8, (2, 2), activation='relu', padding='same')(input_img)
	encoded = Conv2D(16, (2, 2), activation='relu', padding='same')(encoded)
	#encoded = MaxPooling2D(pool_size=(2, 2))(encoded)
	encoded = Conv2D(32, (2, 2), activation='relu', padding='same')(encoded)
	encoded = MaxPooling2D(pool_size=(2, 2))(encoded)
	encoded = Conv2D(64, (2, 2), activation='relu', padding='same')(encoded)
	encoded = MaxPooling2D(pool_size=(2, 2))(encoded)
	encoded = Conv2D(64, (2, 2), activation='relu', padding='same')(encoded)
	encoded = MaxPooling2D(pool_size=(2, 2))(encoded)
	encoded = Conv2D(32, (2, 2), activation='relu', padding='same')(encoded)
	encoded = Conv2D(16, (2, 2), activation='relu', padding='same')(encoded)
	encoded = Conv2D(1, (2, 2), activation='relu', padding='same', name='latent_space')(encoded)

    #decoder
	decoded = Conv2D(16, (2, 2), activation='relu', padding='same')(encoded)
	decoded = Conv2D(32, (2, 2), activation='relu', padding='same')(decoded)
	decoded = UpSampling2D((2,2))(decoded)
	#decoded = UpSampling2D((2,2))(decoded)
	decoded = Conv2D(64, (2, 2), activation='relu', padding='same')(decoded)
	decoded = UpSampling2D((2,2))(decoded)
	decoded = Conv2D(64, (2, 2), activation='relu', padding='same')(decoded)
	decoded = UpSampling2D((2,2))(decoded)
	decoded = Conv2D(32, (2, 2), activation='relu', padding='same')(decoded)
	#decoded = UpSampling2D((2,2))(decoded)
	decoded = Conv2D(16, (2, 2), activation='relu', padding='same')(decoded)
	#decoded = UpSampling2D((2,2))(decoded)
	decoded = Conv2D(8, (2, 2), activation='relu', padding='same')(decoded)
	decoded = Conv2D(2, (2, 2), activation='tanh', padding='same')(decoded)
	return decoded