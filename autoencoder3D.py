from keras.layers import Conv3D, MaxPooling3D, Conv2D, MaxPooling2D, Input, UpSampling2D
from keras.models import Model
from normalize import normalize
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np 

class Autoencoder:

	def __init__(self, save_file, img_rows=32, img_cols=32, depth=2):

		self.img_cols = img_cols
		self.img_rows = img_rows
		self.depth = depth
		self.save_file = save_file

	def encode(self, data):

		'''encoded = Conv2D(filters=24, kernel_size=(2,2), activation='relu', padding='same')(data)
		encoded = MaxPooling2D((2,2))(encoded)
		#encoded = Conv3D(filters=24, kernel_size=(3,3,2), activation='relu', padding='same')(encoded)'''

		encoded = Conv2D(32, (3,3), activation='relu',padding='same')(data)
		encoded = MaxPooling2D((2,2))(encoded)
		encoded = Conv2D(64, (3,3), activation='relu',padding='same')(encoded)
		encoded = MaxPooling2D((2,2))(encoded)
		encoded = Conv2D(128, (2,2), activation='relu',padding='same')(encoded)
		encoded = Conv2D(192, (2,2), activation='relu',padding='same')(encoded)
		encoded = MaxPooling2D((2,2))(encoded)
		encoded = Conv2D(256, (2,2), padding='same')(encoded)

		return encoded

	def decode(self,data):

		'''decoded = Conv2D(filters=24, kernel_size=(2,2), activation='relu', padding='same')(data)
		#decoded = UpSampling2D((2,2))(decoded)
		decoded=Conv2D(filters=32, kernel_size=(2,2), activation='sigmoid')(decoded)
		#decoded = Conv3D((2,2,2), activation='relu', padding='same')(decoded)'''

		decoded = Conv2D(256, (2,2), activation='relu', padding='same')(data)
		decoded = UpSampling2D((2,2))(decoded)
		decoded = Conv2D(128, (2,2), activation='relu', padding='same')(decoded)
		decoded = Conv2D(128, (2,2), activation='relu', padding='same')(decoded)
		decoded = UpSampling2D((2,2))(decoded)
		decoded = Conv2D(92, (3,3), activation='relu', padding='same')(decoded)
		decoded = Conv2D(64, (3,3), activation='relu', padding='same')(decoded)
		decoded = UpSampling2D((2,2))(decoded)
		decoded = Conv2D(32, (3,3), activation='relu', padding='same')(decoded)
		decoded = Conv2D(2, (3,3), activation='tanh', padding='same')(decoded)
  
		return decoded

	def createAutoencoder(self,inputs):

		encoder = self.encode(inputs)
		decoder = self.decode(encoder)
		model = Model(inputs, decoder)
		model.compile(optimizer='adam', loss='mean_squared_error')
		model.summary()

		return model

	def train(self, X):

		inputs = Input((self.img_rows, self.img_cols, self.depth))
		model = self.createAutoencoder(inputs)
		model_checkpoint = ModelCheckpoint(self.save_file, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
		model.fit(X,X, batch_size=40, epochs=2)

		return model

	def predict(self, datas):
		
		inputs = Input((self.img_rows, self.img_cols,self.depth))
		model = self.createAutoencoder(inputs)
		model.load_weights(self.save_file)
		datas = normalize(datas)
		res = model.predict(datas, batch_size=40, verbose=1)
