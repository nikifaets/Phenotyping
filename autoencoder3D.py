import plaidml.keras
plaidml.keras.install_backend()
from keras.layers import Conv3D, MaxPooling3D, Conv2D, MaxPooling2D, Input, UpSampling2D
from keras.models import Model
from keras import optimizers
from normalize import normalize
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np 
from keras import regularizers
import networks

class Autoencoder:

	def __init__(self, save_file, img_rows=32, img_cols=32, depth=1):

		self.img_cols = img_cols
		self.img_rows = img_rows
		self.depth = depth
		self.save_file = save_file


	def createAutoencoder(self,inputs):

		decoder = networks.cellChannel1(inputs)
		model = Model(inputs, decoder)

		rmsprop = optimizers.RMSprop(lr=1e-3, rho=0.9, epsilon=None, decay=0.0)

		model.compile(optimizer='adam', loss='mean_squared_error')
		model.summary()

		return model

	def train(self, X):

		inputs = Input((self.img_rows, self.img_cols, self.depth))
		model = self.createAutoencoder(inputs)
		model_checkpoint = ModelCheckpoint(self.save_file, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)

		model.fit(X,X, batch_size=25, epochs=1)
		return model

	def predict(self, datas):
		
		inputs = Input((self.img_rows, self.img_cols,self.depth))
		model = self.createAutoencoder(inputs)
		model.load_weights(self.save_file)
		datas = normalize(datas)
		res = model.predict(datas, batch_size=20, verbose=1)
		return res