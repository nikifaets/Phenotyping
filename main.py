import utils
from normalize import normalize
import visualize
import numpy as np
from autoencoder3D import Autoencoder
import extract_data as extr
import keras
from get_samples import get_samples
import cv2
from extract_data import read_channel
from keras.models import load_model, Model
from keras import backend

## Set GPU using tf 
#import tensorflow as tf 
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.85 # you can set this value to be anything really
#from keras.backend.tensorflow_backend import set_session
#set_session(tf.Session(config=config))

<<<<<<< HEAD
#data = utils.load_array("data/PV/X.bc")
#data_both_channels = data_both_channels[:10]
'''temp = np.moveaxis(data_both_channels,-1,1)
data = np.zeros((data_both_channels.shape[0], 1, data_both_channels.shape[1], data_both_channels.shape[2]))
for i in range(len(data_both_channels)):
	nucl, cells = test(i, data_both_channels)
	
	data[i][0] = temp[i][1]
data=np.moveaxis(data,1,-1)
np.save("cells_PV.npy", data)'''
data = np.load("cells_PV.npy")
#data = data[:1000]
=======
data_pv = utils.load_array("data/PV/X_nucleus.bc")

data = data_pv[:8000]
>>>>>>> 518b64dc0ff25475924c07731194b76e9266aa4a
train_data = normalize(data)
print("data range", np.min(train_data), np.max(train_data))
print(np.max(data), type(data), data.shape)
ae = Autoencoder(save_file='weights-batviktor2.hdf5', depth=1)
model = ae.train(train_data)
res = model.predict(train_data)

#----------get output from latent space
layer_name = 'latent_space'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(train_data)
np.save('output_PV.npy',intermediate_output)

#----------visualize
visualize.visualize_ae(num=700, samples=3, model=model, data=train_data)


<<<<<<< HEAD
counter = 0
for j in gd_imgs:
	gd1,gd2 = j
	cv2.imshow("ground_truth1_"+str(counter), gd1)
	cv2.imshow("ground_truth2_"+str(counter), gd2)
	counter+=1
	
cv2.waitKey()
=======


>>>>>>> 518b64dc0ff25475924c07731194b76e9266aa4a
