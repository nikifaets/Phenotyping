import utils
from normalize import normalize
import numpy as np
from autoencoder3D import Autoencoder
import extract_data as extr
import keras
from get_samples import get_samples
import cv2
from extract_data import read_channel
from keras.models import load_model, Model, model_from_json
from keras import backend

## Set GPU using tf 
#import tensorflow as tf 
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.85 # you can set this value to be anything really
#from keras.backend.tensorflow_backend import set_session
#set_session(tf.Session(config=config))

data = utils.load_array("data/PV/X_nucleus.bc")[:2000]

train_data = data/(256*256)
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

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")


