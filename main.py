import utils
from normalize import normalize
import numpy as np
from autoencoder3D import Autoencoder
from test import test
import cv2
from extractData import read_channel
from visualize import visualize
from keras.models import load_model, Model

## Set GPU using tf 
import tensorflow as tf 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.85 # you can set this value to be anything really
from keras.backend.tensorflow_backend import set_session
set_session(tf.Session(config=config))

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
train_data = normalize(data)
print(np.max(data), type(data), data.shape)
ae = Autoencoder(save_file='weights-batviktor2.hdf5', depth=1)
print("SDFDSFDSFDSFDSDFS")
model = ae.train(train_data)
#model.save('bestModel-batviktor2mix.h5')

#----------get output from latent space

layer_name = 'latent_space'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(train_data)
intermediate = intermediate_output[30]
print(type(intermediate_output), intermediate_output.shape)
print(type(intermediate_layer_model))
np.save('compressed12.npy',intermediate_output)
print("INTERMEDIATE LAYER", type(intermediate_output), intermediate_output.shape)
print("INTERMEDIATE LAYER", type(intermediate), intermediate.shape)
#----------visualize
num=700
samples=3
test_data=train_data[num:num+samples]
#print("test_data Type", type(test_data), test_data.shape)
res = model.predict(test_data)

#print("Type", type(res), type(res[0][0]), len(res))

#print(np.max(res), np.max(data))
#print("STDEV", np.std(res), np.std(data), "MEAN", np.mean(res), np.mean(data))
#print('RES', res[0])

res*=255
data = data/256
#print(np.max(res), "MEAN", np.mean(res))
test_imgs = list()
gd_imgs = list()
for i in range(samples):
	print('i', i)
	cell1,cell2 = test(i,res)
	gd1, gd2 = test(num+i, data)
	gd1 = cv2.resize(gd1,None, fx=4, fy=4, interpolation = cv2.INTER_LINEAR)
	gd2 = cv2.resize(gd2,None,fx=4, fy=4, interpolation = cv2.INTER_LINEAR)
	cell1 = cv2.resize(cell1,None,fx=4, fy=4, interpolation = cv2.INTER_LINEAR)
	cell2 = cv2.resize(cell2,None,fx=4, fy=4, interpolation = cv2.INTER_LINEAR)
	test_imgs.append((cell1, cell2))
	gd_imgs.append((gd1,gd2))

print(len(gd_imgs), len(test_imgs))

counter = 0
for i in test_imgs:
	cell1,cell2 = i
	cv2.imshow("nucleus1_"+str(counter), cell1)
	cv2.imshow("nucleus2_"+str(counter), cell2)
	counter+=1

counter = 0
for j in gd_imgs:
	gd1,gd2 = j
	cv2.imshow("ground_truth1_"+str(counter), gd1)
	cv2.imshow("ground_truth2_"+str(counter), gd2)
	counter+=1
	
cv2.waitKey()


