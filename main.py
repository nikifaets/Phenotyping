import utils
from normalize import normalize
import numpy as np
from autoencoder3D import Autoencoder
from test import test
import cv2
from extractData import read_channel
from visualize import visualize
from keras.models import load_model, Model


data = utils.load_array("data/PV/X.bc")
data = np.append(data, utils.load_array("data/IBA1/X.bc"), axis=0)
print("DATA", data.shape)
#data = data[:8000]
'''data = data/256

for i in range(len(data)):
	img1,img2  = test(i, data)
	img1 = cv2.medianBlur(img1, 2)
	img2 = cv2.medianBlur(img2, 2)'''

train_data = normalize(data)
print(np.max(data), type(data), data.shape)
ae = Autoencoder(save_file='weights.hdf5')
model = ae.train(train_data)
model.save('bestModel.h5')

#----------get output from latent space

layer_name = 'latent_space'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)
intermediate = intermediate_output[30]
np.save('compressed.npy',intermediate_output)
print("INTERMEDIATE LAYER", type(intermediate_output), intermediate_output.shape)
print("INTERMEDIATE LAYER", type(intermediate), intermediate.shape)
#----------visualize
num=450
samples=2
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


