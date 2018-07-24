import utils
from normalize import normalize
import numpy as np
from autoencoder3D import Autoencoder
from test import test
import cv2
from extractData import read_channel


data = utils.load_array("data/PV/X.bc")
data = data[:9000]
data = normalize(data)
print(np.max(data), type(data), data.shape)
ae = Autoencoder(save_file='weights.hdf5')
model = ae.train(data)

num=600
samples=3
test_data=data[num:num+samples]
print("test_data Type", type(test_data), test_data.shape)
res = model.predict(test_data)

print("Type", type(res), len(res))
res*=255
data*=255

test_imgs = list()
gd_imgs = list()
for i in range(samples):
	print('i', i)
	cell1,cell2 = test(samples,res)
	gd1, gd2 = test(num+samples, data)
	gd1 = cv2.resize(gd1,None,fx=4, fy=4, interpolation = cv2.INTER_LINEAR)
	gd2 = cv2.resize(gd2,None,fx=4, fy=4, interpolation = cv2.INTER_LINEAR)
	cell1 = cv2.resize(cell1,None,fx=4, fy=4, interpolation = cv2.INTER_LINEAR)
	cell2 = cv2.resize(cell2,None,fx=4, fy=4, interpolation = cv2.INTER_LINEAR)
	test_imgs.append((cell1, cell2))
	gd_imgs.append((gd1,gd2))

print(len(gd_imgs), len(test_imgs))
counter = 0
for i,j in zip(test_imgs, gd_imgs):
	cell1,cell2 = i
	gd1,gd2 = j
	cv2.imshow("ground_truth1_"+str(counter), gd1)
	cv2.imshow("ground_truth2_"+str(counter), gd2)
	cv2.imshow("nucleus1_"+str(counter), cell1)
	cv2.imshow("nucleus2_"+str(counter), cell2)
	counter+=1
	
cv2.waitKey()



