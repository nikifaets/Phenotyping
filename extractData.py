import numpy as np  
import cv2
import matplotlib.pyplot as plt
import utils

def read_channel(img, idx, layer, dim = 32):

	print("read_channel", idx, "len", len(img[idx]))
	res = np.zeros((dim,dim), np.uint8)
	for i in range(len(img[idx])):
		for j in range(len(img[idx][i])):
			res[i][j] = img[idx][i][j][layer]
	return res
'''
patches_PV = "data/PV/X.bc/"
PV = utils.load_array(patches_PV)
print(type(PV))
print(PV.shape)
print(PV[0].shape)
img = PV[0][:][:][0]
print(PV[100][28][10][1])
#PV = int(np.divide(PV,255))
PV = PV/256
print(PV[100][28][10][1])
PV = PV.astype(np.uint8)

print(np.max(PV))

centerNum = 500
img = read_channel(PV, centerNum, 0)
img1 = read_channel(PV, centerNum,1)

cv2.imshow("cell", img)
cv2.imshow("cel1", img1)
cv2.waitKey()
#PV = utils.load_array('__0.blp')
'''