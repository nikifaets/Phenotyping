import numpy as np  
import cv2
import matplotlib.pyplot as plt
import utils

def read_channel(img, idx, layer, shape=2, dim = 32):

	if shape == 2:
		res = np.zeros((dim,dim), np.uint8)

		for i in range(len(img[idx])):

			for j in range(len(img[idx][i])):

				res[i][j] = img[idx][i][j][layer]

	elif shape ==3:

		res = np.zeros((dim,dim,1), np.uint16)

		for i in range(len(img[idx])):

			for j in range(len(img[idx][i])):

				res[i][j][0] = img[idx][i][j][layer]
	return res

def read_layer(img, layer, dim=32):

	channel = list()
	for i in range(0, len(img)):
		img_curr = read_channel(img, i, layer, shape=3)
		channel.append(img_curr)
	channel = np.array(channel)
	return channel

def save_layers(PV, IBA):

	#save PV
	nucleus_channel_PV = read_layer(PV,0)
	utils.save_array("data/PV/X_cells_only.bc", nucleus_channel_PV)

	#save IBA
	nucleus_channel_IBA1 = read_layer(IBA,0)
	utils.save_array("data/IBA1/X_cells_only.bc", nucleus_channel_IBA1)


