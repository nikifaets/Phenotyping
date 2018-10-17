import numpy as np  
import cv2
import matplotlib.pyplot as plt
import utils

def read_channel(img, idx, layer, shape=2, dim = 32):

	#read a sample patch
	if shape == 2:
		res = np.zeros((dim,dim), np.uint8)
		for i in range(len(img[idx])):
			for j in range(len(img[idx][i])):
				#print("res[i][j]", img[idx][i][j][layer], idx, i, j, layer)
				res[i][j] = img[idx][i][j][layer]

	elif shape ==3:
		res = np.zeros((dim,dim,1), np.uint16)
		for i in range(len(img[idx])):
			for j in range(len(img[idx][i])):
				#print("res[i][j]", img[idx][i][j][layer], idx, i, j, layer)
				res[i][j][0] = img[idx][i][j][layer]
	return res

def read_layer(img, layer, dim=32):

	#Read all the patches from a layer
	channel = list()
	for i in range(0, len(img)):
		img_curr = read_channel(img, i, layer, shape=3)
		channel.append(img_curr)
	channel = np.array(channel)
	return channel

def save_layers(PV, IBA):

	#save PV
	nucleus_channel_PV = read_layer(PV,0)
	utils.save_array("data/PV/X_nucleus.bc", nucleus_channel_PV)

	#save IBA
	nucleus_channel_IBA1 = read_layer(IBA,0)
	utils.save_array("data/IBA1/X_nucleus.bc", nucleus_channel_IBA1)


def get_positive():

#-------------read and write postivie indices

	nuclei_centers_PV = np.load('data/PV/nuclei_centers.npy')
	ground_truth_PV = np.load('data/PV/ground_truth.npy')

	positives_idx_PV = list()

	for center in nuclei_centers_PV:
		for i in range(0, len(ground_truth_PV)):

			if np.array_equal(center, ground_truth_PV[i]):
				positives_idx_PV.append(i)


	nuclei_centers_IBA = np.load('data/IBA1/nuclei_centers.npy')
	ground_truth_IBA = np.load('data/IBA1/ground_truth.npy')
	positives_idx_IBA = list()


	for center in nuclei_centers_IBA:
		for i in range(0, len(ground_truth_IBA)):

			if np.array_equal(center, ground_truth_IBA[i]):
				positives_idx_IBA.append(i)

	indices_PV = np.array(positives_idx_PV)
	indices_IBA = np.array(positives_idx_IBA)
	np.save('positive_indices_PV.npy',indices_PV)
	np.save('positive_indices_IBA1.npy', indices_IBA)
