import numpy as np
import math

def normalize(X):
	#X is 4D array
	d1,d2,d3,d4 = X.shape
	X = np.reshape(X,(d1*d2*d3*d4))
	print(X.shape)
	mean = np.mean(X)
	std = np.std(X)

	X = (X - mean)/std

	max_val = np.max(X)
	min_val = np.min(X)
	addition = (1-max_val/(max_val-min_val))
	scale = (max_val-min_val)

	print("BEFORE NORMALIZATION", np.mean(X), np.std(X), np.max(X), np.min(X), "SCALE and ADD", scale, addition)

	X = X/scale + addition

	print("NORMALIZATION",np.mean(X), np.std(X), np.max(X), np.min(X))
	#X = (X - np.min(X))/(np.max(X)-np.min(X))
	

	X = np.reshape(X, (d1,d2,d3,d4))
	print(X.shape, np.max(X), np.min(X),np.mean(X))
	return X