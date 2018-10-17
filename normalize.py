import numpy as np
import math

def normalize(X):

	if len(X.shape) == 3:
		#X is 3D array
		d1,d2,d3 = X.shape
		X = np.reshape(X,(d1*d2*d3))
		print(X)
		mean = np.mean(X)
		std = np.std(X)

		#X = (X - mean)/std

		max_val = np.max(X)
		min_val = np.min(X)
		addition = (1-max_val/(max_val-min_val))
		scale = (max_val-min_val)


		X = X/scale + addition
		

		X = np.reshape(X, (d1,d2,d3))
		'''X_new = np.zeros((d1,d2,d3,1), np.float32)
		for i in range(len(X)):
			for j in range(len(X[i])):
				for l in range(len(X[i][j])):
					X_new[i][j][l][0] = X[i][j][l]'''
			
		return X
	elif len(X.shape) == 4:
		#X is 4D array
		d1,d2,d3,d4 = X.shape
		X = np.reshape(X,(d1*d2*d3*d4))
		mean = np.mean(X)
		std = np.std(X)

		#X = (X - mean)/std

		max_val = np.max(X)
		min_val = np.min(X)
		addition = (1-max_val/(max_val-min_val))
		scale = (max_val-min_val)


		X = (X-min_val)/(scale)-0.5
		

		X = np.reshape(X, (d1,d2,d3,d4))
		return X

	