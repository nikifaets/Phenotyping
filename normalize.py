import numpy as np
import math

def normalize(X):

	if len(X.shape) == 3:

		d1,d2,d3 = X.shape
		X = np.reshape(X,(d1*d2*d3))
		print(X)
		mean = np.mean(X)
		std = np.std(X)

		X = (X - mean)/std

		
		X = np.reshape(X, (d1,d2,d3))
			
		return X

	elif len(X.shape) == 4:
		
		d1,d2,d3,d4 = X.shape
		X = np.reshape(X,(d1*d2*d3*d4))
		mean = np.mean(X)
		std = np.std(X)

		X = (X - mean)/std
		X -= np.min(X)
		

		X = np.reshape(X, (d1,d2,d3,d4))
		return X

	