import numpy as np

def normalize(X):
	#X is 4D array
	d1,d2,d3,d4 = X.shape
	X = np.reshape(X,(d1*d2*d3*d4))
	'''print(X.shape)
	mean = np.mean(X)
	std = np.std(X)
	print(mean, std, np.max(X))
	X = (X - mean)/std'''

	X = (X - np.min(X))/(np.max(X)-np.min(X))
	

	X = np.reshape(X, (d1,d2,d3,d4))
	print(X.shape, np.max(X))
	return X