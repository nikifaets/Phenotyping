from extract_data import read_channel
import numpy as np
import cv2

def get_samples(num,data):

	last_idx = data.shape[-1]
	test = np.zeros((32,32,2))
	first = read_channel(data, num, 0)
	sec = read_channel(data,num,last_idx-1)
	
	return (first,sec)