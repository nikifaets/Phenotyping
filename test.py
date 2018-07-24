from extractData import read_channel
import numpy as np
import cv2

def test(num,data):
	test = np.zeros((32,32,2))
	num = num
	first = read_channel(data, num-1, 0)
	sec = read_channel(data,num-1,1)
	
	return (first,sec)