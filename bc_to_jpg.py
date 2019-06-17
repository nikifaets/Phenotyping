import cv2
import numpy as np 
import utils 


def preprocess(data):

	kernel = np.ones((2,2), np.uint8)

	#for img in data:

		#img = cv2.medianBlur(data, 3)
	cv2.threshold(data, 20, np.max(data), cv2.THRESH_BINARY)
	data = cv2.medianBlur(data,5)
	return data


#pv = utils.load_array("data/IBA1/X_nucleus.bc")/256
data = np.load("data/IBA1/balanced/IBA1_split.npy")
data = data.reshape(data.shape[:-1])
print("SHAPE", data.shape)
#pv = data.astype(np.uint8)



iba = preprocess(data)
iba = iba.reshape((len(iba),32,32,1))
print("new shapes", iba.shape)
#print(pv.shape)
np.save("data/IBA1/balanced/IBA1_split_preprocessed_thresh.npy", iba)
'''counter=0
data = data.reshape(data.shape[:-1])/256
data = data.astype(np.uint8)
print(data.shape)
print(type(data[2][2][3]))

for img in data:
	#cv2.imshow("img", img)
	#print(img.shape, type(img[3][3]))
	#cv2.waitKey()
	cv2.imwrite("data/IBA1/balanced/images_jpg_preprocessed/iba"+str(counter)+".jpg", img)
	counter+=1
#print(pv.shape)'''