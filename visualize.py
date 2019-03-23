import utils
import numpy as np
import matplotlib.pyplot as plt
from get_samples import get_samples
from normalize import normalize 
from keras.models import model_from_json
from extract_data import read_channel
import cv2

def visualize(test_imgs, gt_imgs):

	fig=plt.figure(figsize=(8, 8))
	columns = (len(gt_imgs)+len(test_imgs))*2
	rows = 2
	count=0
	for i in range(1, len(test_imgs)):
	    gt1 = gt_imgs[i][0]
	    gt2 = gt_imgs[i][1]
	    cell1 = test_imgs[i][0]
	    cell2 = test_imgs[i][1]
	    fig.add_subplot(rows, columns, i+count); count+=1
	    plt.imshow(cell2)
	plt.show()

def visualize_ae(predicted, data, start_idx, samples):

	
	#predicted = normalize(predicted)
	#data = normalize(data)
	
	print("range predicted", np.min(predicted), np.max(predicted))
	print("range data", np.min(data), np.max(data))



	for i in range(0,samples):

		img_processed = read_channel(predicted, i, layer=0, shape=3, dim=predicted.shape[1])
		img = read_channel(data, i, layer=0, shape=3, dim=data.shape[1])
		plt.subplot(samples, 2, 1+i)
		plt.imshow(img.reshape(img.shape[0],img.shape[1]), cmap=plt.get_cmap('gray'))
		plt.subplot(samples, 2, 1+5+i)

		plt.imshow(img_processed.reshape(img_processed.shape[0],img_processed.shape[1]), cmap=plt.get_cmap('gray'))

	plt.show()
	
	
data = utils.load_array("data/PV/X_nucleus.bc")/(256*256)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

samples = 5
start_idx = 205


predicted = loaded_model.predict(data[start_idx:start_idx+samples])

data*=256

predicted += (0-predicted.min())
predicted = predicted*(256/predicted.max())

print("range predicted", np.min(predicted), np.max(predicted))
print("range data", np.min(data), np.max(data))
visualize_ae(predicted, data[start_idx:start_idx+samples], start_idx, samples)