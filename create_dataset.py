import numpy as np  
import utils 

def create_dataset(labels, data):

	#reorder the dataset so that positive samples are in the beginning of the array

	used = set()
	used_in_new = set()
	data_new = np.empty(data.shape, np.uint16)
	labels_new = np.zeros(len(data), np.bool_)

	counter=0
	labels_size = len(labels) 

	for i in range(0, len(labels)):

		if labels[i]:
			used.add(i)
			used_in_new.add(counter)
			data_new[counter] = data[i]
			counter+=1

	#reorder the array with labels
	for i in range(0, counter):
		labels_new[i] = True

	#reorder the array with images
	filler_new = 0

	for i in range(counter, len(data)):

		if i not in used_in_new:
			
			data_new[i] = data[filler_new]
			filler_new+=1
		

	#np.save("data/PV/balanced/PV_split.npy")
	#np.save("data/PV/balanced/labels.npy", labels_new)

	



data = utils.load_array("data/PV/X_cells_only.bc")

labels = np.load("data/PV/labels.npy")

create_dataset(labels,data)