import numpy as np  
import utils 
import random

def create_dataset(labels, data):

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

	# make the $counter first elements in labels True
	for i in range(0, counter):
		labels_new[i] = True

	filler_new = 0
	for i in range(counter, len(data)):

		if i not in used_in_new:
			
			print("haidee")
			data_new[i] = data[filler_new]
			filler_new+=1
		

	#np.save("data/IBA1/balanced/IBA1_split.npy", data_new)
	np.save("data/PV/balanced/labels.npy", labels_new)

	



data = utils.load_array("data/PV/X_nucleus.bc")
data_merged = utils.load_array("data/PV/X.bc")

labels = np.load("data/PV/labels.npy")

create_dataset(labels,data)