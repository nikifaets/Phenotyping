import numpy as np
import utils


def extract_by_average(data,z, dims=2, save=False):

	print("IN EXTRACT", data.shape, z.shape)
	avg = np.average(z)
	condition = z>=avg
	z = np.extract(condition, z)
	size = len(z)

	cleaned = np.empty((size,dims))

	for i in range(0,dims):
		cleaned[:,i] = np.extract(condition,data[:,i])
	
	print("AFTER CONDITION", cleaned.shape, z.shape)

	if save:
		np.save(cleaned, save)
	return cleaned,z

def save_cleaned(initial, new_data, z, positives):

	original_data = []

	if initial.endswith(".bc"):
		original_data = utils.load_array(initial)

	elif initial.endswith(".npy"):
		original_data = np.load(initial)

	avg = np.average(z)
	condition = z>=avg
	z = np.extract(condition, z)

	print("TRUE IDX", condition.nonzero()[0])

	cleaned_shape = list(original_data.shape[1:])
	cleaned_shape.insert(0,len(z))
	cleaned_shape = tuple(cleaned_shape)

	cleaned = np.empty(cleaned_shape, np.uint16)
	print("in save",cleaned.shape)
	print("in save - condition", len(condition))
	print("in save - len original_data", len(original_data))


	counter=0
	for i in range(0, len(z)):

		#if counter>=2880:
			#print(counter)
		cleaned[counter] = original_data[i]
		counter+=1

	update_positives(condition, positives)
	np.save(new_data, cleaned)

def update_positives(condition, positives):

	counter = 0
	for i in range(0, len(positives)):
		print("condition", condition[positives[i]])
		idx = int(positives[i])
		if condition[idx]:
			counter+=1

	new_positives = np.empty(counter, np.uint16)
	counter = 0

	for i in range(0, len(positives)):

		idx = int(positives[i])
		if condition[idx]:
			print("yes")
			new_positives[counter] = idx
			counter+=1

	print(counter)
	np.save("cleaned_positive_indices_PV.npy", new_positives)
