import numpy as np  

nc = np.load("data/IBA1/nuclei_centers.npy")
gt = np.load("data/IBA1/ground_truth.npy")
labels = np.empty(len(nc), np.bool_)
counter=0
for i in range(0, len(nc)):

	labels[i]=False
	center = nc[i]
	for cell in gt:
		res = (center==cell)
		if res.all():
			print(i)
			labels[i]=True

np.save("data/IBA1/labels.npy", labels)