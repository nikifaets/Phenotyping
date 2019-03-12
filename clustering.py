import numpy as np 
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from sklearn.datasets.samples_generator import make_blobs
from scipy.stats import gaussian_kde


compressed = np.load('compressed12.npy')

labels_truth = np.load("data/PV/labels.npy")[:len(compressed)]

print(compressed.shape)
compressed = np.reshape(compressed, (compressed.shape[0], compressed.shape[-1]*compressed.shape[1]*compressed.shape[2]))
print(compressed.shape)
sample = compressed[5]
kmeans = KMeans(n_clusters=2, random_state=0)


labels = kmeans.fit_predict(compressed)

#REDUCING WITH PCA 
ipca = IncrementalPCA(n_components=2, batch_size=2)
ipca.fit(compressed)
compressed = ipca.transform(compressed)
print("PCA TYPE AND SHAPE", type(ipca))
print("INPUT TYPE AND SHAPE", type(compressed), compressed.shape)

#VISUALIZATION

plt.figure(1)

print(labels_truth)
print("SUM", np.sum(labels_truth))

counter=0

color = np.empty((len(labels_truth)), np.float32)
for i in range (0, len(labels_truth)):
	if labels_truth[i]==1:
		counter+=1
		color[i] = 0.2
	else:
		color[i] = 0.8

print(counter)
labels_truth+=1
labels_truth = labels_truth*100

print(labels_truth.shape)
print(color.shape)
plt.scatter(compressed[:,0], compressed[:,1], c=color)
plt.plot()


swapped_compressed = np.swapaxes(compressed, 1, 0)
#plt.figure(2)
# stack cells and non-cells
z = gaussian_kde(swapped_compressed)(swapped_compressed)
z_sorted = np.copy(z)
idx = z_sorted.argsort()
x, y, z_sorted = compressed[idx][0], compressed[idx][1], z_sorted[idx]
swapped_compressed = np.swapaxes(swapped_compressed, 0, 1)
print('SHAPE', swapped_compressed.shape)
#plt.scatter(swapped_compressed[:,0], swapped_compressed[:,1], cmap='viridis', c = z_sorted, zorder=1)
#plt.plot()
'''
plt.figure(3)

cleaned, z_condition = extr.extract_by_average(compressed, z)
print(z.shape, cleaned.shape, compressed.shape)
plt.scatter(cleaned[:,0], cleaned[:,1])
plt.plot()

plt.show()'''
plt.show()
np.save("z.npy", z)

truePositive=0
falsePositive=0
trueNegative=0
falseNegative=0

for i in range(0, len(labels_truth)):

	gt = labels_truth[i]
	km = labels[i]

	if gt == 1 and km ==1:
		truePositive+=1

	if gt ==1 and km == 0:
		falseNegative+=1

	if gt == 0 and km == 0:
		trueNegative+=1

	if gt == 0 and km == 1:
		falsePositive+=1

print("True Positive", truePositive)
print("False Positive", falsePositive)
print("True Negative", trueNegative)
print("False Negative", falseNegative)