import numpy as np 
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from sklearn.datasets.samples_generator import make_blobs


compressed = np.load('compressed12.npy')
positive = np.load('positive_indices.npy')
positive_PV = np.load('data/PV/ground_truth.npy')

print(compressed.shape)
compressed = np.reshape(compressed, (compressed.shape[0], compressed.shape[-1]*compressed.shape[1]*compressed.shape[2]))
print(compressed.shape)
sample = compressed[5]
kmeans = KMeans(n_clusters=2, random_state=0)


labels = kmeans.fit_predict(compressed)
labels_pos = np.zeros(labels.shape)
labels_pos_only = np.zeros(len(positive_PV))
compressed_pos_only = np.ones((len(positive_PV), compressed.shape[1]))
print('SHAPE', compressed_pos_only.shape, compressed.shape)
# make negative samples 0 and positive -1
for i in range(len(positive_PV)):

	if positive[i] < 13570:
		labels_pos[positive[i]] = 1
		#compressed_pos_only[] = 

print("min and max", np.min(labels_pos), np.max(labels_pos), len(labels_pos))
print('SHAPE', compressed_pos_only.shape, compressed.shape)
purple = list()
yellow = list()
labels_yellow = list()
labels_purple = list()




#REDUCING WITH PCA 
ipca = IncrementalPCA(n_components=2, batch_size=2)
ipca.fit(compressed)
compressed = ipca.transform(compressed)
print("PCA TYPE AND SHAPE", type(ipca))
print("INPUT TYPE AND SHAPE", type(compressed), compressed.shape)

for i in range(len(labels_pos)):
	if labels_pos[i] == 0:
		yellow.append(compressed[i])
		labels_yellow.append(0)
	else:
		purple.append(compressed[i])
		labels_purple.append(1)

purple = np.array(purple)
yellow = np.array(yellow)
labels_purple = np.array(labels_purple)
labels_yellow = np.array(labels_yellow)
print("PURPLE AND YELLOW", purple.shape, yellow.shape)
#VISUALIZATION

'''X, y_true = make_blobs(n_samples=len(compressed), centers=2,
                       cluster_std=0.60, random_state=0, )'''
#plt.scatter(compressed[:,0], compressed[:,1], cmap='viridis', c = labels)
print('SHAPE', compressed.shape)
plt.scatter(yellow[:,0], yellow[:,1], cmap='viridis', c = 'green', zorder=1)
plt.scatter(purple[:,0], purple[:,1], cmap='viridis', c = 'red',zorder=2)
plt.plot()
plt.show()

counter = 0
counterTrue=0
for i in range(len(labels)):
	if labels[i] == 1:
		counter+=1
	if labels[i]==1 and labels_pos[i]==1:
		counterTrue+=1

print("COUNTER", counter, counterTrue)
cluster1 = np.argmin(labels)
cluster2 = np.argmax(labels)

