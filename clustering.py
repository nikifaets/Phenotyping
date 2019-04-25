import numpy as np 
from sklearn.cluster import KMeans, DBSCAN, MeanShift, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from sklearn.datasets.samples_generator import make_blobs
from scipy.stats import gaussian_kde
import utils
from keras.datasets import mnist
from keras.models import model_from_json, Model
from normalize import normalize



def clusterKmeans(data):

	dbscan = DBSCAN()

	kmeans = KMeans(n_clusters=2, random_state=0)
	labels = kmeans.fit_predict(compressed)

	return labels

def clusterDBSCAN(data):

	dbscan = DBSCAN(eps=2, min_samples=200)
	labels = dbscan.fit_predict(data)

	return labels

def clusterMeanShift(data):

	mshift = MeanShift()
	labels = mshift.fit_predict(data)

	return labels

def clusterAgglomerativeClustering(data):

	aggl = AgglomerativeClustering(n_clusters=10, linkage='complete')
	labels = aggl.fit_predict(data)

	return labels

def performPCA(data, n_components):

	ipca = IncrementalPCA(n_components=n_components, batch_size=n_components)
	ipca.fit(data)
	data = ipca.fit_transform(data)

	return data

def performTSne(data, n_components):

	tsne = TSNE(n_components=n_components)
	data = tsne.fit_transform(data)

	return data

def ae_predict(data):

	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("model.h5")

	#----------get output from latent space

	data = normalize(data)
	layer_name = 'latent_space'
	intermediate_layer_model = Model(inputs=loaded_model.input,
	                                 outputs=loaded_model.get_layer(layer_name).output)
	intermediate_output = intermediate_layer_model.predict(data)

	return intermediate_output

def evaluateClustering(labels_truth, labels):


	score = np.zeros((10,10), np.uint8)

	for i in range(0, len(labels_truth)):

		score[labels_truth[i]][labels[i]]+=1


	print("EVALUATION:")
	print(len(labels_truth), len(labels))
	for i in range(0,10):
		print(i)
		print(score[i])



test_size = 800
data = utils.load_array("data/PV/X.bc")[:test_size]
#compressed = np.load("output_PV.npy")[:test_size]
compressed = ae_predict(data)
labels_truth = np.load("data/PV/labels.npy")[:test_size]


print(compressed.shape)
compressed = np.reshape(compressed, (compressed.shape[0], compressed.shape[-1]*compressed.shape[1]*compressed.shape[2]))

labels = clusterKmeans(compressed)
compressed = performTSne(compressed,2)

print("LABELS NUM", len(labels))
#VISUALIZATION

counter=0
color = np.empty((len(labels_truth)), np.float32)
for i in range (0, len(labels_truth)):
	
	color[i] = labels_truth[i]*0.4

counter=0
color_clusters = np.empty((len(labels)), np.float32)
for i in range (0, len(labels)):
		color_clusters[i] = labels[i]*0.1

#seperate positive from negative
condition = labels_truth == 1
compressed_condition = np.empty((len(condition), compressed.shape[1]))

counter=0
for i in range(0, len(condition)):

	if condition[i]:
		compressed_condition[counter] = compressed[i]
		counter+=1
#compressed_condition = np.extract(condition, compressed)
print("CONDITION", condition)
print("AFTER CONDITION SHAPE", compressed_condition.shape)
print("OTIGINAL COMPRESSED SHAPE", compressed.shape)

print(counter)


print(labels_truth.shape)
print(color.shape)

plt.figure(1)
plt.scatter(compressed[:,0], compressed[:,1], c=color, zorder=1)
#plt.scatter(compressed_condition[:,0], compressed_condition[:,1], c="blue", zorder=1)

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
plt.figure(2)
plt.scatter(swapped_compressed[:,0], swapped_compressed[:,1], cmap='viridis', c = z_sorted, zorder=1)
plt.plot()

plt.figure(3)
plt.scatter(compressed[:,0], compressed[:,1], c=color_clusters)
plt.plot()

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

evaluateClustering(labels_truth, labels)