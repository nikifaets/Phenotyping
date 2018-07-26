import numpy as np 
from sklearn.cluster import KMeans

compressed = np.load('compressed.npy')
print(compressed.shape)
compressed = np.reshape(compressed, (compressed.shape[0], compressed.shape[-1]))
print(compressed.shape)
sample = compressed[5]
kmeans = KMeans(n_clusters=2, random_state=0).fit_predict(compressed)
print(kmeans.shape)
print(np.min(kmeans))
cluster1 = np.argmin(kmeans)
cluster2 = np.argmax(kmeans)
print(len(cluster2))
print(len(cluster1))
