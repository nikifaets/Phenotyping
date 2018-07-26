import numpy as np 
from sklearn.cluster import KMeans

compressed = np.load('compressed.py')

sample = compressed[5]
kmeans = KMeans(n_clusters=1, random_state=0).fit(compressed)
