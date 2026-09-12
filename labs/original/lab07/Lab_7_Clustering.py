import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
data = np.array(list(zip(x, y)))

plt.scatter(x, y)
plt.title("Scatter Plot of Sample Points")
plt.show()

linked = linkage(data, method="ward")
dendrogram(linked, orientation="top", distance_sort="ascending", show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendrogram (Ward linkage)")
plt.ylabel("Euclidean Distance")
plt.show()
