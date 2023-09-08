#Libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

#Dataset
dataset = pd.read_csv(r'C:\Users\agnih\Desktop\ML\Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values 


#Dendogram to identify the optimal number of clusters
import scipy.cluster.hierarchy as sch 
dendo = sch.dendrogram(sch.linkage(X,method = 'ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

#Training Hierarchical Model Clustering
from sklearn.cluster import AgglomerativeClustering
hierarchical =  AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
#Number of clusters are defined by understanding the Demograph
# affinity is simply the type of distance that will be computed in order to measure the variants within  clusters.
y_hc = hierarchical.fit_predict(X)

#Visualizing the clusters
plt.scatter(X[y_hc == 0 ,0],X[y_hc == 0 ,1],s=100,c = 'red',label = 'Cluster1') 
plt.scatter(X[y_hc == 1 ,0],X[y_hc == 1 ,1],s=100,c = 'blue',label = 'Cluster2')
plt.scatter(X[y_hc == 2 ,0],X[y_hc == 2 ,1],s=100,c = 'green',label = 'Cluster3')
plt.scatter(X[y_hc == 3 ,0],X[y_hc == 3 ,1],s=100,c = 'yellow',label = 'Cluster4')
plt.scatter(X[y_hc == 4 ,0],X[y_hc == 4 ,1],s=100,c = 'indigo',label = 'Cluster5')
plt.title('Mall Customers Clustering')
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
