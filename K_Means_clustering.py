#Libraries
import numpy as np 
import matplotlib.pyplot as  plt 
import pandas as pd 
import seaborn as sns 

#Dataset
dataset = pd.read_csv(r'C:\Users\agnih\Desktop\ML\Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values 
#Here only Annual Income and Spending Score columns are used for identifing the patterns
#In the Unsupervised Learning there is no dependent variable 


#Elbow Method to find optimal number of clusters
from sklearn.cluster import KMeans
wcss = [] #List to store the wcss value for different K values
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++',random_state=1)
    #Here kmeans++ helps in not falling for the trap of random centroids 
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)#inertia_ gives the vslue of wcss
plt.plot(range(1,11),wcss)#Plot the graph
plt.title('Elbow Method')
plt.xlabel('No Of Clusters')
plt.ylabel('WCSS Value')
plt.show()


#Based on the graph therefore the number of clusters is 5
#Training the dataset for the Kmeans Clustering
kmeans = KMeans(n_clusters=5, init='k-means++',random_state=1)
y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)

#Visualizing the clusters
plt.scatter(X[y_kmeans == 0 ,0],X[y_kmeans == 0 ,1],s=100,c = 'red',label = 'Cluster1') 
#Selecting annual income and Spending Score in X of allrows belonging to Cluster 0
plt.scatter(X[y_kmeans == 1 ,0],X[y_kmeans == 1 ,1],s=100,c = 'blue',label = 'Cluster2')
plt.scatter(X[y_kmeans == 2 ,0],X[y_kmeans == 2 ,1],s=100,c = 'green',label = 'Cluster3')
plt.scatter(X[y_kmeans == 3 ,0],X[y_kmeans == 3 ,1],s=100,c = 'yellow',label = 'Cluster4')
plt.scatter(X[y_kmeans == 4 ,0],X[y_kmeans == 4 ,1],s=100,c = 'indigo',label = 'Cluster5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=250,c='cyan',label='Centroids')
plt.title('Mall Customers Clustering')
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
