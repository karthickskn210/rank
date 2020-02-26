# Import application modules
import random 
import pandas as pd
from sklearn.cluster import KMeans #K-Means Clustering
from sklearn.preprocessing import StandardScaler

################################################################
#	Function to groping the data by clustering
################################################################

def unSupervised(data):

	data = data.drop(['SQL_TYPE'],axis=1)
	#kData = kData.loc[['TOTAL_SECONDS','SNIPPETS','THROUGH_PUT_ROWS','THROUGH_PUT_BYTES']]	
	#print(data.head())
	scaler = StandardScaler()
	scaledData = scaler.fit_transform(data)

	random.seed(3)
	mySeed = random.randint(1,500)

	#Initialize our model
	kmeans = KMeans(n_clusters=5,random_state=mySeed)
	kmeans.fit(scaledData)
	
	#Find which cluster each data-point belongs to
	cluster = kmeans.predict(scaledData)
	
	cluster = cluster+1
	
	#Add the cluster vector to our DataFrame, X
	return cluster
	

	
if __name__ == '__main__':

	data = unSupervised(data) # expected data in data frame format
	#print(data.head()

