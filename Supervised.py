# Import system modules
import os

# Import application modules
import numpy as np 
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import pickle

#############################################################################
#	Class for random forest
#############################################################################
class Supervised(object):

	def __init__(self,n_estimators, max_depth, random_state):

		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.random_state = random_state
		self.model = RandomForestClassifier(n_estimators = self.n_estimators, max_depth = self.max_depth, random_state = self.random_state)

	def split(self,dataGetting,split_size,random_state):

		self.data = dataGetting
		self.test_size = split_size
		self.random_state = random_state


		self.data['SQL_TYPE']=self.data['SQL_TYPE'].map({'Internal': 0,'External': 1,'Internal-scan':2,'Internal-broadcast':3})
		#print(self.data.head())

		self.data = self.data[['SQL_TYPE','TOTAL_SECONDS','SNIPPETS','THROUGH_PUT_ROWS','THROUGH_PUT_BYTES','Clusters']]

		inputX = self.data.iloc[:, 0:-1].values

		inputY = self.data.iloc[:, -1].values
			

		xTrain, xTest, yTrain, yTest = train_test_split(inputX, inputY, test_size = self.test_size, random_state = self.random_state)

		return xTrain, xTest, yTrain, yTest


	def train(self,xTrain,yTrain):

		self.x_train = scale(xTrain)
		self.y_train = yTrain

		#print('Training Features Shape:', self.x_train.shape)
		#print('Training Labels Shape:', self.y_train.shape)

		self.model.fit(self.x_train,self.y_train)
		

	def test(self,xTest):

		self.x_test = scale(xTest)
		
		output = self.model.predict(self.x_test)
		confident = self.model.predict_proba(self.x_test)
		print(confident)
		return output,confident

	def accuracy(self,data1,data2):
		
		self.data1 = data1
		self.data2 = data2		
		score = accuracy_score(self.data1, self.data2)*100

		return score
	
	def predictValue(self,mod,data):

		self.mod = mod
		self.data = data
		
		output = self.mod.predict(self.data)
		
		return output

	def save(self,directory):

		if not os.path.exists(directory):
			os.makedirs(directory)
	
		filename = 'random_forest.pkl'
		pickle_file = os.path.join(directory,filename)

		with open(pickle_file,'wb') as file:
			pickle.dump(self.model,file)

		print("Model saved successfully")

		return pickle_file

	def restore(self,file):

		self.pkl_filename = file

		# Load from file
		with open(self.pkl_filename, 'rb') as file:
			pickle_model = pickle.load(file)

		print("Model restore successfully")
		return pickle_model
		
####################################################################################
#	Function to restore the saved model
####################################################################################

def restore(file,xTest):

	pkl_filename = file
	x_test = scale(xTest)

	# Load from file
	with open(pkl_filename, 'rb') as file:
		pickle_model = pickle.load(file)

	print("Model restore successfully")

	print("Test with saved model")
	output = pickle_model.predict(x_test)
	#print(output)	
	#x_test["Clusters"] = output
	return output
	#return x_test

###########################################################################
#	Function to mapping the string values into numerical values
###########################################################################

def split(dataGetting):

	data = dataGetting

	data['SQL_TYPE']=data['SQL_TYPE'].map({'Internal': 0,'External': 1,'Internal-scan':2,'Internal-broadcast':3})
		
	#print(data.head())

	return data

def scale(data):

	scaler = StandardScaler()
	return scaler.fit_transform(data)
