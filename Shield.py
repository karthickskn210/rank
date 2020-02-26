import Unsupervised
import Supervised

import pandas as pd 
pd.options.display.float_format = "{:.3f}".format

import os 

df = pd.read_csv('shield.csv')
#print(df.columns)

df = df.drop(['Unnamed: 0','PLAN_ID', 'SESSION_ID', 'START_TIMESTAMP','END_TIMESTAMP',
	'QUEUE_SECONDS'],axis=1)

dInt = df[df['SQL_TYPE'] == 'Internal']
dExt = df[df['SQL_TYPE'] == 'External']


def calculateProperty(data):

	property = []
	
	clust = [1,2,3,4,5,6]
	for i in range(1,6):
		clust[i] = data[data.Clusters==i]
		print('Cluster ' + str(int(i)) +' property ')
		print(clust[i].describe().loc['mean'])

		
out = Unsupervised.unSupervised(data=dInt)
dInt['Clusters']=out
#calculateProperty(data=dInt)

out = Unsupervised.unSupervised(data=dExt)
dExt['Clusters'] = out 
#calculateProperty(data=dExt)

frames = [dInt,dExt]
data = pd.concat(frames)

# Shuffle the data before train supervised model to avoid overfitting
data = data.sample(frac=1)
print(data.head())

Model = Supervised.Supervised(n_estimators=500,max_depth=24,random_state=42)
print("Model initalised successfully")

xTrain, xTest, yTrain, yTest = Model.split(dataGetting=data,split_size=0.25,random_state=0)

Model.train(xTrain,yTrain)

out,confident = Model.test(xTest)
accuracy = Model.accuracy(data1=out,data2=yTest)
print(accuracy)

con = pd.DataFrame(confident)
"""
print(con.head())

score = con.apply(pd.DataFrame.describe().loc['mean'],axis=1)
print(score.head())
score = con.describe().loc['mean'(axis=1)
con['Mean'] = score
"""
con['Clusters'] = out
print(con.head())
#con.to_csv('confidentComp.csv')

current=os.getcwd()
toSave = os.path.join(current,'saved_model') 
pic_file = Model.save(directory = toSave)
