################################################## Start of Code ##########################################
from __future__ import print_function
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import re
import time
import numpy as np
from operator import add
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.classification import LogisticRegressionModel, LogisticRegressionWithSGD
import gc
################################ Run FileFormatter.py before running this code ############################
if __name__=="__main__":
	sc = SparkContext(appName="LargeStreamHW2")
	sc.setLogLevel("ERROR")
	gc.collect()
	######################################### Character list ##############################################
	chalist=['\n',' ','!','"','#','$','%','&',"'",'(',')','*',',','-','.','/','0','1'
,'2','3','4','5','6','7','8','9',':',';','?','@','A','B','C','D','E','F'
,'G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X'
,'Y','Z','[',']','_','a','b','c','d','e','f','g','h','i','j','k','l','m'
,'n','o','p','q','r','s','t','u','v','w','x','y','z','\x91','\x92','\x93'
,'\x94','\xe0','\xe2','\xe8','\xe9']

	######################################### TReading files ##############################################
	data_s1 = sc.textFile("/home/paperspace/Downloads/s1.txt",20)
	data_split_s1 = data_s1.map(lambda x: x.split(" "))
	c_s1 = data_split_s1.map(lambda x: LabeledPoint(x[-1], x[:-1]))

	data_s2 = sc.textFile("/home/paperspace/Downloads/s2.txt", 50)
	data_split_s2 = data_s2.map(lambda x: x.split(" "))
	c_s2 = data_split_s2.map(lambda x: LabeledPoint(x[-1], x[:-1]))

	data_s3 = sc.textFile("/home/paperspace/Downloads/s3.txt", 50)
	data_split_s3 = data_s3.map(lambda x: x.split(" "))
	c_s3 = data_split_s3.map(lambda x: LabeledPoint(x[-1], x[:-1]))

	data_s4 = sc.textFile("/home/paperspace/Downloads/s4.txt",20)
	data_split_s4 = data_s4.map(lambda x: x.split(" "))
	c_s4 = data_split_s4.map(lambda x: LabeledPoint(x[-1], x[:-1]))

	data_s_full = sc.textFile("/home/paperspace/Downloads/s_full.txt")
	data_split_s_full = data_s_full.map(lambda x: x.split(" "))
	c_s_full = data_split_s_full.map(lambda x: LabeledPoint(x[-1], x[:-1]))
	
	
	print('Data loaded')
	########################################### Training and testing models ##############################
	path = "C:\\Spark\\spark-2.1.0-bin-hadoop2.7\\project_models"
	model_s1_10 = RandomForest.trainClassifier(c_s1, numClasses=93, categoricalFeaturesInfo={}, numTrees=10)
	#model_s1_10.save(sc, 'model_1_10')
	predictions_s1_10 = model_s1_10.predict(c_s4.map(lambda x: x.features))
	collect_s1_10 = predictions_s1_10.collect()
	print('Model_s1_10 Done!')
	
	model_s2_10 = RandomForest.trainClassifier(c_s2, numClasses=93, categoricalFeaturesInfo={}, numTrees=10)
	#model_s2_10.save(sc, 'model_2_10')
	predictions_s2_10 = model_s2_10.predict(c_s4.map(lambda x: x.features))
	collect_s2_10 = predictions_s2_10.collect()
	print('Model_s2_10 Done!')
	
	model_s3_10 = RandomForest.trainClassifier(c_s3, numClasses=93, categoricalFeaturesInfo={}, numTrees=10)
	predictions_s3_10 = model_s3_10.predict(c_s4.map(lambda x: x.features))
	collect_s3_10 = predictions_s3_10.collect()
	print('Model_s3_10 Done!')

	model_s1_20 = RandomForest.trainClassifier(c_s1, numClasses=93, categoricalFeaturesInfo={}, numTrees=20)
	predictions_s1_20 = model_s1_20.predict(c_s4.map(lambda x: x.features))
	collect_s1_20 = predictions_s1_20.collect()
	print('Model_s1_20 Done!')
	
	model_s2_20 = RandomForest.trainClassifier(c_s2, numClasses=93, categoricalFeaturesInfo={}, numTrees=20)
	predictions_s2_20 = model_s2_20.predict(c_s4.map(lambda x: x.features))
	collect_s2_20 = predictions_s2_20.collect()
	print('Model_s2_20 Done!')
	
	model_s3_20 = RandomForest.trainClassifier(c_s3, numClasses=93, categoricalFeaturesInfo={}, numTrees=20)
	predictions_s3_20 = model_s3_20.predict(c_s4.map(lambda x: x.features))
	collect_s3_20 = predictions_s3_20.collect()
	print('Model_s3_20 Done!')

	model_s1_30 = RandomForest.trainClassifier(c_s1, numClasses=93, categoricalFeaturesInfo={}, numTrees=30)
	predictions_s1_30 = model_s1_30.predict(c_s4.map(lambda x: x.features))
	collect_s1_30 = predictions_s1_30.collect()
	print('Model_s1_30 Done!')
	
	model_s2_30 = RandomForest.trainClassifier(c_s2, numClasses=93, categoricalFeaturesInfo={}, numTrees=30)
	predictions_s2_30 = model_s2_30.predict(c_s4.map(lambda x: x.features))
	collect_s2_30 = predictions_s2_30.collect()
	print('Model_s2_30 Done!')
	
	model_s3_30 = RandomForest.trainClassifier(c_s3, numClasses=93, categoricalFeaturesInfo={}, numTrees=30)
	predictions_s3_30 = model_s3_30.predict(c_s4.map(lambda x: x.features))
	collect_s3_30 = predictions_s3_30.collect()
	print('Model_s3_30 Done!')
	
	model_s_full_100 = RandomForest.trainClassifier(c_s_full, numClasses=93, categoricalFeaturesInfo={}, numTrees=100)
	predictions_s_full_100 = model_s_full_100.predict(c_s4.map(lambda x: x.features))
	collect_s_full_100 = predictions_s_full_100.collect()
	print('Model_s1_100 Done!')
	
	########################################### Creating Ensemble Models ##############################
	label_s4 = data_s4.map(lambda x: x[-1])
	collect_label_s4 = label_s4.collect()
	delta1=np.array([1,1,1])
	## Setting the length of loop
	filerange=100000
	Error=np.zeros((filerange,20))
	AgError=np.zeros((filerange,20))
	## Initializing weights
	w9=np.zeros((filerange,9))
	w=np.array([[0.1,0.2,0.7],[0.4,0.5,0.1],[0.3,0.4,0.3]])
	wf=w.reshape((-1,9))/4
	for i in range(filerange):
		
        #z_10 = [LabeledPoint(collect_label_s4[i], [collect_s1_10[i], collect_s2_10[i], collect_s3_10[i]])]

        #z_20 = [LabeledPoint(collect_label_s4[i], [collect_s1_20[i], collect_s2_20[i], collect_s3_20[i]])]

        #z_30 = [LabeledPoint(collect_label_s4[i], [collect_s1_30[i], collect_s2_30[i], collect_s3_30[i]])]

		z_avg_10 = round((collect_s1_10[i] + collect_s2_10[i] + collect_s3_10[i])/3.0)

		z_avg_20 = round((collect_s1_20[i] + collect_s2_20[i] + collect_s3_20[i])/3.0)

		z_avg_30 = round((collect_s1_30[i] + collect_s2_30[i] + collect_s3_30[i])/3.0)

		z_100 = collect_s_full_100[i]
		## Setting Input and target for SGD
		X=np.array([[collect_s1_10[i], collect_s2_10[i], collect_s3_10[i]],[collect_s1_20[i], collect_s2_20[i], collect_s3_20[i]],[collect_s1_30[i], collect_s2_30[i], collect_s3_30[i]]])
		Y_true=np.array([collect_label_s4[i], collect_label_s4[i], collect_label_s4[i]])
		Y = np.sum(np.multiply(w, X), axis=1)

		xf=X.reshape((-1,9))
		yf=np.sum(np.multiply(wf, xf))
		w9[i,:]=wf[0,:]
		Y_hat = Y_true.astype(float)
		ef=yf-Y_hat[0]
		print(ef)
        E=Y-Y_true.astype(float)
		## Learning rate
        lr=0.00001
		## Setting a threshold for error
		if abs(E[0])<1:
			delta1[0]=0
		if abs(E[1])<1:
			delta1[1]=0
		if abs(E[2])<1:
			delta1[2]=0
		## Updating Weights of SGD 3
		delta1=lr*E
		delta=delta1.reshape((3,1))*X
		w-=delta
		df=lr*ef
		if abs(ef)<1:
			df=0
		## Updating Weights of SGD 9
		wf-=df*xf
		print(w)		
		print(wf)
		print(Y)
		print('########################### Iterartion %i ##############################' %(i))
		#print('########################################################################')
		#print('############################## Model 10 trees ##########################')
		#print('Book 1 Model Error: %f  Output: %s'%(collect_s1_10[i]-Y_hat[0],repr(chalist[int(round(collect_s1_10[i]))])))
		#print('Book 2 Model Error: %f  Output: %s'%(collect_s2_10[i]-Y_hat[0],repr(chalist[int(round(collect_s2_10[i]))])))
		#print('Book 3 Model Error: %f  Output: %s'%(collect_s3_10[i]-Y_hat[0],repr(chalist[int(round(collect_s3_10[i]))])))
		#print('Equal Probability Model Error: %f  Output: %s'%(z_avg_10-Y_hat[0],repr(chalist[int(round(z_avg_10))])))
		#print('SGD Model Error: %f  Output: %s'%(E[0],repr(chalist[int(round(Y[0]))])))
		#print('Correct Output: %s'%(repr(chalist[int(round(Y_hat[0]))])))
		#print('SGD weights',w[0])
		#print('########################################################################')
		#print('############################## Model 20 trees ##########################')
		#print('Book 1 Model Error: %f  Output: %s'%(collect_s1_20[i]-Y_hat[0],repr(chalist[int(round(collect_s1_20[i]))])))
		#print('Book 2 Model Error: %f  Output: %s'%(collect_s2_20[i]-Y_hat[0],repr(chalist[int(round(collect_s2_20[i]))])))
		#print('Book 3 Model Error: %f  Output: %s'%(collect_s3_20[i]-Y_hat[0],repr(chalist[int(round(collect_s3_20[i]))])))
		#print('Equal Probability Model Error: %f  Output: %s'%(z_avg_20-Y_hat[0],repr(chalist[int(round(z_avg_20))])))
		#print('SGD Model Error: %f  Output: %s'%(E[1],repr(chalist[int(round(Y[1]))])))
		#print('Correct Output: %s'%(repr(chalist[int(round(Y_hat[0]))])))
		#print('SGD weights',w[1])
		#print('########################################################################')
		#print('############################## Model 30 trees ##########################')
		#print('Book 1 Model Error: %f  Output: %s'%(collect_s1_30[i]-Y_hat[0],repr(chalist[int(round(collect_s1_30[i]))])))
		#print('Book 2 Model Error: %f  Output: %s'%(collect_s2_30[i]-Y_hat[0],repr(chalist[int(round(collect_s2_30[i]))])))
		#print('Book 3 Model Error: %f  Output: %s'%(collect_s3_30[i]-Y_hat[0],repr(chalist[int(round(collect_s3_30[i]))])))
		#print('Equal Probability Model Error: %f  Output: %s'%(z_avg_30-Y_hat[0],repr(chalist[int(round(z_avg_30))])))
		#print('SGD Model Error: %f  Output: %s'%(E[2],repr(chalist[int(round(Y[2]))])))
		#print('Correct Output: %s'%(repr(chalist[int(round(Y_hat[0]))])))
		#print('SGD weights',w[2])
		#print('########################################################################')
		#print('############################## Model 100 trees #########################')
		#print('Deep Model Error: %f  Output: %s'%(z_100-Y_hat[0],repr(chalist[int(round(z_100))])))
		#print('9 Model SGD Error: %f  Output: %s'%(ef,repr(chalist[int(round(yf))])))
		#print('Correct Output: %s'%(repr(chalist[int(round(Y_hat[0]))])))
		#print('########################################################################')
		Error[i,0]=collect_s1_10[i]-Y_hat[0]
		Error[i,1]=collect_s2_10[i]-Y_hat[0]
		Error[i,2]=collect_s3_10[i]-Y_hat[0]
		Error[i,3]=z_avg_10-Y_hat[0]
		Error[i,4]=E[0]
		Error[i,5]=-100
		Error[i,6]=collect_s1_10[i]-Y_hat[0]
		Error[i,7]=collect_s2_10[i]-Y_hat[0]
		Error[i,8]=collect_s3_10[i]-Y_hat[0]
		Error[i,9]=z_avg_10-Y_hat[0]
		Error[i,10]=E[1]
		Error[i,11]=-100
		Error[i,12]=collect_s1_10[i]-Y_hat[0]
		Error[i,13]=collect_s2_10[i]-Y_hat[0]
		Error[i,14]=collect_s3_10[i]-Y_hat[0]
		Error[i,15]=z_avg_10-Y_hat[0]
		Error[i,16]=E[2]
		Error[i,17]=-100
		Error[i,18]=z_100-Y_hat[0]
		Error[i,19]=ef
	CuError=np.cumsum(Error,axis=0)
	print(CuError.shape)
	## Calculating Aggregate error
	for j in range(len(CuError)):
		if j==0:
			AgError[j,:]=CuError[j,:]
		else:
			AgError[j,:]=CuError[j,:]/j	
	## Plotting figures
	plt.figure(1)	
	im=plt.imshow(Error,aspect='auto')
	plt.colorbar(im,orientation='horizontal')
	plt
	plt.ylabel('Lines')
	plt.xlabel('Models')
	plt.figure(2)
	im=plt.imshow(Error[-9:,:],aspect='auto')
	plt.colorbar(im,orientation='horizontal')
	plt.ylabel('Lines')
	plt.xlabel('Models')
	plt.figure(3)
	#plt.plot(AgError[:,0],'.-',label='Model s1_10')
	#plt.plot(AgError[:,1],'^-',label='Model s2_10')
	#plt.plot(AgError[:,2],'o-',label='Model s3_10')
	#plt.plot(AgError[:,3],'v-',label='Model Equal 10')
	#plt.plot(AgError[:,4],'s-',label='Model SGD 10')
	#plt.plot(AgError[:,6],'*-',label='Model s1_20')
	#plt.plot(AgError[:,7],'8-',label='Model s2_20')
	#plt.plot(AgError[:,8],'p-',label='Model s3_20')
	#plt.plot(AgError[:,9],'P-',label='Model Equal 20')
	#plt.plot(AgError[:,10],'h-',label='Model SGD 20')
	#plt.plot(AgError[:,12],'+-',label='Model s1_30')
	#plt.plot(AgError[:,13],'x-',label='Model s2_30')
	#plt.plot(AgError[:,14],'X-',label='Model s3_30')
	#plt.plot(AgError[:,15],'H-',label='Model Equal 30')
	#plt.plot(AgError[:,16],'|-',label='Model SGD 30')
	#plt.plot(AgError[:,18],'_-',label='Model Deep 100')
	#plt.plot(AgError[:,19],'D-',label='Model SGD 9')
	#plt.plot(AgError[:,0],label='Model s1_10')
	#plt.plot(AgError[:,1],label='Model s2_10')
	#plt.plot(AgError[:,2],label='Model s3_10')
	#plt.plot(AgError[:,3],label='Model Equal 10')
	plt.plot(AgError[:,4],label='Model SGD 10')
	#plt.plot(AgError[:,6],label='Model s1_20')
	#plt.plot(AgError[:,7],label='Model s2_20')
	#plt.plot(AgError[:,8],label='Model s3_20')
	#plt.plot(AgError[:,9],label='Model Equal 20')
	plt.plot(AgError[:,10],label='Model SGD 20')
	#plt.plot(AgError[:,12],label='Model s1_30')
	#plt.plot(AgError[:,13],label='Model s2_30')
	#plt.plot(AgError[:,14],label='Model s3_30')
	#plt.plot(AgError[:,15],label='Model Equal 30')
	plt.plot(AgError[:,16],label='Model SGD 30')
	plt.plot(AgError[:,18],label='Model Deep 100')
	plt.plot(AgError[:,19],label='Model SGD 9')
	plt.xlabel('Lines')
	plt.ylabel('Aggregate error')
	plt.legend()
	plt.figure(4)
	plt.plot(w9[:,0],label='weight 1 of sgd 9')
	plt.plot(w9[:,1],label='weight 2 of sgd 9')
	plt.plot(w9[:,2],label='weight 3 of sgd 9')
	plt.plot(w9[:,3],label='weight 4 of sgd 9')
	plt.plot(w9[:,4],label='weight 5 of sgd 9')
	plt.plot(w9[:,5],label='weight 6 of sgd 9')
	plt.plot(w9[:,6],label='weight 7 of sgd 9')
	plt.plot(w9[:,7],label='weight 8 of sgd 9')
	plt.plot(w9[:,8],label='weight 9 of sgd 9')
	plt.xlabel('Lines')
	plt.ylabel('Weights of sgd 9')

	plt.legend()
	plt.show()

		
################################################## End of Code ############################################