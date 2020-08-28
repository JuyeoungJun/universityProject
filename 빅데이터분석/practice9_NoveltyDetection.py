import numpy as np
from sklearn.svm import OneClassSVM as Novelty
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import confusion_matrix
from pyspark import SparkConf, SparkContext

# load the dataset using numpy loadtxt function
train = np.loadtxt("train.data", delimiter=',')
test = np.loadtxt("test.data", delimiter=',')
# Set the parameters we will use
nu = 0.1
gamma = 0.1
numPartition = 30
# Preprocess the dataset
# 1. We don't need the label of the training data points
trData = train[:,:-1]
# 2. We need the label of the test data points.
tsData, tsLabel = test[:,:-1], test[:,-1].astype(np.int).tolist()

# Configure the spark context
conf = SparkConf()
sc = SparkContext(conf=conf)
# Make data type as RDD, using sc.parallelize()
trRDDs = sc.parallelize(trData.tolist(), numPartition)
tsRDDs = sc.parallelize(tsData.tolist(), numPartition)
# Save the data points in memory
trRDDs.cache()
tsRDDs.cache()
# Train the novelty detection model using fit() function and training datapoints
novel = Novelty(nu=nu, kernel="rbf", gamma=gamma)
novel.fit(trRDDs.collect())
# Broad cast model
novel = sc.broadcast(novel)
# Predict the label of the test datapoints
# novelty detection model returns 1(inlier) or -1(outlier)
result = tsRDDs.map(lambda x:novel.value.predict(np.array(x).reshape(1,-1)))
result = result.collect()

prediction = [int(x[0]) for x in result]
#real = tsLabel.copy()
real = list(tsLabel)
# Get the performance value using the algorithm
accuracy = accuracy_score(real, prediction)
f1score = f1_score(real, prediction, average = 'macro')
tn, fp, fn, tp = confusion_matrix(prediction, real).ravel()
# You can save it in result.txt file
f = open("result.txt", "w")
f.write("Novelty Detection Results:\n")
f.write("ACC: {:.4f}, F1Score: {:.4f}\n".format(accuracy,f1score))
f.write("Confusion Matrix\n")
f.write("{} {}\n".format(tn,fp))
f.write("{} {}\n".format(fn,tp))
# Don't forget! Stop spark context
sc.stop()
