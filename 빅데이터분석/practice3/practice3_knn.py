import time
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier as KNN

from pyspark import SparkConf, SparkContext

def LOAD_DATA(data):
    print("Loading {} dataset".format(data))
    mnist = fetch_openml(data)
    print("Successfully load data")
    return mnist

K = 11
numTrain = 30000
numTest = 10000
numTotal = numTrain + numTest

# If you get JAVA out of memory(OOM),
# then you can solve the problem with increasing numPartition
numPartition = 500

# Load data
mnist = LOAD_DATA('mnist_784')
data = mnist.data[:numTotal]
target = mnist.target[:numTotal]

start = time.time()

conf = SparkConf()
sc = SparkContext(conf=conf)

trData, tsData = data[:numTrain], data[numTrain:numTotal]
trLabel, tsLabel = target[:numTrain], target[numTrain:numTotal]

trRDDs = sc.parallelize(trData.tolist(), numPartition)
tsRDDs = sc.parallelize(tsData.tolist(), numPartition)

trRDDs.cache()
tsRDDs.cache()

Knn = KNN(n_neighbors = K).fit(trRDDs.collect(), trLabel)
Knn = sc.broadcast(Knn)

results = tsRDDs.map(lambda x:Knn.value.predict(np.array(x).reshape(1,-1)))
results = results.collect()

prediction = [int(x[0]) for x in results]
real = [int(x[0]) for x in tsLabel]

accuracy = accuracy_score(real, prediction)
f1score = f1_score(real, prediction, average='macro')
f = open("result.txt",'w')
f.write("accuracy : {:.4f}\n".format(accuracy))
f.write("f1score: {:.4f}".format(f1score))

sc.stop()

end = time.time()

g = open("time.txt",'a')
g.write("single-threading time: {:.4f}\n".format(end-start))
