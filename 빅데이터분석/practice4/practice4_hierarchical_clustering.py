import numpy as np
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import BisectingKMeans as BSK
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

from sklearn.metrics.cluster import normalized_mutual_info_score as NMI

nTrain = 1500

def sort_by_target(digits):
	try:
		Data = digits[:,:-1]
		Target = digits[:,-1]

		reorder_train = np.array(sorted([(target, i) for i, target 
                                   in enumerate(Target[:nTrain])]))[:,1]
		reorder_test = np.array(sorted([(target, i) for i, target 
                                  in enumerate(Target[nTrain:])]))[:,1]
		Data[:nTrain] = Data[reorder_train.astype(np.int64).tolist()]
		Target[:nTrain] = Target[reorder_train.astype(np.int64).tolist()]
		Data[nTrain:] = Data[(reorder_test + nTrain).astype(np.int64).tolist()]
		Target[nTrain:] = Target[(reorder_test + nTrain).astype(np.int64).tolist()]

		digits = np.concatenate((Data,Target.reshape(-1,1)), axis = 1)

		return digits[:nTrain], digits[nTrain:]
	except:
		return None

def parsePoint(line):
	return [int(x) for x in line.split(',')]

conf = SparkConf()
conf.set("spark.master","local")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

data = sc.textFile("practice5.data")
data_label = data.map(parsePoint)
data_label = np.array(data_label.collect())

trainData, testData = sort_by_target(data_label)
trainData = map(lambda x: (int(x[-1]), Vectors.dense(x[:-1])), trainData)
testData = map(lambda x: (int(x[-1]), Vectors.dense(x[:-1])), testData)

trainData = sqlContext.createDataFrame(trainData, schema=["label","features"])
trFeat = trainData.select([c for c in trainData.columns if c in ["features"]])
trLab = trainData.select([c for c in trainData.columns if c in ["label"]])

testData = sqlContext.createDataFrame(testData, schema=["label","features"])
tsFeat = testData.select([c for c in testData.columns if c in ["features"]])
tsLab = testData.select([c for c in testData.columns if c in ["label"]])

bkm = BSK(k=10, minDivisibleClusterSize=1.0)
model = bkm.fit(trFeat)

predict = model.transform(tsFeat).select("prediction")
predict = predict.rdd.flatMap(lambda x: x).collect()

Label = [int(row['label']) for row in tsLab.collect()]

f = open('result.txt','w')
f.write('NMI of hierarchical clustering\n')
f.write('{:.4f}'.format(NMI(Label,predict)))

sc.stop()
