from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
import math

def parseFeat(line):
    values = [float(x) for x in line.replace(',',' ').split(' ')]
    return values[:-1]

def parseLabel(line):
    values = [float(x) for x in line.replace(',',' ').split(' ')]
    return values[-1]

def error(point, model):
    center = model.centers[model.predict(point)]
    return math.sqrt(sum([x**2 for x in (point-center)]))

conf = SparkConf()
conf.set("spark.master", "local")
sc = SparkContext(conf=conf)

data = sc.textFile("practice6_train.csv")
trData = data.map(parseFeat)

data = sc.textFile("practice6_test.csv")
tsData = data.map(parseFeat)
tsLabel = data.map(parseLabel)

kmeans_list = []
for i in range(30):
    kmeans_list.append(KMeans.train(trData,k=10,maxIterations=100, seed=i))

obj_list = []
for i in range(30):
    obj_list.append(trData.map(
        lambda point: error(point, kmeans_list[i])).reduce(lambda x,y: x+y))

kmeans = kmeans_list[obj_list.index(min(obj_list))]
tsPredict = kmeans.predict(tsData)

nmi_score = NMI(list(tsPredict.collect()), list(tsLabel.collect()))

f = open('result.txt','w')
f.write('NMI of K-Means clustering\n')
f.write('{:.4f}'.format(nmi_score))

sc.stop()
