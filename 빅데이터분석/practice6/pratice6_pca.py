import numpy as np
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg.distributed import RowMatrix
from sklearn.datasets import fetch_openml

conf = SparkConf()
conf.set("spark.master","local")
sc = SparkContext(conf=conf)

mnist = fetch_openml('mnist_784')

data = mnist.data[:10000]
rdd = sc.parallelize(data.tolist(),300)
rdd.cache()
mat = RowMatrix(rdd)

pc_rdd = mat.computePrincipalComponents(16)
pc = pc_rdd.toArray()
pct = np.transpose(pc)

image_shape = (28,28)
fig,axes = plt.subplots(2, 8, figsize=(15,12),subplot_kw = {'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(pct, axes.ravel())):
   ax.imshow(component.reshape(image_shape), cmap='gray_r')

plt.savefig('result.png')
sc.stop()
