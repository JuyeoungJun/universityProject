import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from pyspark import SparkConf, SparkContext

numPartition = 100

train = np.loadtxt("train.data", delimiter=',')
test = np.loadtxt("test.data", delimiter=',')
f = open("test.txt","w")
f.write(train)

trX, trY = train[:,:-1], train[:,-1]
tsX, tsY = test[:,:-1], test[:,-1]

conf = SparkConf()
sc = SparkContext(conf=conf)

trRDDs = sc.parallelize(trX.tolist(), numPartition)
tsRDDs = sc.parallelize(tsX.tolist(), numPartition)

trRDDs.cache()
tsRDDs.cache()

Linear = SVC(kernel="linear")
Kernel = SVC(kernel="rbf")

Linear.fit(trRDDs.collect(), trY)
Kernel.fit(trRDDs.collect(), trY)
Linear = sc.broadcast(Linear)
Kernel = sc.broadcast(Kernel)

Linear_result = tsRDDs.map(lambda x:Linear.value.predict(np.array(x).reshape(1,-1)))
Kernel_result = tsRDDs.map(lambda x:Kernel.value.predict(np.array(x).reshape(1,-1)))
Linear_result = Linear_result.collect()
Kernel_result = Kernel_result.collect()

Linear_pred = [int(x[0]) for x in Linear_result]
Kernel_pred = [int(x[0]) for x in Kernel_result]

Linear_acc = accuracy_score(tsY.astype(np.int).tolist(), Linear_pred)
Kernel_acc = accuracy_score(tsY.astype(np.int).tolist(), Kernel_pred)
Linear_f1 = f1_score(tsY.astype(np.int).tolist(), Linear_pred, average='macro')
Kernel_f1 = f1_score(tsY.astype(np.int).tolist(), Kernel_pred, average='macro')
Linear_tn, Linear_fp, Linear_fn, Linear_tp = confusion_matrix(
    Linear_pred, tsY.astype(np.int).tolist()).ravel()
Kernel_tn, Kernel_fp, Kernel_fn, Kernel_tp = confusion_matrix(
    Kernel_pred, tsY.astype(np.int).tolist()).ravel()

f = open("result.txt","w")
f.write("Linear ACC: {:.4f}, Kernel ACC: {:.4f}\n".format(Linear_acc,Kernel_acc))
f.write("Linear F1score: {:.4f}, Kernel F1score: {:.4f}\n".format(Linear_f1,Kernel_f1))
f.write("Linear Confusion\n")
f.write("{} {}\n".format(Linear_tn, Linear_fp))
f.write("{} {}\n".format(Linear_fn, Linear_tp))
f.write("Kernel Confusion\n")
f.write("{} {}\n".format(Kernel_tn, Kernel_fp))
f.write("{} {}\n".format(Kernel_fn, Kernel_tp))
f.close()

sc.stop()
