from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.regression import RidgeRegressionWithSGD, LassoWithSGD
import math

conf = SparkConf()
conf.set("spark.master", "local")
sc = SparkContext(conf=conf)

def parsePoint(line):
	try:
		values = [float(x) for x in line.replace(',',' ').split(' ')]
		return LabeledPoint(values[0], values[1:])
	except:
		return None

data = sc.textFile("train.data")
trainData = data.map(parsePoint)

data = sc.textFile("test.data")
testData = data.map(parsePoint)
    
# Least Square Regression
model_least = LinearRegressionWithSGD.train(trainData, intercept=True)

valuesAndPreds = testData.map(lambda p: 
                              (p.label, model_least.predict(p.features)))
MSE = valuesAndPreds.map(lambda vp: 
                         (vp[0]-vp[1])**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
test_cnt = valuesAndPreds.count()
least_RMSE_test = math.sqrt(MSE)

valuesAndPreds = trainData.map(lambda p: 
                               (p.label, model_least.predict(p.features)))
MSE = valuesAndPreds.map(lambda vp: 
                         (vp[0]-vp[1])**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
train_cnt = valuesAndPreds.count()
least_RMSE_train = math.sqrt(MSE)

    
# Ridge Regression
model_ridge = RidgeRegressionWithSGD.train(trainData, regParam=0.01, intercept=True)

valuesAndPreds = testData.map(lambda p: 
                              (p.label, model_ridge.predict(p.features)))
MSE = valuesAndPreds.map(lambda vp: 
                         (vp[0]-vp[1])**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
test_cnt = valuesAndPreds.count()
ridge_RMSE_test = math.sqrt(MSE)

valuesAndPreds = trainData.map(lambda p: 
                               (p.label, model_ridge.predict(p.features)))
MSE = valuesAndPreds.map(lambda vp: 
                         (vp[0]-vp[1])**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
train_cnt = valuesAndPreds.count()
ridge_RMSE_train = math.sqrt(MSE)

# Lasso Regression
model_lasso = LassoWithSGD.train(trainData, regParam=0.01, intercept=True)

valuesAndPreds = testData.map(lambda p: 
                              (p.label, model_lasso.predict(p.features)))
MSE = valuesAndPreds.map(lambda vp: 
                         (vp[0]-vp[1])**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
test_cnt = valuesAndPreds.count()
lasso_RMSE_test = math.sqrt(MSE)

valuesAndPreds = trainData.map(lambda p: 
                               (p.label, model_lasso.predict(p.features)))
MSE = valuesAndPreds.map(lambda vp: 
                         (vp[0]-vp[1])**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
train_cnt = valuesAndPreds.count()
lasso_RMSE_train = math.sqrt(MSE)
    
f = open("result.txt","w")
f.write('RMSE train / test\n')
f.write('LEAST {:.4f}, {:.4f}\n'.format(least_RMSE_train, least_RMSE_test))
f.write('RIDGE {:.4f}, {:.4f}\n'.format(ridge_RMSE_train, ridge_RMSE_test))
f.write('LASSO {:.4f}, {:.4f}\n'.format(lasso_RMSE_train, lasso_RMSE_test))

sc.stop()
