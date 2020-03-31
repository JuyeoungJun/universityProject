library(class)
d.wine <- read.csv("wine.csv", header=TRUE) #read data from csv file
head(d.wine)
d.wine$wine <- as.factor(d.wine$wine)
#fix seed 
set.seed(1)
#devide test set and training set
ran <- sample(1:nrow(d.wine), 0.7*nrow(d.wine))
d.wine_train_y <- d.wine[ran,1]
d.wine_test_y <- d.wine[-ran,1]
d.wine_train <- d.wine_norm[ran,]
d.wine_test <- d.wine_norm[-ran,]
#make model and get training error rate
pre <- knn(d.wine_train,d.wine_train,d.wine_train_y,k=3)
train_err = 1 - mean(pre == d.wine_train_y)
#make model and get test error rate
pre <- knn(d.wine_train,d.wine_test,d.wine_train_y,k=3)
test_err = 1 - mean(pre == d.wine_test_y)

cat("training error = ",train_err,"test error = ",test_err)
