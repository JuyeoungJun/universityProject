library(MASS)
d.wine <- read.csv("wine.csv", header=TRUE) #read data from csv file
head(d.wine)
#fix seed 
set.seed(1)
d.wine$wine <- as.factor(d.wine$wine)
#depart test set and training set
ran <- sample(1:nrow(d.wine), 0.7*nrow(d.wine))
d.wine_train <- d.wine[ran,]
d.wine_test <- d.wine[-ran,]
#make lda model
lda1 <- lda(wine~., d.wine_train)
#get training error rate
pre = predict(lda1,newdata = d.wine_train, type = "response")
train_err = 1 - mean(pre$class == d.wine_train$wine)
#get test error rate
pre = predict(lda1,newdata = d.wine_test, type = "response")
test_err = 1 - mean(pre$class == d.wine_test$wine)

cat("training error = ",train_err,"test error = ",test_err)
