library(e1071)
d.wine <- read.csv("wine.csv", header=TRUE) #read data from csv file
head(d.wine)
d.wine$wine <-as.factor(d.wine$wine)
#fix seed
set.seed(1)
#devide test set and training set
ran <- sample(1:nrow(d.wine), 0.7*nrow(d.wine))
d.wine_train <- d.wine[ran,]
d.wine_test <- d.wine[-ran,]
#make model
model <- naiveBayes(wine ~ ., data = d.wine_train)
#get training error rate
m.pre <- predict(model, d.wine_train)
train_err = 1-mean(m.pre == d.wine_train$wine)
#get test error rate
m.pre <- predict(model, d.wine_test)
test_err = 1-mean(m.pre == d.wine_test$wine)

cat("training error = ",train_err,"test error = ",test_err)
