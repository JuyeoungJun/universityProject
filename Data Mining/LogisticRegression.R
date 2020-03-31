d.wine <- read.csv("wine.csv", header=TRUE, sep=",") #read data from csv file
head(d.wine)
d.wine$wine <-as.factor(d.wine$wine)
#fix seed
set.seed(1)
d.wine$wine <- as.factor(d.wine$wine)
#devide test set and training set 
ran <- sample(1:nrow(d.wine), 0.7*nrow(d.wine))
d.wine_train <- d.wine[ran,]
d.wine_test <- d.wine[-ran,]
#make logistic regression model
mylogit <- glm(wine~.,data = d.wine_train,family = binomial)
#get training error rate
pre <- predict(mylogit,newdata = d.wine_test,type= "response")
pred <- rep("A",nrow(d.wine_train))
pred[pre>0.5] = "B"
table(pred,d.wine_train$wine)
train_err = 1-mean(pred == d.wine_train$wine)
#get test error rate
pre <- predict(mylogit,newdata = d.wine_test,type= "response")
pred <- rep("A",nrow(d.wine_test))
pred[pre>0.5] = "B"
table(pred,d.wine_test$wine)
test_err = 1-mean(pred == d.wine_test$wine)
cat("training error = ",train_err,"test error = ",test_err)
