library(neuralnet)
d.wine <- read.csv("wine.csv", header=TRUE) #read data from csv file
head(d.wine)
d.wine$wine <-as.factor(d.wine$wine)
#fix seed
set.seed(1)
#depart test set and training set
ran <- sample(1:nrow(d.wine), 0.7*nrow(d.wine))
d.wine_train <- d.wine[ran,]
d.wine_test <- d.wine[-ran,]
#make nueral network model
nn <- neuralnet(wine~.,data = d.wine_train, hidden=c(3,3,3), linear.output = FALSE, threshold=0.01)
#predict test set 
pre <- predict(nn,d.wine_train, type = "class")
#get training error rate 
get <- matrix(1:length(pre)/2,nrow = length(pre)/2, ncol = 1)
for(i in 1:(length(pre)/2)){
  if(pre[i,1] > pre[i,2]) {
    get[i] = 'A'
  }
  else {
    get[i] = 'B'
  }
}
train_err = 1-mean(get == d.wine_train$wine) 
#get test error rate 
pre <- predict(nn,d.wine_test,type = "class")
get <- matrix(1:length(pre)/2,nrow = length(pre)/2, ncol = 1)
for(i in 1:(length(pre)/2)){
  if(pre[i,1] > pre[i,2]) {
    get[i] = 'A'
  }
  else {
    get[i] = 'B'
  }
}
test_err = 1-mean(get == d.wine_test$wine)
cat("training error = ",train_err, "test error = ",test_err)


