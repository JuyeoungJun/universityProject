#read csv file
do.pain<-read.csv("pain_relief.csv",header = TRUE)
#calculate distance 
dt.dis<-dist(do.pain, method = "euclidean")
#make a cluster
hc.pain<-hclust(dt.dis,"ward.D2")
#set seed
set.seed(100)
plot(hc.pain)
clust.member<-cutree(hc.pain, k = 3)
clust.member
do.pain$clust <- clust.member
#make a kmeans
kmeans.member<-kmeans(do.pain, 3)
kmeans.member
#compare k means and hcluster
diff <- 0
for(i in 1:100){
  if(kmeans.member$cluster[i] == clust.member[i])
  {
    print("TRUE")
    
  }
  else{
    print("FALSE")
    diff <- diff + 1
  }
}

different <- diff/100
different

#make a plot
library("cluster")
do.pain
boxplot(no.stomach.upset~clust,data=do.pain,main="no.stomach.upset",xlab="cluster")
boxplot(no.side.effect~clust,data=do.pain,main="no.side.effect",xlab="cluster")
boxplot(stop.pain~clust,data=do.pain,main="stop.pain",xlab="cluster")
boxplot(work.quickly~clust,data=do.pain,main="work.quickly",xlab="cluster")
boxplot(keep.me.awake~clust,data=do.pain,main="keep.me.awake",xlab="cluster")
boxplot(limited.relief~clust,data=do.pain,main="limited.relief",xlab="cluster")
#summary of cluster
summary(subset(do.pain,do.pain$clust == 1))
summary(subset(do.pain,do.pain$clust == 2))
summary(subset(do.pain,do.pain$clust == 3))


