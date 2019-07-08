install.packages("rmarkdown")

setwd("~/Desktop/STA141A/final project")
#learned the way computer works in http://l3d.cs.colorado.edu/courses/CSCI1200-96/binary.html
#read and write binary files in ?readBin
#https://stats.idre.ucla.edu/r/faq/how-can-i-read-binary-data-into-r/
###1.incomplete
load_training_images= function(in_dir = "~/Desktop/STA141A/final project",
                out_file = "~/Desktop/STA141A/final project/training.rds"){
bin1=as.integer(readBin(con="data_batch_1.bin",what="raw",n=3073*10000,size=1,endian="big"))
bin2=as.integer(readBin(con="data_batch_2.bin",what="raw",n=3073*10000,size=1,endian="big"))
bin3=as.integer(readBin(con="data_batch_3.bin",what="raw",n=3073*10000,size=1,endian="big"))
bin4=as.integer(readBin(con="data_batch_4.bin",what="raw",n=3073*10000,size=1,endian="big"))
bin5=as.integer(readBin(con="data_batch_5.bin",what="raw",n=3073*10000,size=1,endian="big"))
train1=matrix(bin1,nrow=10000,ncol=3073,byrow=T)
train2=matrix(bin2,nrow=10000,ncol=3073,byrow=T)
train3=matrix(bin3,nrow=10000,ncol=3073,byrow=T)
train4=matrix(bin4,nrow=10000,ncol=3073,byrow=T)
train5=matrix(bin5,nrow=10000,ncol=3073,byrow=T)

bin_t<-rbind(train1,train2,train3,train4,train5)

saveRDS(bin_t, out_file)
}
load_training_images()

#???directory in function. refer to discussion note 10 by Jiahui Guan

#test

load_testing_images=function(in_dir = "~/Desktop/STA141A/final project",
                                out_file = "~/Desktop/STA141A/final project/testing.rds"){
bin6=as.integer(readBin(con="test_batch.bin",what="raw",n=3073*10000,size=1,endian="big"))
bin_te<-matrix(bin6,ncol=3073,nrow=10000,byrow=T)
saveRDS(bin_te, out_file)
}
load_testing_images()

#2
training=readRDS("training.rds")
testing=readRDS("testing.rds")
data_rescale<-function(labels,k=500)sort(as.vector(sapply(unique(labels),function(i)which(labels==i))[1:k,]))
train2<-training[data_rescale(training[,1],k=500),]
test2<-testing[data_rescale(testing[,1],k=100),]

train2<-saveRDS(train2,"train2.rds")
test2<-saveRDS(test2,"test2.rds")
#getting the class names for image function.
dat.names <- readLines("batches.meta.txt", n = 10)

test2<-readRDS("test2.rds")
train2<-readRDS("train2.rds")
view_image=function(x){
  library(grid)
  img <-train2[x,-1]
  r <- matrix(img[1:1024], ncol=32, byrow = TRUE)
  g <- matrix(img[1025:2048], ncol=32, byrow = TRUE)
  b <- matrix(img[2049:3072], ncol=32, byrow = TRUE)
  img_col <- rgb(r,g,b, maxColorValue = 255)
  dim(img_col) <- dim(r)
  grid.raster(img_col, interpolate=FALSE)
  classnum<-train2[x,1]
  print(dat.names[classnum+1])
}

##3.
train2=as.data.frame(train2)
train2<-train2[order(train2$V1),]
set.seed(6969)
view_image(sample(1:500,size=1))
dev.copy2pdf()
#random select images in each class group. Save current plots, refer to https://stackoverflow.com/questions/26034177/r-saving-multiple-ggplots-using-a-for-loop/26078489?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
for(i in 1:9){
index=sample((i*500+1):((i+1)*500),size=1)
file_name = paste("plot_", i, ".pdf", sep="")
pdf(file_name)
view_image(index)
dev.off()
}
#extract the data for three channels that are in 1024 pixels each.
#red
red<-train2[,2:1025]
names(red)<-as.character(c(1:1024))
pix<-function(x){c((x%/%32)+1,x%%32)}#determine the pixel location.
sapply(as.integer(names(tail(sort(sapply(red,sd)),1))),pix)#larger sd() values are in tail.
sapply(as.integer(names(head(sort(sapply(red,sd)),1))),pix)

#green
green<-train2[,1026:2049]
names(green)<-as.character(c(1:1024))
sapply(as.integer(names(tail(sort(sapply(green,sd)),1))),pix)
sapply(as.integer(names(head(sort(sapply(green,sd)),1))),pix)
#blue
blue<-train2[,2050:3073]
names(blue)<-as.character(c(1:1024))
sapply(as.integer(names(tail(sort(sapply(blue,sd)),1))),pix)
sapply(as.integer(names(head(sort(sapply(blue,sd)),1))),pix)
#4.
#change test2 and train2 back to matrix format.
test2<-readRDS("test2.rds")
train2<-readRDS("train2.rds")
library(parallelDist)

########distance function.Consume about 20mins.
dist_mat<-parDist(rbind(train2,test2)[,-1],method="euclidean")
dist_mat<-as.matrix(dist_mat1)
##########
#change row and column names to observation classes.
rownames(dist_mat) <- c(train2[,1], test2[,1])
colnames(dist_mat) <- c(train2[,1], test2[,1])
saveRDS(dist_mat,"dist_mat.rds")
dist_mat<-readRDS("dist_mat.rds")
#select test observations as prediction points.
predict_knn <- function(test_index=1:1000,train_index=1:5000,dist_mat1=dist_mat[5001:6000,1:5000],k=5){
  c(sapply(test_index, function(x) names(which.max(table(names(sort(dist_mat1[x,train_index])[1:k]))))))
}

#5
labels = names(dist_mat[,1])
cv_error_knn = function(dist_mat1 = dist_mat[1:5000,1:5000],k){
  mse<- rep(0,10)
  for (i in 0:9) {
    mse[i+1] <- sum(predict_knn(test_index = (500*i+1):(500*(i+1)), train_index = -((500*i+1):(500*(i+1))), dist_mat1,k) != labels[(500 * i+1):(500*(i+1))]) / 500#10-fold
  }
  return(mean(mse))
}

#6
###euclidean method
cvknn<-sapply(1:15, function(i) cv_error_knn(k=i))
cvknn
###method "manhattan"
dist_matm<-as.matrix(dist(rbind(train2,test2)[,-1],method="manhattan"))
##### 20mins consumption.
rownames(dist_matm) <- c(train2[,1], test2[,1])
colnames(dist_matm) <- c(train2[,1], test2[,1])
saveRDS(dist_matm,"dist_matm.rds")
dist_matm<-readRDS("dist_matm.rds")
labels=names(dist_matm[,1])
cvknnm<-sapply(1:15, function(i) cv_error_knn(dist_mat1=dist_matm[1:5000,1:5000],k=i))
cvknnm
#start plot two lines on ggplot2. https://stackoverflow.com/questions/3777174/plotting-two-variables-as-lines-using-ggplot2-on-the-same-graph?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
cv6<-as.data.frame(cbind(c(1:15),cvknn,cvknnm))
library(ggplot2)
ggplot(aes(x=V1),data=cv6)+
  geom_line(aes(y=cvknn,colour="Euclidean"))+
  geom_line(aes(y=cvknnm,colour="Manhattan"))+
  geom_point(aes(y=cvknn))+
  geom_point(aes(y=cvknnm))+
  labs(x="k",y="CV error rate")
order(cvknnm)
#7. confusion matrix. refer to https://www.youtube.com/watch?v=FAr2GmWNbT0
#prefer Manhattan method to Euclidean refers to Q6 plot.
head(order(cvknnm), 3)#best three k by CV error rate.

con<- function(dist_mat1 = dist_matm[1:5000,1:5000] , k){
  pred<- c()
for (i in 0:9) {
pred_i = predict_knn(test_index = (500*i+1):(500*(i+1)), train_index = -((500*i+1):(500*(i+1))), dist_mat1,k)
pred<- c(pred,pred_i)
}
  pred<<-pred #predictions are accessible from outside.
}
library("gridExtra")
library("grid")
#when k=12
con(k=12)
test_l<-names(dist_matm[1:5000,1])
#grid the confusion matrix.
pred1<-pred
grid.table(table(pred1,test_l))
ar12<-sum(diag(table(pred1,test_l))) / 5000#accuracy rate
ar12
#k=13
con(k=13)
pred2<-pred
grid.table(table(pred2,test_l))
ar13<-sum(diag(table(pred2,test_l))) / 5000
ar13
#k=14
con(k=14)
pred3<-pred
grid.table(table(pred3,test_l))
ar14<-sum(diag(table(pred3,test_l))) / 5000
ar14
#8.Found in the Q7 that when k=12 with Manhattan method is the best combination overall.
miscla<-rep(0,10)
tab<-as.matrix(table(pred1,test_l))

for(i in 0:9){
  miscla[i+1]<-sum(tab[i+1,])-tab[i+1,i+1]
}
names(miscla)<-dat.names
sort(miscla)
#9.
#By Euclidean method
#writing a function for the test error calculation.
test_error_knn= function(dist_mat1,k){
  test_lab<- names(dist_mat[5001:6000,1])
sum(test_lab != predict_knn(test_index=1:1000,train_index=1:5000,dist_mat1,k)) / 1000
}

test_error_eud<-sapply(1:15,function(i) test_error_knn(dist_mat1=dist_mat[5001:6000,1:5000],k=i))
test_error_eud
#Manhattan method
test_error_man<-sapply(1:15,function(i) test_error_knn(dist_mat1=dist_matm[5001:6000,1:5000],k=i))
test_error_man
#plot the same thing like Q6
tt9<-as.data.frame(cbind(c(1:15),test_error_eud,test_error_man))
ggplot(aes(x=V1),data=tt9)+
  geom_line(aes(y=test_error_eud,colour="Euclidean"))+
  geom_line(aes(y=test_error_man,colour="Manhattan"))+
  geom_point(aes(y=test_error_eud))+
  geom_point(aes(y=test_error_man))+
  labs(x="k",y="test error rate")
sort(test_error_man)
sort(cvknnm)
