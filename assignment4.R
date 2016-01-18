# Name - Sanket Prabhu , net-id - srp140430
.libPaths()

library(rpart)
library(e1071)
library(class)
library(neuralnet)
library(MASS)
library(adabag)
require(adabag)
library(randomForest)

args <- commandArgs(TRUE)
dataURL<-as.character(args[1])
header<-as.logical(args[2])
classIndex <- as.integer(args[3])

d<-read.csv(dataURL,header = header)

datasets <- c ('http://www.utdallas.edu/~axn112530/cs6375/creditset.csv','http://www.ats.ucla.edu/stat/data/binary.csv','http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data','http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data','http://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
)
getDataset <- function(url){
  i<-1
  for(dataset in datasets){
    if(grepl(url,dataset)){
      return(i)
    }
    i <- i+1
  }  
}
#header <- F
#classIndex <-2
#dataURL <-datasets[[4]]
datasetIndex <- getDataset(dataURL)
#d<-read.csv(dataURL,header = header)
#d<-na.omit(d)


if(dataURL == "http://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data" ) {
  d[[35]] <- ifelse(d[[35]] == "g", 0 , 1)
}

if(dataURL == "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data" ) {
  d[[2]] <- ifelse(d[[2]] == "M", 0 , 1)
}

if(dataURL == "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data" ) {
  d[[2]] <- ifelse(d[[2]] == "N", 0 , 1)
  d[[35]] <- ifelse(d[[35]] == "?",0,d[[35]])
}


print <- function(accuracy,method){
  cat("Method = ", method,", accuracy= ", accuracy,"\n")
}


# Decisison Tree
decisionTree <- function(trainingData,testData,ClassTraining,ClassTest) {
  decisionTreeModel <- rpart(ClassTraining ~., parms = list(split = 'information'),data = trainingData, method = 'class') 
  predictResults <- predict(decisionTreeModel, newdata=testData, type="class")
  accuracy <-sum(ClassTest==predictResults)/length(predictResults)  * 100
  print(accuracy,"Decision Tree")
}

# SVM
SVM <- function(trainingData,testData,ClassTraining,ClassTest){
  SVMModel <- svm(ClassTraining ~.,data=trainingData,type = "C-classification",cost = 100,gamma=1/nrow(trainingData))
  predictResults <- predict(SVMModel,testData)
  accuracy <- sum(ClassTest==predictResults)/length(predictResults)  * 100
  print(accuracy,"SVM")
}

# Naive bayes
 NaiveBayes <- function(trainingData,testData,ClassTraining,ClassTest,datasetIndex){
   parameters <- list(c("clientid","income","age","loan","LTI","default10yr"),c("gre","gpa","rank","admit"),colnames(trainingData),colnames  (trainingData),colnames(trainingData))
   parameter <- parameters[[datasetIndex]]
   colname <- names(trainingData)
   class <- trainingData[,classIndex]
   formula <- as.formula(paste(paste("as.factor(",colname[classIndex],")"), paste(parameter, collapse=" + "), sep=" ~ "))
   nbModel<-naiveBayes(formula ,data= trainingData)
   predictedValue <- predict(nbModel,testData)
   accuracy<-(sum(testData[[classIndex]] == predictedValue))/nrow(testData) *100
   print(accuracy,"Naive Bayes")
}

# KNN
 KNN <- function(trainingData,testData,ClassTraining,ClassTest){
   parameters <- list(colnames(trainingData),colnames(trainingData),colnames(trainingData),colnames(trainingData),colnames(trainingData))
   parameter <- parameters[[datasetIndex]]
   colname <- names(trainingData)
   class <- trainingData[,classIndex]
   formula <- as.formula(paste(paste("as.factor(",colname[classIndex],")"), paste(parameter, collapse=" + "), sep=" ~ "))
   knnModel <- knn(train=trainingData, test=testData, cl=trainingData[[classIndex]], k = 3, prob=TRUE)
   accuracy<-(sum(testData[[classIndex]] == knnModel))/nrow(testData) * 100
   
 print(accuracy,"KNN")
}

# Logistic Regression
LogisticRegression <- function(trainingData,testData,ClassTraining,ClassTest) {
  LogisticRegrssionModel <- glm(ClassTraining ~., data = trainingData, family = "binomial",  control = list(maxit = 50))
  predition<-predict(LogisticRegrssionModel, newdata=testData, type="response")
  threshold=0.65
  prediction<-sapply(predition, FUN=function(x) if (x>threshold) 1 else 0)
  accuracy <- sum(ClassTest==prediction)/length(prediction) *100
  
  print(accuracy,"Logistic Regresion")
}

# Nueral Networks

NueralNetwork <- function(trainingData,testData,ClassTraining,ClassTest,datasetIndex) {
    parameters <- list(c("age","LTI"),colnames(trainingData)[-ClassTraining],colnames(trainingData)[-ClassTraining],colnames(trainingData)[-ClassTraining],colnames(trainingData)[-ClassTraining])
    parameter <- parameters[[datasetIndex]]
    colname <- names(trainingData)
    class <- trainingData[,ClassTraining]
    formula <- as.formula(paste(colname[ClassTraining], paste(parameter, collapse=" + "), sep=" ~ "))
    nnModel <- neuralnet(formula, trainingData, hidden = 2, lifesign = "minimal",linear.output = FALSE, threshold = 0.1)
    temp_test <- subset(testData, select = parameter)
    nnModel.results <- compute(nnModel, temp_test)
    results <- data.frame(actual = testData[[ClassTraining]], prediction = nnModel.results$net.result)
    results$prediction <- round(results$prediction)
    accuracy<- sum (results$prediction== results$actual)/nrow(results) * 100
  
  print(accuracy,"Nueral Networks")
}

# Bagging
Bagging <- function(trainingData,testData,ClassTraining,ClassTest,datasetIndex) {
  parameters <- list(c("clientid","income","age","loan","LTI"),c("gre","gpa","rank","admit"),colnames(trainingData),colnames(trainingData),colnames(trainingData))
  parameter <- parameters[[datasetIndex]]
  colname <- names(trainingData)
  class <- trainingData[,ClassTraining]
  
  formula <- as.formula(paste(colname[ClassTraining], paste(parameter, collapse=" + "), sep=" ~ "))
  trainingData[[ClassTraining]] <- as.factor(trainingData[[ClassTraining]])
  
  baggingModel <- adabag::bagging(formula, data = trainingData, mfinal=1 )
  
  table(baggingModel$class, trainingData[[ClassTest]], dnn = c("Predicted Class", "Observed Class"))
  error<- 1 - sum(baggingModel$class == trainingData[[ClassTest]]) / length(trainingData[[ClassTest]])
 
  predbagging <- predict.bagging(baggingModel, newdata = testData)
  accuracy <-sum(predbagging$class == testData[[ClassTest]]) / length(testData[[ClassTest]]) * 100
  
  print(accuracy,"Bagging")
}

# Boosting
Boosting <- function(trainingData,testData,ClassTraining,ClassTest,datasetIndex) {
  parameters <- list(c("clientid","income","age","loan","LTI"),c("gre","gpa","rank","admit"),colnames(trainingData),colnames(trainingData),colnames(trainingData))
  parameter <- parameters[[datasetIndex]]
  colname <- names(trainingData)
  class <- trainingData[,ClassTraining]
  formula <- as.formula(paste(colname[ClassTraining], paste(parameter, collapse=" + "), sep=" ~ "))
  trainingData[[ClassTraining]] <- as.factor(trainingData[[ClassTraining]])
  boostingModel <- adabag::boosting(formula, data = trainingData, mfinal=10 )
  table(boostingModel$class, trainingData[[ClassTest]], dnn = c("Predicted Class", "Observed Class"))
  predboosting <- predict.boosting(boostingModel, newdata = testData)
  accuracy <- sum(predboosting$class == testData[[ClassTest]]) / length(testData[[ClassTest]]) * 100
  print(accuracy,"Boosting")
}


# Random Forests
RandomForests <- function(trainingData,testData,ClassTraining,ClassTest){
  
  parameters <- list(c("clientid","income","age","loan","LTI"),colnames(trainingData)[-classIndex],colnames(trainingData)[-classIndex],colnames(trainingData)[-classIndex],colnames(trainingData)[-classIndex])
  parameter <- parameters[[datasetIndex]]
  colname <- names(trainingData)
  class <- trainingData[,classIndex]
  formula <- as.formula(paste(colname[classIndex], paste(parameter, collapse=" + "), sep=" ~ "))
  trainingData[[classIndex]] <- as.factor(trainingData[[classIndex]])
  rf <- randomForest(formula, data=trainingData, importance=TRUE, proximity=TRUE)
  prediction<-predict(rf, newdata=testData)
  actual<-testData[[classIndex]]
  accuracy <- sum(prediction == actual) / length(actual) * 100
  print(accuracy,"Random Forest")
}


set.seed(123)
for(i in 1:10) {
  cat("Running sample ",i,"\n")
  sampleInstances<-sample(1:nrow(d),size = 0.9*nrow(d))
  trainingData<-d[sampleInstances,]
  testData<-d[-sampleInstances,]
  
  
  Class<-d[,as.integer(args[3])]
  classIndex <- as.integer(args[3])
  ClassTraining <- trainingData[,as.integer(args[3])]
  ClassTest <-testData[,as.integer(args[3])]
  ClassTest <- testData[,classIndex]
 
 

  decisionTree(trainingData,testData,ClassTraining,ClassTest)
  SVM(trainingData,testData,ClassTraining,ClassTest)
  NaiveBayes(trainingData,testData,ClassTraining,ClassTest,datasetIndex)
  KNN(trainingData,testData,ClassTraining,ClassTest)
  LogisticRegression(trainingData,testData,ClassTraining,ClassTest)
  NueralNetwork(trainingData,testData,classIndex,classIndex,datasetIndex)
  Bagging(trainingData,testData,classIndex,classIndex,datasetIndex)
  Boosting(trainingData,testData,classIndex,classIndex,datasetIndex)
  RandomForests(trainingData,testData,classIndex,classIndex)

}



