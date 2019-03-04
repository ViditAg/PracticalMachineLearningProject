---
title: "Human Activity Recognition: Predicting Dumbbells exercise actitivity"
author: "Vidit Agrawal"
date: "March 3, 2019"
output: html_document
---
## Executive Summary

Using devices such as Nike FuelBand, and Fitbit a large amount of data about personal activity is available. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. The goal of this project is to perform Human activity recognition to predict "the quality of an activity". The activity we are analysing is Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). We use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.

We use supervised machine learning algorithms like classification tree, random forest and generalized boosting model to perform classification task. We start with training data for this classification, split it randomly in training and validation dataset. For an ensemble, we estimate accuracy for all above mentioned algorithms and select the best performing algorithm. Finally, we show the out of sample error and make predict on final testing dataset.

### Getting data

Before loading the data one must download the train and test data from following sources save them in working directory. 

[Training data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

[Testing data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

Loading the training and test Data
```{r}
training<-read.csv("training.csv",header=TRUE)
testing<-read.csv("testing.csv",header=TRUE)
library(caret)
library(rpart)
library(randomForest)
library(gbm)
```

### Cleaning data

First we check for NA values and removing features with NA values.
```{r}
NAFraction<-apply(training, 2, function(x) sum(is.na(x))/length(x))
NonNA_features<-NAFraction==0
sum(NonNA_features)
training<-training[NonNA_features]
testing<-testing[NonNA_features]
```

Next we check for features with blank values and removing those features.
```{r}
BlankFraction<-apply(training, 2, function(x) sum(x=="")/length(x))
NonBlank_features<-BlankFraction==0
sum(NonBlank_features==0)
training<-training[NonBlank_features]
testing<-testing[NonBlank_features]
```

Let's look how the training data looks now.
```{r}
str(training)
```

We can see that first 7 columns represents the id, names, and timestamps related features. In my point of view these features should be excluded to train the prediction algorithm. We need to focus on values recorded by sensors to make prediction.

```{r}
training<-training[-c(1:7)]
testing<-testing[-c(1:7)]
```

### Analysis

We split the training dataset in training and validation set randomly with 60\%:40\% split. Doing it 10 randomizations we estimate the mean accuracy for the classification algorithm to find the best one.

Model # 1. Classification tree.
```{r}
set.seed(1234)
Accuracy_CART=rep(0,10)
for (i in 1:10){
  inTrain<-createDataPartition(training$classe, p = 0.6,list=FALSE)
  training_train<-training[ inTrain,]
  training_valid<-training[-inTrain,]
  modFit_CART <- rpart(classe ~ ., data=training_train, method="class")
  pred_CART <- predict(modFit_CART, training_valid, type = "class")
  Conf_CART<-confusionMatrix(pred_CART, training_valid$classe)
  Accuracy_CART[i]<-Conf_CART$overall[1]
}
```
Mean accuracy
```{r}
mean(Accuracy_CART)
```

Model # 2. Random Forest. Although we should a tree of size 10 but we check other values as well. 
```{r}
set.seed(1234)
Accuracy_RF=rep(0,10)
for (i in 1:10){
  inTrain<-createDataPartition(training$classe, p = 0.6,list=FALSE)
  training_train<-training[ inTrain,]
  training_valid<-training[-inTrain,]
  modFit_RF <- randomForest(classe ~ ., data=training_train, ntree=10)
  pred_RF <- predict(modFit_RF, training_valid,type='class')
  Conf_RF<-confusionMatrix(pred_RF, training_valid$classe)
  Accuracy_RF[i]<-Conf_RF$overall[1]
}
```
Mean Accuracy
```{r}
mean(Accuracy_RF)
```

Model # 3. Generalized Boosting Model
```{r}
set.seed(1234)
Accuracy_GBM<-rep(0,10)
for (i in 1:10){
  modFit_GBM <- gbm(classe ~ ., data=training_train,n.trees = 10,verbose=FALSE,distribution="multinomial")
  Col_classe<-factor(c('A','B','C','D','E'))
  pred_GBM<- predict(modFit_GBM, training_valid,n.trees=10,type="response")
  pred_GBM<-Col_classe[apply(pred_GBM, 1,function(x) which(max(x)==x))]   
  Conf_GBM<-confusionMatrix(pred_GBM, training_valid$classe)
  Accuracy_GBM[i]<-Conf_GBM$overall[1]
}
```
Mean accuracy
```{r}
mean(Accuracy_GBM)
```

We saw that random forest is the best performing model on the validation data set but we further combine 10 random forest models. Further, passing the data through random forest model to get a final accuracy and out-of-sample error.
```{r}
set.seed(1234)
pred_RF_matrix<-matrix(nrow=length(training_valid[,1]),ncol=10)
for (i in 1:10){
  inTrain<-createDataPartition(training$classe, p = 0.6,list=FALSE)
  training_train<-training[ inTrain,]
  training_valid<-training[-inTrain,]
  modFit_RF <- randomForest(classe ~ ., data=training_train, ntree=10)
  pred_RF_matrix[,i] <- predict(modFit_RF, training_valid,type='class')
}
Pred_combine<-data.frame(pred_RF_matrix,training_valid$classe)
Model_Comb<-train(training_valid.classe~.,method="rf",ntree=10,data=Pred_combine)
Pred_Comb<-predict(Model_Comb,Pred_combine)
Conf_Comb<-confusionMatrix(Pred_Comb, training_valid$classe)
Conf_Comb
```
```{r}
plot(Conf_Comb$table)
```


```{r}
Accuracy_Comb<-Conf_Comb$overall[1]
Accuracy_Comb*100
```

Out-of-sample error
```{r}
Out_ofSample_Error<-(1-Accuracy_Comb)*100
print(Out_ofSample_Error)
```

Finally, making prediction on the testing dataset.
```{r}
set.seed(1234)
pred_RF_test<-matrix(nrow=length(testing[,1]),ncol=10)
inTrain<-createDataPartition(training$classe, p = 0.6,list=FALSE)
training_train<-training[ inTrain,]
modFit_RF <- randomForest(classe ~ ., data=training_train, ntree=10)
pred_RF_test <- predict(modFit_RF, testing,type='class')
```

### Source

The data for this project come from this source: 
