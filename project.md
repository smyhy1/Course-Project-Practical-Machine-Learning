---
title: "Predicting quality of exercises using machine learning techniques"
author: "Haobo Yuan"
date: "June 28, 2020"
output: html_document
---
## Synopsis 
This is the final assignment for the Practical Machine Learning class on Coursera.  The assignment is to use the techniques presented in the course to build the best machine learning model that we can on the assigned data, and then produce a prediction on a quiz data set.  This report begins with a description of the data, proceeds to data processing and exploratory data analysis.  From there it details the steps taken to build the best machine learning model, and finally presents the results on the validation set.  The final model achieves over 98% accuracy in out-of-sample validation, an amazingly high value.

## The Data  
The data for this analysis consist of a set of observations taken from motion sensors on subjects as they perform a weightlifting exercise.  The motion sensors are attached to the subjects' belt, arm, forearm, and dumbbell.  Several different readings were from the motion sensors as the subjects performed the exercise either correctly or incorrectly in one of four ways.  The manner in which the subject intended to perform the exercise is a factor variable.  The final components of the data set are identifying characteristics of the subject, time stamps, etc, as well as summary statistics from the motion sensors.  
As described in *Exploratory Analysis and Data Processing* below, the data consist of two different types of observations: periodic observations during the movement, and summaries of observations on the motion components captured over the entire movement.

More information on the data and the study that produced it can be found [here](http://groupware.les.inf.puc-rio.br/har).  See the section on the *Weight Lifting Exercises Dataset*.  

## Loading the Data and Packages
The training data provided for the assignment is available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) and the test data is available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv). To download the data and read it into R:  
```{r cache=TRUE}
url1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
if (!file.exists("./data")) dir.create("./data")
download.file(url1, "./data/pml-training.csv")
download.file(url2, "./data/pml-testing.csv")
dld <- date()
pmlTrain <- read.csv("./data/pml-training.csv",na.strings=c("NA",""), comment.char = "")
pmlTest <- read.csv("./data/pml-testing.csv",na.strings=c("NA",""),comment.char = "")
```  
The data used here were downloaded on `r dld`  

This analysis relies on a relatively large number of R packages: caret for training models; rpart, glmnet, gbm, and randomForest for modeling methods; parallel and doParallel for processing with multiple cores; plyr, dplyr and tidyr for data manipulation; and ggplot2 and lattice for graphics.  Several of these packages also load dependencies:  
```{r}
suppressMessages(library(caret, warn.conflicts=FALSE, quietly=TRUE,verbose=FALSE))
suppressMessages(library(rpart, warn.conflicts=FALSE, quietly=TRUE,verbose=FALSE))
suppressMessages(library(glmnet, warn.conflicts=FALSE, quietly=TRUE,verbose=FALSE))
suppressMessages(library(gbm, warn.conflicts=FALSE, quietly=TRUE,verbose=FALSE))
suppressMessages(library(randomForest, warn.conflicts=FALSE, quietly=TRUE,verbose=FALSE))
suppressMessages(library(parallel, warn.conflicts=FALSE, quietly=TRUE,verbose=FALSE))
suppressMessages(library(doParallel, warn.conflicts=FALSE, quietly=TRUE,verbose=FALSE))
suppressMessages(library(plyr, warn.conflicts=FALSE, quietly=TRUE,verbose=FALSE))
suppressMessages(library(dplyr, warn.conflicts=FALSE, quietly=TRUE,verbose=FALSE))
suppressMessages(library(tidyr, warn.conflicts=FALSE, quietly=TRUE,verbose=FALSE))
suppressMessages(library(ggplot2, warn.conflicts=FALSE, quietly=TRUE,verbose=FALSE))
suppressMessages(library(lattice, warn.conflicts=FALSE, quietly=TRUE,verbose=FALSE))
```
  
## Exploratory Analysis and Data Processing
### Initial Exploration  
The "training" dataset is very large--`r dim(pmlTrain)[1]` observations on `r dim(pmlTrain)[2]` different variables.  This means care must be taken with the computational load when training models. By contrast, the "testing" dataset is very small, particularly in comparison to the "training" set--only `r dim(pmlTest)[1]` observations on `r dim(pmlTest)[2]` variables.  The variables in each set do not overlap perfectly:  
```{r}
trainOnly <- names(pmlTrain)[!(names(pmlTrain) %in% names(pmlTest))]
testOnly<- names(pmlTest)[!(names(pmlTest) %in% names(pmlTrain))]
```
The variable `r trainOnly` only appears in the "training" set, which is the response we are looking to predict.  It is replaced in the "testing" set with `r testOnly`.  The size contrast and the lack of the response variable in the "testing" set means that the "training" set should be treated as the entire dataset, partitioned into train, test, and validation.  The "testing" set should be treated as the quiz and processed in the same way as the "training" data, but otherwise left alone until final prediction.  

The dataset is full of missing values. Fully `r round((sum(is.na(pmlTrain))/(prod(dim(pmlTrain))))*100,1)`% of the values are missing.  Interestingly, those missing values are clustered by predictor--predictors either have a large number of missing values, or none at all.  
```{r}
naCount <- apply(pmlTrain, 2, FUN=function(x){sum(is.na(x))})
naTbl <- plyr::count(naCount)
```
`r naTbl[1,2]` of the variables have `r naTbl[1,1]` missing values. The remaining `r naTbl[2,2]` have `r naTbl[2,1]` missing values--`r round(naTbl[2,1]/dim(pmlTrain)[1]*100,1)`% of the values in those columns.  Looking at a subset (for brevity) the variables which are and are not mostly missing sheds some light on what is going on.  
No values missing:
```{r}
print(head(names(naCount[naCount==0]),20))
```
Most values missing:
```{r}
print(head(names(naCount[naCount>0]),20))
```
The key difference is that variables with no missing values are either identifying (user\_name etc) or direct observations from the motion sensors, while variables with missing values are summaries (kurtosis, skewness, avg, etc).  This means there are two fundamentally different types of observations in the data--direct and summary.  The variable new\_window identifies which is which:
```{r}
print(unique(pmlTrain$new_window[is.na(pmlTrain$skewness_yaw_arm)]))
print(unique(pmlTrain$new_window[!is.na(pmlTrain$skewness_yaw_arm)]))
```
Even though it is not explicitly described in the data documentation, this leads to the conclusion that each excersize in the data was recorded over a period of time which contained several signals from each of the motion sensors.  These observations are grouped by the num\_window variable.  Summary statistics were then computed at the end of that recording window, and are identified with new\_window = "yes".  

### Data Processing
Given the two fundamentally different types of observations in the data, a decision needs to be made on whether to (1) include only the direct observations, (2) include only the summary observations, or (3) include both.  Since the ultimate goal of the assignment is to predict the quiz dataset, the makeup of that dataset drives the decision:
```{r}
print(unique(pmlTest$new_window))
```
The quiz dataset contains only direct observations, so only direct observations will be included for training and evaluation models.  This has the additional benefit of allowing the removal of the variables with mostly missing values, reducing computational complexity.  Purely descriptive variables should be removed as well since a model trained on, say, who the subject was might bot be useful.  Removing the unwanted observations and variables from both data sets:
```{r}
pmlTrain <- pmlTrain[as.character(pmlTrain$new_window) != "yes",]
pmlTest <- pmlTest[as.character(pmlTest$new_window) != "yes",]
naCount2<- apply(pmlTest, 2, FUN=function(x){sum(is.na(x))})
pmlTrain <- pmlTrain[,naCount==0]
pmlTest <- pmlTest[,naCount2==0]
pmlTrain <- pmlTrain[,-(1:7)]
pmlTest <- pmlTest[,-(1:7)]
```
### Further Exploration  
Now that the dataset is more manageable, further exploratory analysis will inform model selection.  Exercise classification is multinomial, and exercise type "A" (correct form) is most common:
```{r fig.width=12}
qplot(pmlTrain$classe, fill=pmlTrain$classe, 
      xlab = "Exercise Classification", main = "Histogram of exercise class")
```
Mutinomial response limits the options available for machine learning model selection, since many either only handle binomial classification or are too computationally intensive for the available resources.

Plotting the predictors by type of signal, sensor, and direction against exercise classcan give a picture of the overall data in a single plot.  This requires reshaping the data:
```{r fig.width=12, cache=TRUE}
pmlTrainPlot <- gather(pmlTrain, reading, signal, -classe)
pmlTrainPlot$reading <- as.character(pmlTrainPlot$reading)
pmlTrainPlot <- mutate(
    pmlTrainPlot, direction="total/complex", sensor=NA, device=NA)
ind <- grepl("^.+_[xyz]",pmlTrainPlot$reading)
pmlTrainPlot$direction[ind] <- gsub(
    "(^.+_)([xyz])","\\2",
    pmlTrainPlot$reading[ind])
pmlTrainPlot$sensor <- gsub(
    "(^.*_)(forearm|belt|arm|dumbbell)(.*$)",
    "\\2",pmlTrainPlot$reading)
pmlTrainPlot$device <- gsub(
    "(^.*_*)(pitch|roll|yaw|accel|gyros|magnet)(.*$)",
    "\\2",pmlTrainPlot$reading)
mp <- ggplot(pmlTrainPlot, aes(x=device,y=signal, fill=classe))
mp <- mp + geom_boxplot() + ggtitle("Boxplot of signal by device and location")
mp <- mp + facet_grid(direction~sensor) + ylim(-800,800)
suppressWarnings(print(mp))
```  
Based on this plot, no obvious pattern emerges to aid in model selection.  However the wide variation across variables means that it is worth exploring centering and scaling the data during model training.  

## Model Selection  
### Data Partition and Validation  
Given the large number of observations in the data, 60% are used for training, 20% for testing, and 20% for validating.  The training data are used for training the models, the testing data for selecting and fine-tuning the models, and the validation data for final model evaluation, as suggested in the class.  The models themselves are trained using 10-fold cross validation to balance bias, variance, and computation time needed for training.  
```{r}
set.seed(3633)
inTrain <- createDataPartition(pmlTrain$classe, p=.8, list=FALSE)
pmlTrainTrain<-pmlTrain[inTrain,]
pmlTrainValidate<-pmlTrain[-inTrain,]
inTrain2 <- createDataPartition(pmlTrainTrain$classe, p=.75, list=FALSE)
pmlTrainTest <- pmlTrainTrain[-inTrain2,]
pmlTrainTrain<-pmlTrainTrain[inTrain2,]
myControl <- trainControl(method="cv", number=10)
```
### Methodology  
This analysis first evaluates data pre-processing uptions using a simple classifcation tree model.  Based on those results basic tree (rpart) and GLM (glmnet) models are evaluated, followed by bagged (GBM) and boosted (random forest) tree models.  Bagged or boosted linear models were not possible due to computation poer constraings.  These models are compared based on cross-validation in the trainig data set.  The best two are compared on the test data set.  The best of those is then fine-tuned if possible, and finally out-of-sample predictions are evaluated against the validation data set.  

### Evaluating pre-processing  
Pre-processing is tested by training a basic classification tree in four different ways: with data as-is, pre-processed with pricinpal component analysis, pre-processed with scaling, and pre-processed with centering and scaling.  The same seed is used before training each model in order to restrict the differences to pre-processing.  The models are then compared based on internal cross-validation to select the best approach to pre-processing
```{r fig.width=12, cache=TRUE}
set.seed(1415)
baseTree <- train(classe~., method="rpart", 
                  data=pmlTrainTrain, trControl = myControl)
set.seed(1415)
pcaTree <- train(classe~., method="rpart", preProcess="pca",
                  data=pmlTrainTrain, trControl = myControl)
set.seed(1415)
STree <- train(classe~., method="rpart", preProcess="scale",
                  data=pmlTrainTrain, trControl = myControl)
set.seed(1415)
CSTree <- train(classe~., method="rpart", preProcess=c("center","scale"),
                  data=pmlTrainTrain, trControl = myControl)
myRes <- resamples(list(as.is=baseTree, pca=pcaTree, s.only=STree, c.and.s=CSTree))
bwplot(myRes)
```  
The above plot shows the estimated out-of-sample accuracy and kappa for using internal cross-validation for each of the 10 different folds of the training data.  This shows almost no impact for scaling or centering and scaling, and markedly worse performance for principal component analysis.  Based on these results, the data will not be pre-processed in the following steps.  

### Evaluating model types
Training GLM, GBM, and Random Forest models to add to the Classifcation Tree trained above:
```{r fig.width=12, cache=TRUE}
set.seed(1417)
myControl <- trainControl(method="cv", number=10, allowParallel = TRUE)
myCluster <- makeCluster(detectCores()-1)
registerDoParallel(myCluster)
baseGLM <- train(classe~., method="glmnet", family="multinomial",
                 data=pmlTrainTrain, trControl = myControl)
baseGBM <- train(classe~., method="gbm", data=pmlTrainTrain,
                 verbose=FALSE, trControl = myControl)
baseRF <- train(classe~., method="rf", data=pmlTrainTrain, 
                trControl = myControl, prox=TRUE)
myRes2 <- resamples(list(Tree=baseTree, GLM=baseGLM, 
                        GBM=baseGBM, Random.Forest = baseRF))
bwplot(myRes2)
```  
Based on these results, the Random Forest and GBM models are clear winners, both with estimated out-of-sample accuracy of over 90%.  The results are so high that over-fitting may be a problem.  Evaluating those two on the test (remember, 20% of the original training) data, we have the following:

```{r}
confusionMatrix(predict(baseGBM, pmlTrainTest),pmlTrainTest$classe)
confusionMatrix(predict(baseRF, pmlTrainTest),pmlTrainTest$classe)
```
The random forest model continues to be the most accurate, even on the test set.  

### Fine-tuning the final model
It is computationally intensive, however, so it is worth checking a pared-down version.  The data is too large to use any of the caret package's automatic feature selection algorithms.  The Variable Importance measure shows which variables have the largest impact on the model, with an importance from 0-100.  The next version of the model only includes those variables with importance greater than 10 from the previous random forest model:  
```{r cache=TRUE}
set.seed(3400)
vi <- varImp(baseRF)
viData <- cbind.data.frame(names = row.names(vi$importance),
                           imp = vi$importance$Overall)
viData$names <- as.character(viData$names)
keepVars <- viData$names[viData$imp >= 10]
keepVars <- c(keepVars,"classe")
impRF <- train(classe~., method="rf", 
               data=pmlTrainTrain[,keepVars], 
                trControl = myControl, prox=TRUE)
stopCluster(myCluster)
```  
In comparison to the base Random Forest model:
```{r fig.width=12}
myRes3 <- resamples(list(plain.Random.Forest=baseRF, Trimmed.Random.Forest=impRF))
bwplot(myRes3)
```
So far, the trimmed-down random forest model looks promising based on estimates of out-of-sample accuracy from cross-validation.  On the test set:
```{r}
fintestCM <- confusionMatrix(predict(impRF, pmlTrainTest),pmlTrainTest$classe)
print(fintestCM)
```  
The performance of this model is nearly as good as the initial random forest, with approximately $1/5$ fewer predictors.  This trimmed random forest model is the final model for this analysis.  

## Evaluation of the final model  
### Interpretability  
As a random forest model, the final model has almost no practical intepretability.  While this may be a drawback for many types of problems, it is acceptable in this context.  The likely use of such a model is to give a user feedback on whether they performed the exercise correctly or, if not, which common mistake they made as corresponds to the predicted class.  Trading model accuracy for interpretability would not help this goal.  

### Out-of-sample error estimate  
Out-of-sample error rates were estimated in three different ways--during the training itself using 10-fold cross validation and the internal random forest out-of-bag error estimate, as well as on the test data prior to final evaluation on the validation set.  The cross-validation estimate ranged from `r round(range(1-impRF$resample$Accuracy)[1],4)` to `r round(range(1-impRF$resample$Accuracy)[2],4)`.  The out-of-bag error rate estimate was  0.0086.  The test estimate error rate was `r round(1-fintestCM$overall[[1]],4)`, with an estimated range from `r round(1-fintestCM$overall[[4]],4)` to `r round(1-fintestCM$overall[[3]],4)`.  

### Evaluation on the validation set  
The last step is to check the predictions of the final model on the validation set, which has been held out until this point:  
```{r}
confusionMatrix(predict(impRF, pmlTrainValidate),pmlTrainValidate$classe)
```  
Clearly, this model has extremely high accuracy, even on the validation data set.  How well it would do on subjects not used to train the data, however, remains to be seen.  

### Predicting the quiz data
The ultimate goal of this assignment was to predict the 20 observations of quiz data with the classifications withheld.  The resulting predictions are below.  
```{r}
outQuiz <- predict(impRF, pmlTest)
names(outQuiz) <- pmlTest$problem_id
print(outQuiz)
```  


## Citations  
1.  The data used in this report is licensed by the authors under the Creative Commons license.  Their exploration of the data can be found in:  
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. *[Qualitative Activity Recognition of Weight Lifting Exercises](http://groupware.les.inf.puc-rio.br/work.jsf?p1=11201)*. Proceedings of 4th International Conference in Cooperation with SIGCHI
(Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

2.  Thanks to Len Greski for a helpful article on configuring caret to train models more efficiently.  The article can be found [here.](https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md)


3.  Thanks to deas on graphical comparisons of machine learning algorithm results.  The page can be found [here.](http://machinelearningmastery.com/compare-models-and-select-the-best-using-the-caret-r-package/) 

## Session Info for Reproducability  
```{r}
sessionInfo()
```
