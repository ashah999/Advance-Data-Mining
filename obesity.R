# Load all the packages required for the analysis
library(dplyr) # Data Manipulation
library(Amelia) # Missing Data: Missings Map
library(ggplot2) # Visualization
library(scales) # Visualization
library(caTools) # Prediction: Splitting Data
library(car) # Prediction: Checking Multicollinearity
library(ROCR) # Prediction: ROC Curve
library(e1071) # Prediction: SVM, Naive Bayes, Parameter Tuning
library(rpart) # Prediction: Decision Tree
library(rpart.plot) # Prediction: Decision Tree
library(randomForest) # Prediction: Random Forest
library(caret) # Prediction: k-Fold Cross Validation

setwd("C:/Users/Anish/Downloads/eating-health-module-dataset1")
data<-read.csv("C:/Users/Anish/Downloads/eating-health-module-dataset1/ehresp_2014.csv")

str(data)


#Checking for Missing Values
colSums(is.na(data)|data=='')

head(data$eeincome1)
BMImean <-mean(data$erbmi)
BMImedian <- median(data$erbmi)
BMImedian
BMImean

BMImax <- max(data$erbmi)
BMIMin <- min(data$erbmi)
BMImax
BMIMin

as.double(data$erbmi)
plot(data$erbmi)
length(data$erbmi)
temp <- abs(data)
#Creating new cloumn by taking mean of BMI .
#Adult BMI chart showing ranges "obese I: BMI 30–34.9," "obese II: BMI 35–39.9" and "obese III: BMI ≥ 40."
#If your BMI is less than 18.5, it falls within the underweight range.
#If your BMI is 18.5 to <25, it falls within the normal. 
#If your BMI is 25.0 to <30, it falls within the overweight range. If your BMI is 30.0 or higher







temp$obease = ifelse(temp$erbmi>=26.5,"Yes","No")
write.csv(temp,file="main.csv")
str(temp)  

table(temp$obease)

hist(temp$erbmi)
ggplot(temp,aes(temp$obease,temp$erbmi)) +                                                  
  geom_boxplot(aes(fill=factor(temp$obease)),alpha=0.5) +
  ggtitle("BMI distribution against Obesity")


maxbmi <- max(data[!is.na(temp$erbmi),]$erbmi)


temp$range <- cut(temp$erbmi,breaks =c(0,18.5,25,30,maxbmi),labels = c("0-18.5","18.5-25","25-30","30"))
table(temp$range)
prop.table(table(temp$range))
colSums(is.na(temp)|temp=='')
#filter(data, is.na(data$obease)==TRUE|data$obease=='')
#data$obease[is.na(data$obsese)] <- "Yes"
#colSums(is.na(data)|data=='')
#table(data$obease)
as.numeric(temp$erbmi)
a<-write.csv(temp,file="main1.csv")
str(temp)

final <- read.csv("final.csv")

set.seed(1975)
index <- sample(1:dim(final)[1],dim(temp)[1]* .75 ,replace = FALSE)
training <-final[index,]
testing <-final[-index,]

model <- glm(erbmi ~ euexfreq + euwgt + euhgt + ertpreat + eufastfdfrq, data = temp)
plot(model)
par(mfrow = c(2,2))
plot(model)

avPlots(model)
library(corrplot)
corrplot.mixed(corr = cor(temp[,3:37]),tl.pos = "lt")
colSums(is.na(data)|data=='')
###########################################Naive Bayes#########################################################

library(e1071)
library(pROC)
testing$obease<- as.factor(testing$obease)
training$obease <- as.factor(training$obease)
#For Tunning Naive Bayes
#search_grid <- expand.grid(usekernel = c(TRUE, FALSE),  fL = 0:5,  adjust = seq(0, 5, by = 1))

nb <- naiveBayes(training,training$obease)
nbpredict <- predict(nb,newdata = testing[,-21])
caret :: confusionMatrix(nbpredict,testing$obease,positive = "Yes")
auc(naiveBayes)
str(nbpredict)
str(testing$obease)
nbpredic <- as.numeric(nbpredict)

#plotting ROC CUrve
nb1 <- prediction(as.numeric(nbpredict), testing$obease)
roc_nb <- performance(nb1, measure = "tpr", x.measure = "fpr")
plot(roc_nb)
auc(testing$obease, nbpredict)

library(mlr)
#Create a classification task for learning on obease Dataset and specify the target feature
task = makeClassifTask(data = training, target = "obease")
#Initialize the Naive Bayes classifier
selected_model = makeLearner("classif.naiveBayes")
#Train the model
NB_mlr = train(selected_model, task)
#Read the model learned  
NB_mlr$learner.model
#Predict on the dataset without passing the target feature
predictions_mlr = as.data.frame(predict(NB_mlr, newdata = testing[,-21]))
##Confusion matrix to check accuracy
table(predictions_mlr[,1],testing$obease)
caret :: confusionMatrix(predictions_mlr[,1],testing$obease,positive = "No")

####################################Feature Selection ##############################################################
# 1 ] Using Boruta 
library(Boruta)
boruta_output <- Boruta(training$obease ~ ., data=na.omit(training), doTrace=0) 
names(boruta_output)
boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE)
print(boruta_signif)  

dat <- boruta_signif

roughFixMod <- TentativeRoughFix(boruta_output)
boruta_signif <- getSelectedAttributes(roughFixMod)
print(boruta_signif)
imps <- attStats(roughFixMod)
imps2 = imps[imps$decision != 'Rejected', c('meanImp', 'decision')]
head(imps2[order(-imps2$meanImp), ])  # descending sort

# 2] Random Forest 
library(randomForest)
final$obease <- as.factor(final$obease)
model1 <- randomForest(training$obease ~ ., data = training, importance = TRUE,ntree=100)
model1
predTrain <- predict(model1, testing, type = "class")
caret::confusionMatrix(testing$obease, predTrain)
# Checking classification accuracy
table(predTrain, testing$obease)  
predValid <- predict(model1, testing, type = "class")
# Checking classification accuracy
mean(predValid == testing$obease)                    
table(predValid,testing$obease)
importance(model1)        
varImpPlot(model1) 

#plotting ROC CUrve
rf <- prediction(as.numeric(predTrain), testing$obease)
roc_rf <- performance(rf, measure = "tpr", x.measure = "fpr")
plot(roc_rf)
auc(testing$obease, predTrain)

#######################################Decision Tree###########################################################

library(C50)
c50model <- C5.0(training$obease ~., data=training, trials=10)
plot(c50model)
summary(c50model)
#indexdecision <- sample(1:length(temp),length(temp)*.30, replace= FALSE)
#training_tree <- temp[indexdecision]
#testing_tree <- temp[-indexdecision]
cFifity <- C5.0(training$obease ~ .,data = training)
cFifity
c <- predict(cFifity,testing[,-21])
caret :: confusionMatrix(c,testing$obease,positive = "No")

#plotting ROC CUrve
dt <- prediction(as.numeric(c), testing$obease)
roc_dt <- performance(dt, measure = "tpr", x.measure = "fpr")
plot(roc_dt)
auc(testing$obease, c)

#Winnowing Feature Selection process
cFiftyWinnow <- C5.0(training$obease ~ ., data = training, control = C5.0Control(winnow = TRUE))
c <- predict(cFiftyWinnow,testing[,-38])
caret :: confusionMatrix(testing$obease,c,positive = "Yes")

control <- trainControl(method="repeatedcv", number=10, repeats=5) #5 x 10-fold cv
metric <- "Kappa"
temp <- as.data.frame(temp)
optimModel <- train(temp$obease~., data=temp, method="C5.0", metric=metric, trControl=control)
plot(optimModel)


str(training)
str(testing)


######################################Support Vector Machine (SVM)################################
library(kernlab)

svm_model <- svm (obease ~., data=training)
summary (svm_model)
pred1 <- predict (svm_model, testing)
library (caret)

caret::confusionMatrix (pred1, testing$obease)

#plotting ROC curve
pr <- prediction(as.numeric(pred1), testing$obease)
roc <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(roc)
auc(testing$obease, pred1)







