#Richa Bhattacharya
#Mentor David Cohn
#Machine learning portion 

code.directory= "Machine Learning Analysis"
setwd = code.directory

# Loading libraries needed
library(ggplot2)
library(dplyr)
library(tidyr)
library(survival)
library(survminer)
library(caret)
library(stats)
library(randomForest)
library(e1071)
library(ipred)
library(gbm)

# Read in data
data.extract = read.csv('dataextract.csv', stringsAsFactors = FALSE)

set.seed(1427)

machine.learning.data = filter(data.extract,
                               Survival.Months > 0 & Survival.Status == 1) 

median.survival = median(machine.learning.data$Survival.Months)

machine.learning.data = machine.learning.data %>%
  mutate(label = ifelse(Survival.Months >= median.survival, "G", "L")) %>%
  select(label, Grade, Histology, Gender, Medical.Facility, 
         Percent.Aneuploidy, Mutation.Count, Age.at.diagnosis)  

machine.learning.data$Grade = as.factor(machine.learning.data$Grade)
machine.learning.data$Histology = as.factor(machine.learning.data$Histology)
machine.learning.data$label = as.factor(machine.learning.data$label)
machine.learning.data$Gender = as.factor(machine.learning.data$Gender)
machine.learning.data$Medical.Facility = as.factor(machine.learning.data$Medical.Facility)
machine.learning.data$Mutation.Count = as.integer(machine.learning.data$Mutation.Count)

training.indices = createDataPartition(machine.learning.data$label, p = .8, list = FALSE)

training.data = machine.learning.data[training.indices, ]
  data.imputation = preProcess(training.data, method = c("bagImpute"))
training.data = predict(data.imputation, training.data)

test.data = machine.learning.data[-training.indices, ]
test.data = predict(data.imputation, test.data)

lasso.lambda.parameters = seq(from = -2, to = 3, by = 1)
lasso.lambda.parameters = 10^(lasso.lambda.parameters)
lasso.alpha = 1
lasso.grid = expand.grid(lambda = lasso.lambda.parameters, alpha = lasso.alpha)

random.forest.mtry.parameters = c(ncol(machine.learning.data) - 1, 3, 5)
random.forest.grid = expand.grid(mtry = random.forest.mtry.parameters)

fit.control = trainControl(method = "cv", number = 3, classProbs = TRUE,
                           summaryFunction = twoClassSummary)

lasso.logistic.regression.model = train(label ~., data = training.data,
                                        method = "glmnet", metric = "ROC",
                                        tuneGrid = lasso.grid, trControl = fit.control)

ggplot(lasso.logistic.regression.model)

random.forest.model = train(label ~., data = training.data, method = "rf",
                            metric = "ROC", tuneGrid = random.forest.grid, 
                            trControl = fit.control)

ggplot(random.forest.model)


lasso.test.predictions = predict(lasso.logistic.regression.model, newdata = test.data)
random.forest.predictions = predict(random.forest.model, newdata = test.data)


lasso.confusion.matrix = confusionMatrix(lasso.test.predictions, test.data$label)
random.forest.confusion.matrix = confusionMatrix(random.forest.predictions, test.data$label)

#I think utilizing a boosted tree machine learning model 
#(method = "gbm") might be better. As such, perhaps you could implement 
#a boosted tree machine learning model, based on the code that I provided 
#for the logistic regression and random forest models?

#MY ADDED CODE
boost.parameters = c(ncol(machine.learning.data) - 1, 3, 5) #what are the parameters?
boost.grid = expand.grid(mtry = boost.parameters)
boost. = gbm(label~., data = training.data, distribution = "gaussian", n.tree = 100)
boost.model = train(label ~., data = training.data, method = "rf",
                            metric = "ROC", tuneGrid = random.forest.grid, 
                            trControl = fit.control)

















