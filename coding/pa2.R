library(cluster) 
library(caret)
library(ROCR)
library(Metrics)
library('e1071')

setwd("/Users/Chauncey/Workspace/data-mining/coding") # set working directory 

# Task 1
titantic = read.csv("titanic3.csv") # read in the Titantic dataset from .csv file
titantic[titantic == ""] = NA;

sample_size = floor(0.8 * nrow(titantic)) # get 80% of sample size
set.seed(1)
train_set_index = sample(seq_len(nrow(titantic)), size = sample_size)

training_data = titantic[train_set_index, ]
testing_data = titantic[-train_set_index, ]


# Task 2
missing_training_data = sapply(training_data, function(x) sum(is.na(x)))
missing_testing_data = sapply(testing_data, function(x) sum(is.na(x)))

# Task 3
# pclass, sex, age, sibsp, parch, fare, cabin, boat, home.dest

# Task 4
# 1. Delete cabin, body, boat, home attributes since there are too many missing value
# 2. For remaining attributes age, fare, use the meaning value as the missing value
# 3. Skipped categorical data like embark, 

drop <- c("cabin", "boat", "body", "home.dest")
training_data = training_data[,!(names(training_data) %in% drop)]
testing_data = testing_data[,!(names(testing_data) %in% drop)]

training_data$age[is.na(training_data$age)] <- mean(training_data$age, na.rm = TRUE)
testing_data$age[is.na(testing_data$age)] <- mean(testing_data$age, na.rm = TRUE)
training_data$fare[is.na(training_data$fare)] <- mean(training_data$fare, na.rm = TRUE)
testing_data$fare[is.na(testing_data$fare)] <- mean(testing_data$fare, na.rm = TRUE)

# Task 5
# Learn a logistic regression model from the training data
# Skip attributes name and ticket since they have unique values 
regression_training_data = training_data[c("survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked")]
regression_res = glm(survived~., family = binomial(link = 'logit'),  regression_training_data)
# print(summary(regression_res))
# print(anova(regression_res, test = 'Chisq'))

# Task 6
# Apply the logistic regression model to predict the class labels of the test data. 
# Plot the confusion matrix. What is the accuracy of the model?
prediction_res = round(predict(regression_res, newdata = testing_data, type = "response"))
confusion_matrix = confusionMatrix(data = prediction_res, reference = testing_data$survived)
# print(confusion_matrix) # Accuracy: 0.8092

# Task 7
# Plot the ROC curve of your logistic regression model for varying probability thresholds
prediction_res = predict(regression_res, newdata = testing_data, type = "response")
pr = prediction(prediction_res, testing_data$survived)
predit_preformance = performance(pr, measure = "tpr", x.measure = "fpr")
plot(predit_preformance)  
print(auc(testing_data$survived, prediction_res)) # AUC: 0.841636

# Task 8
# Learn SVM models from the training data, using linear and radial kernels
linear_tune_res = tune(svm, survived~., data = training_data, kernel = "linear")
radial_tune_res = tune(svm, survived~., data = training_data, kernel = "radial")
# print(radial_tune_res$best.parameters) # dummyparameter 0
# print(linear_tune_res$best.parameters) # dummyparameter 0

# Task 9
# Apply the best SVM model to predict the class labels of the test data. Plot the confusion matrix.
# svm_radical_res = round(predict(radial_tune_res$best.model, testing_data))
# confusion_matrix_radical = confusionMatrix(data = svm_radical_res, reference = testing_data$survived)
svm_linear_res = round(predict(linear_tune_res$best.model, testing_data))
confusion_matrix_linear = confusionMatrix(data = svm_linear_res, reference = testing_data$survived)
print(confusion_matrix_linear)

# Task 10
# Plot the ROC curve of your best SVM model for varying probability thresholds
svm_linear_res = predict(linear_tune_res$best.model, testing_data)
pr = prediction(svm_linear_res, testing_data$survived)
predit_preformance = performance(pr, measure = "tpr", x.measure = "fpr")
plot(predit_preformance)
print(auc(testing_data$survived, svm_linear_res)) # AUC: 0.8923407
