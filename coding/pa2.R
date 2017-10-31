library(cluster)
library(caret)
library(ROCR)
library(Metrics)
library('e1071')

# setwd("/Users/Chauncey/Workspace/data-mining/coding") # set working directory

# Task 1
# Read in the dataset and split the dataset randomly into 80% training data and 20% test data
set.seed(1) # set the seed of sample to 1
titantic = read.csv("titanic3.csv") # read in the Titantic dataset from .csv file

# split the dataset randomly into 80% training data and 20% test data
traing_data_size = floor(0.8 * nrow(titantic))
training_data_index = sample(seq_len(nrow(titantic)), size = traing_data_size) 
training_data = titantic[training_data_index, ]
test_data = titantic[-training_data_index, ]

# Task 2
# Report the number of missing values per attribute in the training and test dataset
training_data[training_data == ""] = NA # set all empty values in training set as NA
test_data[test_data == ""] = NA # set all empty value in test set as NA
missing_training_data = sapply(training_data, function(x) sum(is.na(x)))
missing_test_data = sapply(test_data, function(x) sum(is.na(x)))
# print(missing_training_data)
# print(missing_test_data)

# Task 3
# name, ticket, boat, body
drop <- c("boat", "body", "name", "ticket", "cabin", "home.dest")
training_data = training_data[,!(names(training_data) %in% drop)]
test_data = test_data[,!(names(test_data) %in% drop)]

# Task 4
# 1. For attributes age, replace the missing value with the meaning value
# 2. Attribute fare and embarked has very few missing values, delete the
#    records that have missing values in these attributes. 
training_data$age[is.na(training_data$age)] = mean(training_data$age, na.rm = TRUE)
test_data$age[is.na(test_data$age)] = mean(test_data$age, na.rm = TRUE)
training_data = training_data[!is.na(training_data$fare),]
test_data = test_data[!is.na(test_data$fare),]
training_data = training_data[!is.na(training_data$embarked),]
test_data = test_data[!is.na(test_data$embarked),]

# Task 5
# Learn a logistic regression model from the training data
regression_res = glm(survived~., family = binomial(link = 'logit'),  training_data)
# print(summary(regression_res))

# Task 6
# Apply the logistic regression model to predict the class labels of the test data.
# Plot the confusion matrix. What is the accuracy of the model?
prediction_res = round(predict(regression_res, newdata = test_data, type = "response"))
confusion_matrix = confusionMatrix(data = prediction_res, reference = test_data$survived)
print(confusion_matrix$table) 
print(confusion_matrix$overall[1]) # Accuracy: 0.8092

# Task 7
# Plot the ROC curve of your logistic regression model for varying probability thresholds
prediction_res = predict(regression_res, newdata = test_data, type = "response")
pr = prediction(prediction_res, test_data$survived)
predit_preformance = performance(pr, measure = "tpr", x.measure = "fpr")
plot(predit_preformance, main="ROC Curve for Logistic Regression Model")
print(auc(test_data$survived, prediction_res)) # AUC: 0.841636

# Task 8, 9 for untrained kernels
# Learn SVM models from the training data, using linear and radial kernels
# Apply the best SVM model to predict the class labels of the test data. Plot the confusion matrix.
linear_tune_res = svm(survived~., data = training_data, kernel = "linear")
radial_tune_res = svm(survived~., data = training_data, kernel = "radial")
svm_linear_res = round(predict(linear_tune_res, test_data))
confusion_matrix_linear = confusionMatrix(data = svm_linear_res, reference = test_data$survived)
svm_radial_res = round(predict(radial_tune_res, test_data))
confusion_matrix_radial = confusionMatrix(data = svm_radial_res, reference = test_data$survived)
print(confusion_matrix_linear)
print(confusion_matrix_radial)

# Task 8, 9 for trained kernels
# Learn SVM models from the training data, using linear and radial kernels
# Apply the best SVM model to predict the class labels of the test data. Plot the confusion matrix.
linear_tune_res = tune.svm(survived~., data = training_data, kernel = "linear", cost = seq(1, 6, 1), epsilon = seq(0.4, 0.6, 0.05))
radial_tune_res = tune.svm(survived~., data = training_data, kernel = "radial", cost = seq(1, 6, 1), gamma = 0.125, epsilon = seq(0.3, 0.5, 0.1))
print(linear_tune_res$best.parameters)
print(radial_tune_res$best.parameters)
svm_linear_res = round(predict(linear_tune_res$best.model, test_data))
confusion_matrix_linear = confusionMatrix(data = svm_linear_res, reference = test_data$survived)
svm_radial_res = round(predict(radial_tune_res$best.model, test_data))
confusion_matrix_radial = confusionMatrix(data = svm_radial_res, reference = test_data$survived)
print(confusion_matrix_linear)
print(confusion_matrix_radial)

# Task 10
# Plot the ROC curve of your best SVM model for varying probability thresholds
svm_linear_res = predict(radial_tune_res$best.model, test_data)
pr = prediction(svm_linear_res, test_data$survived)
predit_preformance = performance(pr, measure = "tpr", x.measure = "fpr")
plot(predit_preformance, main = "ROC Curve for SVM Model with Radial Kernel")
print(auc(test_data$survived, svm_linear_res)) # AUC: 0.8567096
