library(tree)
library(randomForest)
setwd("/Users/Chauncey/Workspace/data-mining/coding") # set working directory

############################ Data preprocessing ########################################################

# Read in the dataset and split the dataset randomly into 80% training data and 20% test data
set.seed(1) # set the seed of sample to 1
titantic = read.csv("titanic3.csv") # read in the Titantic dataset from .csv file

# split the dataset randomly into 80% training data and 20% test data
traing_data_size = floor(0.8 * nrow(titantic))
training_data_index = sample(seq_len(nrow(titantic)), size = traing_data_size) 
training_data = titantic[training_data_index, ]
test_data = titantic[-training_data_index, ]

# Report the number of missing values per attribute in the training and test dataset
training_data[training_data == ""] = NA # set all empty values in training set as NA
test_data[test_data == ""] = NA # set all empty value in test set as NA

# Drop name, ticket, boat, body
drop <- c("boat", "body", "name", "ticket", "cabin", "home.dest")
training_data = training_data[,!(names(training_data) %in% drop)]
test_data = test_data[,!(names(test_data) %in% drop)]

# 1. For attributes age, replace the missing value with the meaning value
# 2. Attribute fare and embarked has very few missing values, delete the
#    records that have missing values in these attributes. 
training_data$age[is.na(training_data$age)] = mean(training_data$age, na.rm = TRUE)
test_data$age[is.na(test_data$age)] = mean(test_data$age, na.rm = TRUE)
training_data = training_data[!is.na(training_data$fare),]
test_data = test_data[!is.na(test_data$fare),]
training_data = training_data[!is.na(training_data$embarked),]
test_data = test_data[!is.na(test_data$embarked),]

# convert integer to factor to generate classification tree instead of regression tree
training_data$survived = factor(training_data$survived)
test_data$survived = factor(test_data$survived)

#########################################################################################################

# Task 1
decision_tree <- tree(survived~., data = training_data)
print(summary(decision_tree))
plot(decision_tree); text(decision_tree)

# Task 2  most important five attribute: sex, pclass, fare, age, sibsp

# Task 3
cv_decision_tree = cv.tree(decision_tree)
plot(cv_decision_tree);
best_size = cv_decision_tree$size[which(cv_decision_tree$dev == min(cv_decision_tree$dev))]
decision_tree_pruned = prune.misclass(decision_tree, best = best_size)
plot(decision_tree_pruned); text(decision_tree_pruned)

# Task 4
predict_res = predict(decision_tree_pruned, test_data, "class")
confusion_matrix = confusionMatrix(data = predict_res, reference = test_data$survived) # 0.8206

# Convert factor back to numeric to enable prediction
predict_res = as.numeric(as.character(predict_res))
actual_value = as.numeric(as.character(test_data$survived))
pr = prediction(predict_res, actual_value)
predit_preformance = performance(pr, measure = "tpr", x.measure = "fpr")
plot(predit_preformance, main = "ROC Curve for pruned decision tree")
print(auc(test_data$survived, predict_res))

# Task 5
random_forest = randomForest(survived~., data = training_data, nTree = 100)
random_forest_predict = predict(random_forest, test_data, type = "class")
print(mean(random_forest_predict == actual_value))

# Task 6
random_forest = randomForest(survived~., data = training_data, nTree = 100)
random_forest_predict = predict(random_forest, test_data, type = "class")
print(mean(random_forest_predict == actual_value))

# Task 7 
# sex, fare, age, pclass, sibsp
# attributes are the same compared with attributes selected in task2
imp = importance(random_forest)
imp_plot = varImpPlot(random_forest)
