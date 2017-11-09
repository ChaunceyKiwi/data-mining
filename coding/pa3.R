library(tree)
library(cluster)
library(caret)
library(ROCR)
library(Metrics)
library(randomForest)
library('e1071')
setwd("/Users/Chauncey/Workspace/data-mining/coding") # set working directory

############################ Data preprocessing ########################################################
set.seed(1) # set the seed of sample to 1
titantic = read.csv("titanic3.csv") # read in the Titantic dataset from .csv file
traing_data_size = floor(0.8 * nrow(titantic)) # split the dataset into training data and test data
training_data_index = sample(seq_len(nrow(titantic)), size = traing_data_size) 
training_data = titantic[training_data_index, ]
test_data = titantic[-training_data_index, ]
training_data[training_data == ""] = NA # set all empty values in training set as NA
test_data[test_data == ""] = NA # set all empty value in test set as NA
drop <- c("boat", "body", "name", "ticket", "cabin", "home.dest") # Drop name, ticket, boat, body
training_data = training_data[,!(names(training_data) %in% drop)]
test_data = test_data[,!(names(test_data) %in% drop)]
training_data$age[is.na(training_data$age)] = mean(training_data$age, na.rm = TRUE)
test_data$age[is.na(test_data$age)] = mean(test_data$age, na.rm = TRUE)
training_data = training_data[!is.na(training_data$fare),]
test_data = test_data[!is.na(test_data$fare),]
training_data = training_data[!is.na(training_data$embarked),]
test_data = test_data[!is.na(test_data$embarked),]

# convert integer to factor to generate classification tree instead of regression tree
training_data$survived = factor(training_data$survived)
test_data$survived = factor(test_data$survived)
training_data$pclass = factor(training_data$pclass)
test_data$pclass = factor(test_data$pclass)

############################### Task of Assignment 3 ######################################################

# Task 1, the size of tree is 233
# Task 2, the most important five attributes: parch, sibsp, sex, pclass, fare
decision_tree = tree(survived~., data = training_data, split = "gini")
print(decision_tree)
print(summary(decision_tree))
plot(decision_tree)
print(nrow(decision_tree$frame))
text(decision_tree)

# Task 3 best_size as 10
cv_decision_tree = cv.tree(decision_tree, FUN = prune.misclass)
plot(cv_decision_tree)
best_size = min(cv_decision_tree$size[which(cv_decision_tree$dev == min(cv_decision_tree$dev))])
decision_tree_pruned = prune.tree(decision_tree, best = best_size)
plot(decision_tree_pruned); text(decision_tree_pruned)

# Task 4
predict_res = predict(decision_tree_pruned, test_data, type = "vector")
predict_res = predict_res[, 2]
predict_res_bi = round(predict_res)
confusion_matrix = confusionMatrix(data = predict_res_bi, reference = test_data$survived) # 0.771

# Convert factor back to numeric to enable prediction
actual_value = as.numeric(as.character(test_data$survived))
pr = prediction(predict_res, actual_value)
print(paste0("Accuracy: ", mean(predict_res_bi == actual_value))) # 0.770992366412214
predit_preformance = performance(pr, measure = "tpr", x.measure = "fpr")
plot(predit_preformance, main = "ROC Curve for pruned decision tree")
print(paste0("AUC: ", auc(test_data$survived, predict_res))) # AUC = 0.835386029411765

# Task 5, 6
# num = 100, accuracy = 0.8053435, AUC = 0.8896752
# num = 95, accuracy = 
set.seed(1)
index = 1
candidate_number = c(90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100)
accuracy = rep(0, length(candidate_number))
AUC = rep(0, length(candidate_number))
iter = 10
for (numOfTree in candidate_number) {
  for (i in 1:iter) {
    random_forest = randomForest(survived~., data = training_data, ntree = numOfTree)
    random_forest_predict = predict(random_forest, test_data, type = "prob")
    random_forest_predict = random_forest_predict[, 2]
    random_forest_predict_bi = round(random_forest_predict)
    pr = prediction(random_forest_predict, actual_value)
    predit_preformance2 = performance(pr, measure = "tpr", x.measure = "fpr")
    # plot(predit_preformance2, main = "ROC Curve for random_forest with size 100")

    accuracy[index] = accuracy[index] + mean(random_forest_predict_bi == actual_value)
    AUC[index] = AUC[index] + auc(test_data$survived, random_forest_predict)
    # print(mean(random_forest_predict_bi == actual_value))
  }

  accuracy[index] = accuracy[index] / iter
  AUC[index] = AUC[index] / iter
  index = index + 1
}

plot(candidate_number, accuracy, type="o", col="blue")
plot(candidate_number, AUC, type="o", col="red")

# Task 7
# sex, fare, age, pclass, sibsp
# attributes chosen are the same compared with
# attributes selected in task2, but the order is different
imp = importance(random_forest)
imp_plot = varImpPlot(random_forest)