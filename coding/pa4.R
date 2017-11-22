library(arules)
library(arulesViz)
data(Groceries)

# Task 1
# Plot a histogram of the number of items (categories) per transaction
itemFrequencyPlot(Groceries,topN=20,type="absolute")
items_per_transaction = summary(Groceries)@lengths
plot(items_per_transaction,
     xlab = "number of items in a transaction",
     ylab = "number of transactions",
     main = "Histogram of the Number of Items per Transaction")

# Task 2
# How many frequent itemsets, closed frequent itemsets, and maximal
# frequent itemsets do you obtain with minimum support = 0.001
frequent_itemset = apriori(Groceries,
    parameter = list(target = "frequent itemsets", support = 0.001))
closed_frequent_itemset = apriori(Groceries,
    parameter = list(target = "closed frequent itemsets", support = 0.001))
maximal_frequent_itemset = apriori(Groceries,
    parameter = list(target = "maximally frequent itemsets", support = 0.001))
# print(frequent_itemset) # 13492
# print(closed_frequent_itemset) # 13464
# print(maximal_frequent_itemset) # 7794

# Task 3
# How many frequent itemsets, closed frequent itemsets, and maximal
# frequent itemsets do you obtain with minimum support = 0.01
frequent_itemset = apriori(Groceries,
    parameter = list(target = "frequent itemsets", support = 0.01))
closed_frequent_itemset = apriori(Groceries,
    parameter = list(target = "closed frequent itemsets", support = 0.01))
maximal_frequent_itemset = apriori(Groceries,
    parameter = list(target = "maximally frequent itemsets", support = 0.01))
# print(frequent_itemset) # 333
# print(closed_frequent_itemset) # 333
# print(maximal_frequent_itemset) # 243

# Task 4
# What are the 10 itemsets with the highest support, and what is their support?
ten_highest_support = inspect(sort(frequent_itemset)[1:10])

# Task 5 Explained in report

# Task 6
# At minimum support = 0.01, how many association rules do you obtain with minimum confidence = 0.9?
# How far do you need to lower the minimum confidence to obtain more than 10 rules?
rules1 = apriori(Groceries, parameter = list(support = 0.01, confidence = 0.9)) # 0 rules
rules2 = apriori(Groceries, parameter = list(support = 0.01, confidence = 0.5175)) # 10 rules

# Task 7
# For minimum support = 0.01 and minimum confidence = 0.5, 
# plot only the rules that have "whole milk" in their right hand side.
rules = apriori(Groceries, 
                parameter = list(support = 0.01, confidence = 0.5),
                appearance = list(rhs="whole milk")) 
inspect(rules)
plot(rules, main = "Rules That Have Whole Milk in Their Right Hand Side")

# Task 8
# Among the rules produced in task 7, which ones have the highest lift?
rules_by_lift <- sort(rules, decreasing=TRUE,by="lift")
inspect(rules_by_lift) # {curd,yogurt} => {whole milk}
