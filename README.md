# Naive Bayes Text Classifier - Hotel Reviews Dataset
(Written originally for a Natural Language Processing course assignment) 

Training set: 1280 reviews  
Test set: 320 reviews  
Labels (2 per review): [deceptive(D), truthful(T)] and [positive(P), negative(N)] 

Problem could be interpreted in following ways:
- 2 separate binary classification tasks (label each review D or T, P or N)
- single 4-way classification task (label each review DP, DN, TP, TN)

Evaluation measure: Precision, Recall, F-Score for each class
