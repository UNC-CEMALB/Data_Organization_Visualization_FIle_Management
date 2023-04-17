# 4.2: Supervised Machine Learning 

This training module was developed by Oyemwenosa N. Avenbuan, Alexis Payton, and Dr. Julia E. Rager

Spring 2023

## Machine Learning Review

Machine Learning is a field of study in computer science that involves creating algorithms(a set of instructions that perform a specific task on a given dataset). Machine Learning is a scientific approach that enables researchers to create models that can automatically adapt to new and unforeseen situations) capable of improving automatically through experience and data.

In other words, instead of being explicitly programmed to perform a task, a machine learning algorithm is designed to learn from examples and data, allowing it to adapt and improve over time. This approach is particularly useful for tasks that are too complex or difficult to be solved using traditional programming methods.

Through Machine Learning, scientists can:

1\. Create a model that adapts to new circumstances that the scientist did not envision.

2\.Detect patterns in large and complex datasets.

3\. Evaluate the effectiveness of these patterns. 

4\. Make informed decisions about how to improve their models.

Ultimately, Machine Learning is a powerful tool that enables researchers to analyze data more effectively, make more accurate predictions, and develop more advanced systems that can learn and evolve over time.

## Types of Learning

In the field of Machine Learning, there are two broad types of learning: supervised learning and unsupervised learning.

**Supervised learning** involves training a machine learning model using a labeled dataset, where each example is associated with a known outcome or target variable. The model is then able to learn how to predict the outcome for new, unseen examples based on the patterns and relationships it identifies in the data.

Supervised Learning can either be: 
1. Classification: Using algorithms to classify categories based on various characteristics
2. Regression: Using algorithms to understand the relationship between independent and dependent variables

![image](https://user-images.githubusercontent.com/96756991/228436976-ec715a4f-575f-4718-89d6-6233695fcd7f.png)
Created with BioRender.com
[Reference] (https://www.ibm.com/cloud/blog/supervised-vs-unsupervised-learning)
![image](https://user-images.githubusercontent.com/96756991/232521668-5752b759-4084-4c2e-be6b-01393378ca40.png)

**Unsupervised learning**, on the other hand, involves training a machine learning model on an unlabeled dataset, where the outcome or target variable is unknown. The model is then tasked with identifying patterns and structures in the data, such as clusters of similar examples or underlying relationships between variables.

![image](https://user-images.githubusercontent.com/96756991/228424183-fdc60f87-f617-47e4-ab8e-f2cc9c1f1400.png)
Created with BioRender.com

It's worth noting that there are also other types of learning in Machine Learning, such as semi-supervised learning and reinforcement learning, which combine elements of both supervised and unsupervised learning.

Overall, the distinction between supervised and unsupervised learning is an important concept in Machine Learning, as it can inform the choice of algorithms and techniques used to analyze and make predictions from data.

## Training Your Model

In Machine Learning, before we can effectively use algorithms to analyze data, we first need to train them. This involves selecting a smaller portion or subset of data, known as training data, to teach the algorithm how to identify distinct patterns. By recognizing these patterns, the algorithm can accurately classify specific cases within a larger and more complex dataset. The process of training an algorithm is essential for enabling it to learn and improve over time, allowing it to make more accurate predictions and better adapt to new and changing circumstances. Ultimately, the effectiveness of a machine learning model depends on the quality and relevance of its training data.

In Machine Learning, the process of developing a model involves dividing the data into three distinct sets:

1\.**The training set:** a subset of the data that is used to fit the model. Essentially, the model learns from this data and uses it to identify patterns and relationships in the data.

2\. **The validation set:** a sample of data that is used to evaluate the model's fit in an unbiased way. It helps develop the model by fine-tuning its parameters and optimizing its performance. This is akin to pop-quizzes that help students improve their understanding and performance.

3\. **The test set:** a sample of data that is used to provide an evaluation of the final model's fit on the training set. This is the model's final exam, as it provides an objective assessment of the model's ability to generalize to new, unseen data.

It's important to note that the test set should only be used once, after the model has been fully developed and fine-tuned on the training and validation sets. Using the test set multiple times during the development process can lead to overfitting, where the model performs well on the test data but poorly on new, unseen data.

Overall, the process of dividing the data into training, validation, and test sets is a crucial step in developing accurate and reliable machine learning models.

[Reference](<https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7>) 


**Other Supervised Machine Learning Algorithms**

Before we create a decision tree and random forest, we want to mention other popular algorithims that can be used in supervised machine learning. Two models that are good to know are: 

1\. **K-nearest Neighbors(KNN):** Uses proximity to make predictions/classify data points 
![image](https://user-images.githubusercontent.com/96756991/232493057-1e7ce98b-6985-44cd-98a9-3cfea5994659.png)
A simple illustration of KNN 

Created with BioRender.com
[Reference](https://www.ibm.com/topics/knn)

2\. **Support Vector Machines(SVM):** Creates best decision boundary line (hyperplane) in n-dimensional space that seperates points into categories so that when new data is presented they can be easily cateogrized. 
![image](https://user-images.githubusercontent.com/96756991/232492988-6da14533-5d09-4bc7-8230-3d680e0fa82b.png)
A simple illustration of SVM 

Created with BioRender.com
[Reference](https://www.javatpoint.com/machine-learning-support-vector-machine-algorithm)

While neither of these algorithms will be used in this training, they are important algorithms that you may come across as you continue on learning and using machine learning. 

## Confusion Matrix 

Before we get into our data and begin to train our model, we want to introduce one more concept that is important when running a classification supervised machine learning model. 

In classification models, a hurdle we have to overcome is accuracy. Sometimes, the data presented has a significant class imbalance, and the model may predict the majority class for all cases, leading to an inaccurate high accuracy score. Most error measurement accounts for total errors, so a confusion matrix is run to overcome this potential accuracy bias. 

A confusion matrix metrics measures help us find the accuracy of our classier through 4 main metrics:

1\. **Accuracy:** the portion of correctly classified values

2\. **Precision:** When the model predicts a positive value, how often is it correct?

3\. **Recall:** How often does the model predict positive values

4\. **F1 Score:** mean of recall and precision 

![image](https://user-images.githubusercontent.com/96756991/232528604-5b561769-efe6-49be-a2e0-783727b7039e.png)
Created with BioRender.com

Using a confusion matrix, we can determine which models provide more accurate outputs for classification. 

[Reference](https://www.simplilearn.com/tutorials/machine-learning-tutorial/confusion-matrix-machine-learning#confusion_matrix_metrics)

## Background on Data 

This dataset is the result of a human-exposure study, involving 27 participants, which investigated molecular alterations in sputum before and after exposure to smoldering red. Throughout the study, biological samples were collected from participants both before and after exposure. Demographic information was also gathered and the proteomic signatures in these samples were analyzed using high-resolution mass spectrometry. For the purpose of this training, we will be focusing solely on the pre-exposure data.

This project meets the UNC’s IRB approval process (IRB# 13–3076 or 18-1895 or 18-2196)

## Questions to answer 

Based on the pre-exposure data, we want to address two questions centered around sex-differences. They include:

1. Can we predict sex based on protein expression?
2. Which proteins best predict sex? 

Using A decision tree we will be able to answer question 1 and using a random forest, we will be able to answer question 2.  

## Working with the data (work in progress)

**Download Packages**
```{r}
install.packages("tidyverse")
install.packages("pheatmap")
install.packages("ggplot2")
install.packages("reshape2")
install.packages("arsenal")
install.packages("superheat")

library(tidyverse)
library(pheatmap)
library(ggplot2)
library(reshape2)
library(arsenal)
library(superheat)
library(readxl)
library(gtsummary)
library(e1071)
library(Hmisc)
library(glmnet)
library(randomForest)
library(pROC)
library(lattice)
library(survival)
library(Formula)
library(caret)
```
**Set working directory**
```{r}
getwd()
setwd("/Users/ritaavenbuan/Desktop")
```

**Read in the Data & change sex into factors**

Changing Sex into factors will help the run the machine learning models run properly.

```{r}
#import the data and remove variables that are not going to be useful for your analysis
pre.dataset <- na.omit(read.csv("Proteomics_Imputed_PreExposureSubjects.csv")) %>%
  select(-SubjectID, -Race, -Ethnicity, -Age, -BMI) %>%
  drop_na() 

#change the class of Sex (from characters to factors)
class(pre.dataset$Sex)
pre.dataset$Sex <- as.factor(pre.dataset$Sex)
```

**Statistical Summary of the Data**

```{r}
pre.dataset %>%
  tbl_summary(by = "Sex", 
               statistic = list(all_continuous() ~ "{mean} ({sd})")) %>%
  add_n() %>%
  add_p(test = list(all_continuous() ~ "aov")) %>%
  as_flex_table()


#post.dataset <-read.csv("Proteomics_Imputed_PostExposureSubjects.csv")
```

**Review the data**

```{r}
head(pre.dataset)
```

**Decision Tree**

A Decision Tree is a type of supervised machine learning model that makes predictions based on how a question was answered. 

![image](https://user-images.githubusercontent.com/96756991/228426001-9a73d4b5-017c-430b-b48f-0aa91dedc4ea.png)
Created with BioRender.com

- _Root node:_ Base of the Decision Tree 
- _Splitting:_ A node divided into sub-nodes 
- _Decision mode:_ A sub-node broken down into additional sub-nodes
- _Leaf node:_ A sub-node that does not split into additional sub-nodes (terminal node). It indicates a potential outcome. 
- _Pruning:_ Removing sub-nodes of a decision tree
- _Branch:_ A section of a decision tree with multiple nodes
- _Pruning:_ The process of removing sub-nodes of a decision tree

In this module, we will start by creating decision tree. 

[Reference](https://www.mastersindatascience.org/learning/machine-learning-algorithms/decision-tree/)

**Set data up for Reproducibility**
```{r}

#set up for reproducibility 
set.seed(15)
control_params <- rpart.control(minsplit = 5)

```

**Splitting data into testing and training set**

In this step we want to split the data into the training set (what the algorithm will use to learn data) and the test set. We also want to make sure that we are cross-validating here. 

_Cross-validate:_ The ability for the machine to predict new data and test its accuracy. We want to make sure our model is not overfitting (giving accurate results for predictions for training data but not new data). 
[Reference](https://learn.g2.com/cross-validation)

```{r}
sex_data_index = createFolds(pre.dataset$Sex, k = 5) #K in Cross Validation is usually 5 or 10 
errors = data.frame()
for (i in 1:length(sex_data_index)){
  sex_train = pre.dataset[-sex_data_index[[i]],]
  sex_test = pre.dataset[sex_data_index[[i]],]
  
  reg_tree = rpart(Sex ~., data = sex_train, method = "class", control = control_params)
  
  vim <- varImp(reg_tree)
  
  pred_tree <- predict(reg_tree, sex_test, type = "class", na.action = na.pass)
  
  cm_tree <- confusionMatrix(pred_tree, sex_test$Sex)
  
  accuracy_tree <- cm_tree$overall["Accuracy"]
}
```

**Preform Confusion Matrix**


```{r}
#perform confusion matrix 
# Set the number of folds for cross-validation
k <- 5

# Create the folds
folds <- cut(seq(1, nrow(pre.dataset)), breaks = k, labels = FALSE)

# Initialize an empty list to store the evaluation metrics for each fold
eval_metrics <- list()

# Loop over each fold
for (i in 1:k) {
  
  # Split the data into training and validation sets for this fold
  train_indices <- which(folds != i)
  valid_indices <- which(folds == i)
  sex_train <- pre.dataset[train_indices, ]
  sex_valid <- pre.dataset[valid_indices, ]
  
  # Fit the decision tree model on the training set
  reg_tree <- rpart(Sex ~., data = sex_train, method = "class", control = rpart.control(minsplit = 5))
  
  # Make predictions on the validation set
  pred_tree <- predict(reg_tree, sex_valid, type = "class")
  
  # Compute the evaluation metrics for this fold
  cm_tree <- confusionMatrix(pred_tree, sex_valid$Sex)
  eval_metrics[[i]] <- c(accuracy = cm_tree$overall['Accuracy'], 
                         sensitivity = cm_tree$byClass['Sensitivity'],
                         specificity = cm_tree$byClass['Specificity'])
}

# Calculate the average evaluation metrics across all folds
eval_metrics <- do.call(rbind, eval_metrics)
avg_eval_metrics <- colMeans(eval_metrics)

#confusion matrix results
print(cm_tree)
```
```{r}
Confusion Matrix and Statistics

          Reference
Prediction F M
         F 2 0
         M 3 1
                                          
               Accuracy : 0.5             
                 95% CI : (0.1181, 0.8819)
    No Information Rate : 0.8333          
    P-Value [Acc > NIR] : 0.9913          
                                          
                  Kappa : 0.1818          
                                          
 Mcnemar's Test P-Value : 0.2482          
                                          
            Sensitivity : 0.4000          
            Specificity : 1.0000          
         Pos Pred Value : 1.0000          
         Neg Pred Value : 0.2500          
             Prevalence : 0.8333          
         Detection Rate : 0.3333          
   Detection Prevalence : 0.3333          
      Balanced Accuracy : 0.7000          
                                          
       'Positive' Class : F               
```

Based on our confusion matrix, we have 50% accuracy which means that our model is accurately classifying proteins based on sex 50% of the time.  

**Visualize the Decision Tree**

We want to visualize the decision tree for the training data created above to see how the model is using the protemics data to categorize sex.

```{r}
reg_tree
rpart.plot(reg_tree, type = 2, extra = 101, under = TRUE, fallen.leaves = TRUE, main = "Decision Tree for Protein Differences by Sex")

cm_tree

table_matrix <- table(sex_test$Sex, pred_tree)
table_matrix
```
![image](https://user-images.githubusercontent.com/96756991/232521526-56740254-ad7e-4356-9d57-aa2c85914205.png)

Based on this decision tree, a classified protein in our proteomics training data set is CFP. When it is greater than 5540, the model predicts that the sex is female; however, when it is less than 5540, it uses another protein called BPIFB4. When BPIFB4 is greater than or equal to 6821, the model predicts the sex is female; else, it is male. 

Now, we are going to now run a more robust model Random Foreset to see if our accuracy increases and to see which proteins best predict sex. 

**Random Forest**

Random Forest creates many decision trees to reach an answer about the data set. 

![image](https://user-images.githubusercontent.com/96756991/228436169-c7fe932a-9aa4-41e2-b56c-9a8830b834a8.png)
Created with BioRender.com

[Reference](https://www.ibm.com/topics/random-forest) 

**Set data up for Reproducibility**
```{r}
rf_classification = function(pre.dataset, outcome, pred_outcome) {
  #setting for reproducibility
  set.seed(15)

```

**Split data into training and test sets**
```{r}
  #splitting data into training and testing sets 
  sex_data_index = createFolds(pre.dataset$Sex, k = 5) #K in Cross Validation is usually 5 or 10 
```

**Create an empty data frame for error, empty numeric vector for accuracy, sensitivity and specificity, and an empty list for importance**
```{r}

  errors = data.frame()
  accuracy = c()
  sensitivity = c()
  specificity = c()
  importance = list()
 ```
 
 **Run Random Forest and Cross-validation**
 ```{r}
  #set up cross valdiation loop using the sex_data_index created previously
  for (i in 1:length(sex_data_index)){
    sex_train = pre.dataset[-sex_data_index[[i]],]
    sex_test = pre.dataset[sex_data_index[[i]],]
    
    # Set up the grid of hyperparameters to search over
    ntree_values = c(50, 250, 500) #number of trees
    p = dim(pre.dataset)[2] - 1 #number of variables in dataset
    mtry_values = c(p/2, sqrt(p), p)
    
    # Set up the tuning grid for the random forest
    tuning_grid = expand.grid(ntree = ntree_values,
                              mtry = mtry_values)
    
    # Fit the random forest model with 5-fold cross-validation
    rf_model = train(as.factor(outcome) ~ ., 
                     data = sex_train, 
                     method = "rf", 
                     trControl = trainControl(method = "cv", 
                                              number = 5), 
                     tuneGrid = tuning_grid)
    
    # Make predictions on the test set
    rf_pred = predict(rf_model, newdata = sex_test)
    
    # Run confusion matrix for the test set
    cm = confusionMatrix(rf_pred, sex_test$Sex)
    accuracy[i] = cm$overall[1]
    sensitivity[i] = cm[2,2]/sum(cm[2,])
    specificity[i] = cm[1,1]/sum(cm[1,])
    
    # Save the variable importance for this fold
    importance[[i]] = varImp(rf_model)$importance
    
  }
  
  # Combine the evaluation metrics across all folds
  eval_metrics = data.frame(accuracy = mean(accuracy), 
                            sensitivity = mean(sensitivity), 
                            specificity = mean(specificity))
  
  # Run the mean importance of each variable across all folds
  variable_importance = data.frame(variable = names(pre.dataset)[-pred_outcome], 
                                   mean_importance = unlist(lapply(importance, 
                                                                   function(x) mean(x[,"MeanDecreaseGini"]))))
  
  # Return the evaluation metrics and variable importance
  return(list(eval_metrics = eval_metrics, variable_importance = variable_importance))
}
```

![image](https://user-images.githubusercontent.com/96756991/232552051-a6016e75-fd91-45fd-b237-056f1e1a56ea.png)

**Mean decrease accuracy**

Mean decrease accuracy tells us how important each feature is in making accurate predictions. A higher mean decrease accuracy score indicates that a feature is more important in making predictions, while a lower score means it's less important. With this particular dataset, we see that proteins such as DCXR, GBP6, HPR, PSME1, DNAJA4, GSTP1, DDT, KRT7, CTNNA1, and KRT18 are proteins that best predict sex differences.
[Reference] (https://plos.figshare.com/articles/figure/Variable_importance_plot_mean_decrease_accuracy_and_mean_decrease_Gini_/12060105/1)

**Mean decrease Gini**

The mean decrease in Gini is based on the decrease in the Gini impurity index, and it measures how each predictor contributes to the purity of the nodes in the decision trees. In other words, the mean decrease Gini score tells us how important a feature is in making decisions in the dataset. The higher the score, the more important the feature is in making decisions and splitting the data. Our data shows that HPR, DCXR, PSMB7, KRT15, HYOU1, and GBP6 are essential proteins for making decisions in this dataset.
[Reference] (https://www.analyticsvidhya.com/blog/2021/03/how-to-select-best-split-in-decision-trees-gini-impurity/)

**Random Confusion Matrix**

```{r}
# Create folds for cross-validation
folds <- createFolds(pre.dataset$Sex, k = 5)

# Initialize empty vectors for storing results
accuracy <- c()
sensitivity <- c()
specificity <- c()

# Loop through each fold
for (i in 1:length(folds)) {
  # Split data into training and test sets
  train_data <- pre.dataset[-folds[[i]], ]
  test_data <- pre.dataset[folds[[i]], ]
  
  # Train random forest model on training data
  rf_model <- randomForest(Sex ~ ., data = train_data, importance = TRUE)
  
  # Make predictions on test data
  predictions <- predict(rf_model, test_data)
  
  # Calculate confusion matrix
  cm <- confusionMatrix(predictions, test_data$Sex)
  cm
  
  # Calculate and store metrics
  accuracy[i] <- cm$overall["Accuracy"]
  sensitivity[i] <- cm$byClass["Sensitivity"]
  specificity[i] <- cm$byClass["Specificity"]
}

# Print mean and standard deviation of metrics
cat("Mean Accuracy:", mean(accuracy), "±", sd(accuracy), "\n")
cat("Mean Sensitivity:", mean(sensitivity), "±", sd(sensitivity), "\n")
cat("Mean Specificity:", mean(specificity), "±", sd(specificity), "\n")

```

```{r}
Mean Accuracy: 0.4933333 ± 0.2349941 
Mean Sensitivity: 0.2666667 ± 0.2527625 
Mean Specificity: 0.65 ± 0.3555122 
```

By comparing the mean accuracy of the Random Forest Model to the Decision Tree, we find that the Decision Tree achieved higher accuracy with this dataset. This indicates that, for this specific dataset, a Decision Tree would be the more appropriate model to use for predicting our data. Considering the relatively low number of observations in the dataset (27), we may not have sufficient data to support the use of a robust Random Forest model.
