4.2: Supervised Machine Learning 

This training module was developed by Oyemwenosa N. Avenbuan, Alexis Payton, and Dr. Julia E. Rager

Spring 2023

## Machine Learning Review

Machine Learning is a field of study in computer science that involves creating algorithms(a set of instructions that perform a specific task on a given dataset). Machine Learning is a scientific approach that enables researchers to create models that can automatically adapt to new and unforeseen situations) capable of improving automatically through experience and data.

In other words, instead of being explicitly programmed to perform a task, a machine learning algorithm is designed to learn from examples and data, allowing it to adapt and improve over time. This approach is particularly useful for tasks that are too complex or difficult to be solved using traditional programming methods.

Through Machine Learning, scientists can:

1\. Create a model that adapts to new circumstances that the scientist did not envision.(add and example)

2\. Detect patterns in large and complex datasets. (add an example)

3\. Evaluate the effectiveness of these patterns. (add an example)

4\. Make informed decisions about how to improve their models. (add an example)

Ultimately, Machine Learning is a powerful tool that enables researchers to analyze data more effectively, make more accurate predictions, and develop more advanced systems that can learn and evolve over time.

## Types of Learning

In the field of Machine Learning, there are two broad types of learning: supervised learning and unsupervised learning.

**Supervised learning** involves training a machine learning model using a labeled dataset, where each example is associated with a known outcome or target variable. The model is then able to learn how to predict the outcome for new, unseen examples based on the patterns and relationships it identifies in the data.

![image](https://user-images.githubusercontent.com/96756991/228436976-ec715a4f-575f-4718-89d6-6233695fcd7f.png)
Created with BioRender.com

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

## The Math Behind Models

Linear algebra is a branch of mathematics that studies the properties and behavior of mathematical objects called vectors and matrices.

Vectors are essentially a list of numbers that represent quantities that have both magnitude and direction, such as velocity or force.

Matrices, on the other hand, are like tables of numbers arranged in rows and columns, which can be used to represent data or perform operations on vectors.

One of the key concepts in linear algebra is the notion of linear transformations, which are functions that preserve the properties of vectors and matrices, such as their linearity and dimensionality. Another important aspect of linear algebra is matrix multiplication, addition, and subtraction, which are operations used to manipulate matrices and are fundamental to many mathematical applications, including neural networks.

Therefore, understanding linear algebra is essential for anyone working in fields that rely on data analysis and modeling, such as machine learning and artificial intelligence.

[Reference](https://www.britannica.com/science/linear-algebra)

## Background on Data 

This dataset is the result of a human-exposure study, involving 27 participants, which investigated molecular alterations in sputum before and after exposure to smoldering red. Throughout the study, biological samples were collected from participants both before and after exposure. Demographic information was also gathered and the proteomic signatures in these samples were analyzed using high-resolution mass spectrometry. For the purpose of this training, we will be focusing solely on the pre-exposure data.

This project meets the UNC’s IRB approval process (IRB# 13–3076 or 18-1895 or 18-2196)

## Questions to answer 

Based on the pre-exposure data, we want to address two questions centered around sex-differences. They include:

1. Can we predict sex based on protein expression?
2. Which proteins best predict sex? 

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
#set the working directory 
pre.dataset <- na.omit(read.csv("Proteomics_Imputed_PreExposureSubjects.csv")) %>%
  select(-SubjectID, -Race, -Ethnicity, -Age, -BMI) %>%
  drop_na() %>%
  mutate(Sex = ifelse(Sex == "M", 1, 2))
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

**KNN Recap**

Before we create a decision tree and random forest, we want to mention a type of algorithim that was mentioned [previously]([url](https://uncsrp.github.io/Data-Analysis-Training-Modules/machine-learning-and-predictive-modeling.html#k-means-analysis)) KNN (K-nearest Neighbors). The concept of KNNs were explained in here, but there are multiple ways to train and test your data. It is important to be able to justify why the model you are using is better than the others that exist. By running KNN, we will get a better sense of if this model is a good fit for the type of data below.

```{r}
#make this reproducible
set.seed(15)

#splitting the data into training and testing sets 
sex_df_index = createDataPartition(pre.dataset$Sex, p = 0.6, list = FALSE)
sex_train = pre.dataset[sex_df_index,]
sex_test = pre.dataset[sex_df_index,]


#training the algorithm
knn_sex = train(Sex ~., data = pre.dataset, method = "knn", tuneLength = 8 , preProcess = c("center", "scale"))

#printing overall accuracies for all tuning parameter 
knn_sex$results[,1:2] #best model: k = 23

#testing algorithm on test set and printing accuracys with the best tuning parameter 
sex_test$sex_pred = predict(knn_sex, newdata = sex_test)
per_class_accuracy <- rep(NA, length(levels(sex_test$Sex)))
for (i in 1:length(per_class_accuracy)){
  per_class_accuracy[i] <-
    sex_test %>%
    filter(Sex == levels(Sex)[i]) %>%
    summarise(accuracy = sum(sex_pred ==levels(Sex)[i])/n()) %>%
    unlist()
  names(per_class_accuracy)[i] <- paste0(levels(sex_test$Sex)[i], "Accuracy")
}

per_class_accuracy

#Note to Alexis: when running the KNN algorithm, I was having trouble finding a K that had a high accuracy. Every time I ran it, the most optimal (highest accuracy) K was 19 however, it exceeded the number of observations in the data set. 
```

Based on the results displayed above, it is apparent that the accuracy of the model is low. This indicates that the algorithm is recommending the creation of more clusters than there are samples available. My analysis suggests that the dataset may be too small and noisy, making it challenging to form relevant clusters. I have decided to explore other models, such as decision trees and random forests, to gain deeper insights into the data.

Now that we have ruled out using the KNN algorithim, we iwll move onto creating a decision tree and random forest and be able to test if these are better models for the dataset.

Now that we have eliminated the KNN algorithm as a viable option, we will proceed to develop a decision tree and random forest. These models will enable us to evaluate their effectiveness in analyzing the dataset and determine whether they are better suited for our task.

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

In this module, we will start by creating a decision without any pruning.

[Reference](https://www.mastersindatascience.org/learning/machine-learning-algorithms/decision-tree/)

```{r}

_Creating a decision tree (no pruning)_

#set up for reproducibility 
set.seed(15)

#splitting data into training and testing sets 
sex_data_index = createFolds(pre.dataset$Sex, k = 5) #K in Cross Validation is usually 5 or 10 
errors = data.frame()
for (i in 1:length(sex_data_index)){
  sex_train = pre.dataset[-sex_data_index[[i]],]
  sex_test = pre.dataset[sex_data_index[[i]],]
  
  reg_tree = rpart(Sex ~., data = sex_train)
  
  #predicting on test set 
  sex_test$sex_pred = predict(reg_tree, newdata = sex_test)
  
  #calculating MSE (add definition)
  error_values = postResample(sex_test$sex_pred, sex_test$Sex)
  
  #adding values to data frame created above 
  errors = rbind(errors, error_values[1]^2)
}

colnames(errors) = c("MSE")
  
#taking averages/sd by method
unpruned_errors = errors %>%
 summarise("CV Error" = mean(MSE), "Std Error" = sd(MSE))

unpruned_errors %>%
  flextable()
  
```

_Pruning the decision tree_

```{r}
#Predicting Sex using a decision tree but we 

#set up for reproducibility 
set.seed(15)

#splitting data into training and testing sets 
sex_data_index1 = createFolds(pre.dataset$Sex, k = 5) #K in Cross Validation is usually 5 or 10 
errors1 = data.frame()
for (i in 1:length(sex_data_index1)){
  sex_train1 = pre.dataset[-sex_data_index1[[i]],]
  sex_test1 = pre.dataset[sex_data_index1[[i]],]
  
  #omit missing imputations
  sex_test1 <- na.omit(sex_test1)

  # Convert the outcome variable to a factor with two levels
  sex_train1$Sex <- factor(sex_train1$Sex, levels = c("1", "2")) # 0 is male, 1 is female

  #now pruning the tree (in this method a new alpha is used everytime)
  reg_tree_pruned <- train(Sex ~., data = sex_train1, method = "rpart", 
                           trControl = trainControl("cv", number = 10),
                           tuneGrid = data.frame(cp = seq(0, 0.1, 0.01)), 
                           control = rpart.control(cp = 0), tuneLength = 10)
  
  #predicting on test set 
  sex_test1$sex_pred = predict(reg_tree_pruned, newdata = sex_test1, type="raw")
  
  # Convert the predicted outcome to a factor variable
  sex_test1$sex_pred <- factor(sex_test1$sex_pred, levels = c("1", "2")) # 1 is male, 2 is female
  
  # Convert actual values to factor and then to numeric
  sex_test1$Sex <- factor(sex_test1$Sex, levels = c("1", "2")) # 0 is male, 1 is female
  sex_test1$Sex <- as.numeric(sex_test1$Sex)
  
  # Convert predicted values to factor and then to numeric
  sex_test1$sex_pred <- factor(sex_test1$sex_pred, levels = c("1", "2")) # 0 is male, 1 is female
  sex_test1$sex_pred <- as.numeric(sex_test1$sex_pred)
  
  #calculating MSE
  error_values = postResample(sex_test1$sex_pred, sex_test1$Sex)
  
  #adding values to df
  errors1 = rbind(errors1, error_values[1]^2)
}

colnames(errors1) = c("MSE")

#taking averages/sd by method 
pruned_errors = errors1 %>%
  summarise("CV Error" = mean(as.numeric(MSE), na.rm = TRUE), "Std Error" = sd(as.numeric(MSE), na.rm = TRUE)) #I was getting an error while using summarize

pruned_errors %>%
  flextable()

```

**Random Forest**

Random Forest combines the outcome of many decisions trees and creates a single result. 

![image](https://user-images.githubusercontent.com/96756991/228436169-c7fe932a-9aa4-41e2-b56c-9a8830b834a8.png)
Created with BioRender.com

[Reference](https://www.ibm.com/topics/random-forest) 

Random Forest Code (in progress)

```{r}
rf_classification = function(pre.dataset, outcome, pred_outcome) {
#setting for reproducibility
set.seed(15)

#splitting data into training and testing sets 
sex_data_index = createFolds(pre.dataset$Sex, k = 5) #K in Cross Validation is usually 5 or 10 


errors = data.frame()
for (i in 1:length(sex_data_index)){
  sex_train = pre.dataset[-sex_data_index[[i]],]
  sex_test = pre.dataset[sex_data_index[[i]],]
  
  ntree_values = c(50, 250, 500) #number of trees
  p = dim(pre.dataset)[2] - 1 #number of variables in dataset
  mtry_values = c(p/2, sqrt(p), p)
  
  reg_rf_pred_tune = list()
  rf_error = list()
  rf_error_df = data.frame()

  metrics = data.frame()
  variable_importance_df = data.frame()
  roc_objects = c()
  threshold_data = data.frame()
  
  # check for missing values in mtry_values and replace them with median
  #mtry_values[is.na(mtry_values)] <- median(mtry_values, na.rm = TRUE)

   for (j in 1:length(ntree_values)){
      for (k in 1:length(mtry_values)){
          reg_rf_pred_tune[[k]] = randomForest(as.formula(paste0(outcome, ~.)), data = sex_train1, 
                                                     ntree = ntree_values[j], mtry = mtry_values[k])
          rf_OOB_errors[[k]] = data.frame("Tree Number" = ntree_values[j], "Variable.Number" = mtry_values[k], 
                                       "OOB_errors" = reg_rf_pred_tune[[k]]$err.rate[ntree_values[j],1])
          rf_error_df = rbind(rf_error_df, rf_OOB_errors[[k]])
  }
  }

   # finding the lowest OOB error using best number of predictors at split and refitting OG tree
        best_oob_errors <- which(rf_error_df$OOB_errors == min(rf_error_df$OOB_errors))
        
        # many models have the lowest errors, so now selecting based on largest number of trees grown
        best_oob_df = rf_error_df[best_oob_errors, ]
        largest_trees = which(best_oob_df$Tree.Number == max(best_oob_df$Tree.Number))
        
        # still duplicate models w/ the lowest errors, so now selecting based on # of predictors = sqrt(p)
        best_tree_df = best_oob_df[largest_trees, ]
        default_predictor_number = which(best_tree_df$Variable.Number == max(best_tree_df$Variable.Number))
}
}
```
