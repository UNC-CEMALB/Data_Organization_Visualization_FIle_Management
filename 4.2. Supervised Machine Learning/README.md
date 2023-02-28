# TAME Module 4.2: Supervised Machine Learning 
Machine Learning Recap 




Machine Learning is a field of study in computer science that involves creating algorithms(a set of instructions that perform a specific task on a given dataset). Machine Learning is a scientific approach that enables researchers to create models that can automatically adapt to new and unforeseen situations) capable of improving automatically through experience and data. 



In other words, instead of being explicitly programmed to perform a task, a machine learning algorithm is designed to learn from examples and data, allowing it to adapt and improve over time. This approach is particularly useful for tasks that are too complex or difficult to be solved using traditional programming methods.

Through Machine Learning, scientists can:
  

1. Create a model that adapts to new circumstances that the scientist did not envision.(add and example)
 

2. Detect patterns in large and complex datasets. (add an example)


3. Evaluate the effectiveness of these patterns. (add an example)
  

4. Make informed decisions about how to improve their models. (add an example)



Ultimately, Machine Learning is a powerful tool that enables researchers to analyze data more effectively, make more accurate predictions, and develop more advanced systems that can learn and evolve over time.

Types of Learning 

In the field of Machine Learning, there are two broad types of learning: supervised learning and unsupervised learning.



Supervised learning involves training a machine learning model using a labeled dataset, where each example is associated with a known outcome or target variable. The model is then able to learn how to predict the outcome for new, unseen examples based on the patterns and relationships it identifies in the data.



Unsupervised learning, on the other hand, involves training a machine learning model on an unlabeled dataset, where the outcome or target variable is unknown. The model is then tasked with identifying patterns and structures in the data, such as clusters of similar examples or underlying relationships between variables.



It's worth noting that there are also other types of learning in Machine Learning, such as semi-supervised learning and reinforcement learning, which combine elements of both supervised and unsupervised learning.



Overall, the distinction between supervised and unsupervised learning is an important concept in Machine Learning, as it can inform the choice of algorithms and techniques used to analyze and make predictions from data.

Training Your Model

In Machine Learning, before we can effectively use algorithms to analyze data, we first need to train them. This involves selecting a smaller portion or subset of data, known as training data, to teach the algorithm how to identify distinct patterns. By recognizing these patterns, the algorithm can accurately classify specific cases within a larger and more complex dataset. The process of training an algorithm is essential for enabling it to learn and improve over time, allowing it to make more accurate predictions and better adapt to new and changing circumstances. Ultimately, the effectiveness of a machine learning model depends on the quality and relevance of its training data.



In Machine Learning, the process of developing a model involves dividing the data into three distinct sets: 

  

1. the training set: a subset of the data that is used to fit the model. Essentially, the model learns from this data and uses it to identify patterns and relationships in the data.
  

2. the validation set: a sample of data that is used to evaluate the model's fit in an unbiased way. It helps develop the model by fine-tuning its parameters and optimizing its performance. This is akin to pop-quizzes that help students improve their understanding and performance.
  

3. test set: a sample of data that is used to provide an evaluation of the final model's fit on the training set. This is the model's final exam, as it provides an objective assessment of the model's ability to generalize to new, unseen data.

It's important to note that the test set should only be used once, after the model has been fully developed and fine-tuned on the training and validation sets. Using the test set multiple times during the development process can lead to overfitting, where the model performs well on the test data but poorly on new, unseen data.



Overall, the process of dividing the data into training, validation, and test sets is a crucial step in developing accurate and reliable machine learning models.

(https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7)



The Math Behind Models


Linear algebra is a branch of mathematics that studies the properties and behavior of mathematical objects called vectors and matrices. 

Vectors are essentially a list of numbers that represent quantities that have both magnitude and direction, such as velocity or force. 

Matrices, on the other hand, are like tables of numbers arranged in rows and columns, which can be used to represent data or perform operations on vectors.



One of the key concepts in linear algebra is the notion of linear transformations, which are functions that preserve the properties of vectors and matrices, such as their linearity and dimensionality. Another important aspect of linear algebra is matrix multiplication, addition, and subtraction, which are operations used to manipulate matrices and are fundamental to many mathematical applications, including neural networks.



Therefore, understanding linear algebra is essential for anyone working in fields that rely on data analysis and modeling, such as machine learning and artificial intelligence.


(https://www.britannica.com/science/linear-algebra)

Specific Model: Decision Trees 

Now that we have discussed some of the basics of machine learning, we will focus the rest of our training on a specific model: Decision Trees and Random Forests. 
