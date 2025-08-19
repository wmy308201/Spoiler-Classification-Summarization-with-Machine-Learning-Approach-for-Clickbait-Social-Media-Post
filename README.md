# Spoiler Classification & Summarization with Machine-Learning Approach for Clickbait Social Media Post

## Background
This project is a submission for a Kaggle Challenge hosted by University of Waterloo, resembling the "Clickbait Challenge at SemEval 2023 - Clickbait Spoiling", with modification on the size of test dataset. The challenge link can be found below:
- Spoiler Classification: https://www.kaggle.com/competitions/clickbait-detection-msci641-s23
- Spoiler Summarization: https://www.kaggle.com/competitions/task-2-clickbait-detection-msci-641-s-25
- Original Challenge at SemEval 2023: https://pan.webis.de/semeval23/pan23-web/clickbait-challenge.html 

## Abstract
This project aims to perform two tasks related to Clickbait post spoiler generation: a classification problem of determining the required spoiler type for a post, and a text generation task of generating spoilers for a post. I have attempted to perform the two tasks by developing neural networks of different architectures to compare model performance.   
  
In task 1, a Convolutional Neural Network (CNN) with a parallel convolutional layer and max-pooling outperformed another model with an Attention mechanism, achieving an F1 score of 58.87%.  
  
In task 2, the T5-base model achieved the best METEOR score with 36.23% for short spoiler generation. This paper further discussed the possible explanation of model performance differences based on architecture, pre-processing procedure and fine-tuning progress.

For specific model developed for each task, please refer to table below:
- Task 1
  1. [Baseline Feedforward Neural Network Classifier](#test)

## Data Set
The dataset used in this research is adopted from Kaggle Clickbait Challenge 2023, which consists of 3,200 posts in training and 400 posts in validation set.    
For each entry in the training and validation dataset, the following fields are available:
- uuid: The uuid of the dataset entry.
- postText: The text of the clickbait post which is to be spoiled.
- targetParagraphs: The main content of the linked web page to classify the spoiler type (task 1) and to generate the spoiler (task 2). Consists of the paragraphs of manually extracted main content.
- targetTitle: The title of the linked web page to classify the spoiler type (task 1) and to generate the spoiler (task 2).
- targetUrl: The URL of the linked web page.
- humanSpoiler: The human generated spoiler (abstractive) for the clickbait post from the linked web page. This field is only available in the training and validation dataset (not during test).
- spoiler: The human extracted spoiler for the clickbait post from the linked web page. This field is only available in the training and validation dataset (not during test).
- spoilerPositions: The position of the human extracted spoiler for the clickbait post from the linked web page. This field is only available in the training and validation dataset (not during test).
- tags: The spoiler type (might be "phrase", "passage", or "multi") that is to be classified in task 1 (spoiler type classification). This field is only available in the training and validation dataset (not during test).

## Task 1 - Spoiler Classfication
Task 1 refers to the classification task which aims to identify the types of spoiler that a post is needed. Based on the content, 3 major types are set to be identified:
1.	‘passage’: indicating a passage or paragraph of content is needed to conclude or spoil the key message of corresponding post
2.	‘phrase’: indicating a phrase of one or few words is needed to conclude or spoil the key message of corresponding post
3.	‘multi’: indicating a mix of phrases, numbers or short passages is needed to conclude or spoil the key message of corresponding post

Therefore, this task is to classify the selected text content into 3 target class labels. 
#test
### Neural Network Model tested for Performance Comparison
#### Fully Connected Neural Network Classifier
This model is the baseline approach for the task. The simple structure is designed to serve as foundation and to better illustrate performance difference for other models designed in this research for this task. The architecture is as follows:
- Input embedding layer (random embedding / word2vec embedding)
- Single hidden layer of 128 neurons with ReLU activation
- Drop out layer
- Fully-connected output layer of SoftMax activation for classification

Cross entropy loss is selected as the loss function and L2 regularization is applied to address overfitting. 

### 2. Convolutional Neutral Network
Convolutional Neural Network (CNN) is built to leverage on its ability to capture local relationships between words. Differing from previous simple NN classifier, it can capture the key phrases or words combination that might be vital for sentiment analysis. 

In addition, as the task aims to identify the type of spoilers instead of basic binary sentiment analysis (i.e. positive/negative tone), CNN is selected to examine if patterns in neighboring words can facilitate the classification. (e.g. numbers following noun patterns, like “7 habits”, “3 places” which can indicate a “multi” type of spoiler should be classified). 

Two types of CNN architecture are constructed. The 1st structure is inspired by the CNN architecture constructed in [Kim's Research on CNN in 2014](https://arxiv.org/abs/1408.5882), which leveraged on applying parallel kernel size parallelly to capture more informative patterns.



