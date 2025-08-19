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
  1. [Baseline Feedforward Neural Network Classifier](#nn-task1)
  2. [Convolutional Neural Network with Max-pooling](#cnn-mp-task1)
  3. [Convolutional Neural Network with Multi-head Attention Layer](#cnn-ma-task1)
  4. [DistillBERT Transformer Model](#distillbert-task1)
- Task 2
  1. [T5-small](#t5-task2)
  2. [T5-base](#t5-task2)
 
- Result [Here](#result) 

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

### Neural Network Model tested for Performance Comparison
<a name="nn-task1"></a>
#### 1. Fully Connected Neural Network Classifier 'NN_task1.py'
This model is the baseline approach for the task. The simple structure is designed to serve as foundation and to better illustrate performance difference for other models designed in this research for this task. The architecture is as follows:
- Input embedding layer (random embedding / word2vec embedding)
- Single hidden layer of 128 neurons with ReLU activation
- Drop out layer
- Fully-connected output layer of SoftMax activation for classification

Cross entropy loss is selected as the loss function and L2 regularization is applied to address overfitting.  
  
Fine-tuning Experiment:
| Hyper-parameters | Value range | Best Value from Cross Validation |
| ------------- | ------------- | ------------- |
| Batch Size  | 32,64 | 32 |
| Learning rate  | 0.01,0.05,0.001, 0.005,0.0001 | 0.001 |
| L2 Regularization | 0.01,0.05,0.001,0.005,0.0001 | 0.005 |
| Dropout Rate | ∈ [0.2,0.7] | 0.5 |
  
<a name="cnn-mp-task1"></a>
#### 2. Convolutional Neutral Network with Max-pooling ('CNN_original.py')
Convolutional Neural Network (CNN) is built to leverage on its ability to capture local relationships between words. Differing from previous simple NN classifier, it can capture the key phrases or words combination that might be vital for sentiment analysis. 

In addition, as the task aims to identify the type of spoilers instead of basic binary sentiment analysis (i.e. positive/negative tone), CNN is selected to examine if patterns in neighboring words can facilitate the classification. (e.g. numbers following noun patterns, like “7 habits”, “3 places” which can indicate a “multi” type of spoiler should be classified). 
   
This is inspired by the CNN architecture constructed in [Kim's Research on CNN in 2014](https://arxiv.org/abs/1408.5882), which leveraged on applying parallel kernel size parallelly to capture more informative patterns.

Structure:
- Input embedding layer (random embedding / word2vec embedding)
- Parallel convolutional layers with varying filter sizes
- Max-pooling layer
- 2nd Drop out layer
- Fully-connected output layer for classification

Fine-tuning Experiment:
| Hyper-parameters | Value range | Best Value from Cross Validation |
| ------------- | ------------- | ------------- |
| Batch Size  | 32,64,128  | 64 |
| Learning rate  | 0.01,0.05,0.001, 0.005,0.0001 | 0.005 |
| L2 Regularization | 0.01,0.05,0.001,0.005,0.0001 | 0.01 |
| Dropout Rate | ∈ [0.2,0.7] | 0.5 |
| Kernel Size | 2,3,4,5,6 | [2,3,4,5] |

  
<a name="cnn-ma-task1"></a>
#### 3. Convolutional Neural Network with Multi-head Attention layer ('CNN_attention.py' & 'CNN_multiattention.py')
The 2nd CNN is the modification of 1st parallel layer design. Instead of using Max-pooling strategy, an Attention layer design is used to replace the Max-pooling to learn the position importance of input sequences. The structure is as follows:

Structure:
- Input embedding layer (random embedding / word2vec embedding)
- Parallel convolutional layers with varying filter sizes
- Attention layer
- 2nd Drop out layer
- Fully-connected output layer for classification

In addition, both single-headed attention layer and multi-head attention layer are tested in 2nd structure separately to compare the result.  

Fine-tuning Experiment:
| Hyper-parameters | Value range | Best Value from Cross Validation |
| ------------- | ------------- | ------------- |
| Batch Size  | 32,64,128 | 64 |
| Learning rate  | 0.01,0.001,0.005 | 0.005 |
| L2 Regularization | 0.01,0.05,0.001 | 0.005 |
| Dropout Rate | ∈ [0.2,0.5] | 0.3 |
| Kernel Size | 2,3,4,5 | [2,3,4,5] |

<a name="distillbert-task1"></a>
#### 4. DistilBERT Transformer Model ('DistilBERT_Task1.ipynb')
DistilBERT Transformer Model is a lighter version of BERT (Bidirectional Encoder Representations from Transformers), developed by Hugging Face. It is a pre-trained model adopted from Hugging Face library and fine-tuned for this classification task. Compared with BERT model with 110M parameters, DistilBERT has a fewer, 66M parameters but offers a more efficient computation. 

Fine-tuning Experiment:
| Hyper-parameters | Value range | Best Value from Cross Validation |
| ------------- | ------------- | ------------- |
| Batch Size  | 32,64 | 64 |
| Learning rate  | 1e-5,3e-5,4e-5,8e-5,1e-4,3e-4,5e-3 | 0.005 |
| Weight Decay (equivalent to L2 reg.) | 0.01,0.05,0.001,0.005,0.0001 | 0.005 |


## Task 2 - Spoiler Summarization
Task 2 refers to the spoiler generation task based on the provided text. This task aims to extract the key message in the post and summarize it accordingly in short sentences.

The text are pre-processed and cleaned with identical approaches stated in section 3.2.1. The following prompt is used for all models as follows:

> Question: What is the key spoiler that '{title}' is inferring in passage? Passage: {text}

where ‘title’ refers to the text title provided in the dataset, and ‘text’ refers to the extracted paragraph in the dataset. Models are thus trained with a target spoiler sentence provided.   

### Transformer Model tested for Performance Comparison
<a name="t5-task2"></a>
#### Text-to-Text Transfer Transformer (T5)   ('T5_Task2.ipynb')
Due to limited size of data set (i.e. 3200 samples), training a transformer model from scratch is not viable for this task. Therefore, this research leverages pre-trained models available in Hugging Face library and performed fine tuning on model parameters to achieve the task.  

Text-To-Text Transfer Transformer (T5) is a encoder-decoder transformer designated for NLP tasks. It is selected as the model for this task for its relatively lighter computation costs and GPU requirements. 

Other models like Pegasus developed by Google are considered but not included in this research due to computational power limitations.

Two T5 models of different sizes are tested for performance comparison in this research as follows:
- T5-small, with 60M parameters
- T5-base, with 220M parameters

Fine-tuning Experiment (for both T5-small & T5-base):
| Hyper-parameters | Value range | Best Value from Cross Validation |
| ------------- | ------------- | ------------- |
| Batch Size  | 32,64 | 16 |
| Learning rate  | 1e-5,3e-5,4e-5,8e-5,1e-4,3e-4,5e-3 | 5e-3 |
| Weight Decay (equivalent to L2 reg.) | 0.01,0.05,0.001,0.005,0.0001 | 0.8 |

<a name="result"></a>
## Result
All combinations of hyper-parameter settings are tested and evaluated. The full log of training process can be found in supplementary documents. The best performance of each model is as follows:  
#### Task 1

| Model |  Best F1 score in Submission |
| -------------  | ------------- |
| Simple NN Classifier  | 0.20850 |
| CNN with Parallel layer & Max-pooling  | 0.58870 |
| CNN with Parallel layer & Attention Layer | 0.54060 |
| CNN with Parallel layer & Multi-head Attention Layer | 0.20850 |
| DistilBERT Transformer Model | 0.20850 |

#### Task 2
| Model |  Best METEOR score in Submission |
| -------------  | ------------- |
| T5-small  | 0.16860 |
| T5-base | 0.36230 |

## Discussion
#### Task 1
For pre-processing in general, using the title of the post instead of a full extracted paragraph gives a better result. Also, keeping stop words in general shows better results. It can be perceived by the length of text input, in which shorter titles enable the models to handle a significantly smaller vocabulary size and focus more on important words. This can be understood from a human perspective. The title of a clickbait post usually provides significant hints about the type of spoilers. In addition, 

For example, in validation data with id "390958297249247233", the title is "What weighs 2032 pounds is orange and can be seen in New York this week?”. The word “What” clearly indicates the spoiler should refer to an object, which infers that a predicted classification should be “phrase”. This can be validated by the corresponding tags/labels of the set and the corresponding spoiler “World’s largest pumpkin”. Therefore, it helps explain why using “title” would have a generally better result than using the full “text”.

For model performance, based on the results, the CNN with a Parallel layer & Max-pooling has the best classification performance with an F1 score of 58.870%. It has a slightly better performance than CNN with an Attention layer. The performance difference may stem from the input text choice. As titles are significantly shorter than the full text, max-pooling with a kernel size of 2 to 5 should be capable of capturing the important words regardless of the position. Therefore, it may help explain why the Max-pooling mechanism, which focuses on only the presence of important features, may outperform the Attention layer, which focuses on important words in varying positions. In addition, Max-pooling also has a much shorter computation time compared to both a single Attention layer and a Multi-head Attention layer, which allows an efficient training process. 

#### Task 2
Both models used the full text and title combined in the prompt as the input. 
Based on model results, T5-base has a better performance with a METEOR score of 36.230%, which is almost double the score of T5-small. The difference in parameter size (220M vs. 60M) suggests that the larger the model, the better it can capture the key message of text.

On fine-tuning the influence of the model, it can be examined with the quality of the spoiler generated. For example, on data with id “0” in the test set, it refers to a man donating his kidney for a friend to save his life. 

In the initial model without tuning, it produces a prediction as follows: “Graham mcmillanhe decided to surprise his friend with the good news and did”, which is barely a meaningful sentence. 

However, with slight tuning on the learning rate to 1e-4 and 3e-4 from 1e-5, it gradually improved the response to generate as follows: “he could donate one of his”, “he is a match”. Eventually, in the best performing setting and after addressing overfitting, the model produced the response as “a kidney”. Therefore, it can be observed that the fine-tuning process is successful in producing a more reasonable response to humans and thus results in a better METEOR score. 


## Conclusion

For text classification, a Convolutional Neural Network with Max-pooling performed the best among all tested models. However, the performance of CNN with the Attention Layer has a second-best score in comparison with less than 4% in F1 score differences. It suggests that with further tuning and modification of the attention layer structure, Attention layers may perform better with their focus on the positional importance of words and high context sensitivity. 

For text generation and summarization, T5-base outperformed T5-small, whose performance differences can be concluded by the varying size of parameters. Larger models like T5-base with 220M parameters can better capture the gist of text and produce better responses. In addition, by observing the quality of response in train/validation/test data during fine-tuning, hyper-parameter tuning will significantly improve model performance and generation quality.

## Limitation & Further Ideas
One major limitation of this research is computation power and memory management, specifically in text generation. This research has attempted to test on other pre-trained models, including Google Pegasus, to compare performance with T5-base. Yet, these models usually require larger and stronger computational power and GPU performance, which is not available for this research. Thus, model performance can be further improved with a stronger GPU and better memory management to accommodate larger models within limited resources. 
