#!/usr/bin/env python3
import argparse
import json
import string
import random
import re
import string
import pickle
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

debug = True
retrain = True
stopword_removal = False
ref_x = 'title'  # title, text

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Hyperparameters
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
L2_REG        = 0.005
DROPOUT_RATE  = 0.2
EARLY_STOPPING_PATIENCE = 5
ACTIVATION = 'relu'

tuning_param = [HIDDEN_SIZE,BATCH_SIZE,NUM_EPOCHS,LEARNING_RATE]    

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(
        description='This is a baseline for task 1 that predicts that each clickbait post warrants a passage spoiler.')
    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The classified output in jsonl format.', required=False)
    return parser.parse_args()


def run_baseline(input_file, output_file):
    
    train_data_dir = '../data/train.jsonl'
    val_data_dir = '../data/val.jsonl'

    train, train_labels, _ = load_preprocess_data(train_data_dir, test_flag = False)
    val, val_labels, _ = load_preprocess_data(val_data_dir, test_flag = False) 


    if debug:
        print("Training data loaded.")
        print(f"Number of training samples: {len(train)}")
        # print(f"Sample text: {train[3]}")
        print(f"S`ample label: {train_labels[0:5]}")

    #create embed models
    embed_md = w2v_train(train, vector_size=100, window=5, min_count=5, workers=4, epochs=10, sg=1)
    
    if stopword_removal:
        save_model(embed_md, 'embed_md_ns.pkl')
    else:
        save_model(embed_md, 'embed_md.pkl')
    

    # create embeddings
    train_emb = embed(train, embed_md)
    val_emb = embed(val, embed_md)

    # create tensorflow & dataloader
    train_set = TensorDataset(torch.tensor(train_emb).double(), torch.tensor(train_labels))
    val_set = TensorDataset(torch.tensor(val_emb).double(), torch.tensor(val_labels))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    
    
    input_size = len(train_emb[0])  # Size of Word2Vec embeddings
    num_classes = len(set(train_labels))  # Number of unique labels
    hidden_size = HIDDEN_SIZE  # Size of hidden layer
    model = NNClassifier(input_size,
                        hidden_size,
                        num_classes,
                        activation = ACTIVATION,
                        dropout_rate = DROPOUT_RATE)
    history, val_acc = NN_train(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, learning_rate= LEARNING_RATE, l2_reg= L2_REG)
    
    torch.save(model.state_dict(), "NeuralNet.pth")
    # ==================================================================================================
    # ==================================================================================================


    with open(input_file, 'r', encoding="utf8") as inp, open(output_file, 'w') as out:
        # Read input data
        
        test, _ , test_id = load_preprocess_data(input_file, test_flag = True)

        test_emb = embed(test, embed_md)

        test_set = TensorDataset(torch.tensor(test_emb).double(),torch.tensor(test_emb).double())
        test_loader = DataLoader(test_set, batch_size= BATCH_SIZE, shuffle = False)
        
        model = NNClassifier(input_size,
                            hidden_size,
                            num_classes,
                            activation = ACTIVATION,
                            dropout_rate = DROPOUT_RATE)
        model.load_state_dict(torch.load('NeuralNet.pth',weights_only = True))
        test_pred = NN_evaluate(model, test_loader)
        if debug:
            print("test predicted")
        # 1 : passage, 2: phrase, 0: multi
        for i in range(0,len(test_pred)):
            if test_pred[i] == 1:
                res = 'passage'
            elif test_pred[i] == 2:
                res = 'phrase'
            else:
                res = 'multi'
            prediction = {'id': test_id[i],'spoilerType': res}
            out.write(json.dumps(prediction)+ '\n')
        if debug:
            print("completed")

def load_preprocess_data(file_path, test_flag):
    with open(file_path, 'r',encoding="utf8") as inp:
        # Read input data
        data = {
            'id': [],
            'title': [],
            'text': [],
            'label': []
        }
        for i in inp:
            i = json.loads(i)
            if test_flag:
                data['id'].append(i['id'])
            else:
                data['id'].append(''.join(i['postId']))
            data['title'].append(''.join(i['postText']))
            data['text'].append(''.join(i['targetParagraphs']))
            if not test_flag:
                data['label'].append(''.join(i['tags']))
        # random.seed(156)
        # random.shuffle(train_data)

        X_raw = [item for item in data[ref_x]]
        y_raw = [item for item in data['label']]
        data_id = data['id']

        X = [tokenize(text) for text in X_raw]
        if stopword_removal:
            X = [remove_stopwords(tokens, load_stopword()) for tokens in X]

        if debug:
            print(f"Sample label: {y_raw[0:5]}")

        le = LabelEncoder()
        y = le.fit_transform(y_raw)

    return X, y, data_id

class NNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation='relu', dropout_rate=0.5):
        super(NNClassifier, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Tanh()

        self.dropout = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(hidden_size, num_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

def NN_train(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, l2_reg=0.001):

    # Initialize tracking variables
    best_val_accuracy = 0.0
    best_model_state = None
    early_stopping_patience = EARLY_STOPPING_PATIENCE
    patience_counter = 0

    # Initialize history dictionary
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(torch.float64)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            # Move batch to device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update metrics
            train_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        # Calculate training metrics
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                # Move batch to device
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                # Update metrics
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
                # Store predictions and labels for confusion matrix
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        val_accuracy = accuracy_score(all_labels, all_predictions)

        # Calculate validation metrics
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / len(val_loader.dataset)

        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
        print('-' * 50)

    
        # Early stopping check
        if epoch_val_acc > best_val_accuracy:
            best_val_accuracy = epoch_val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break

    # Load best model state
    model.load_state_dict(best_model_state)


    return history, val_accuracy

def NN_evaluate(model, test_loader):
   
    model.eval()
    model.to(torch.float64)
    model = model.to(device)

    all_predictions = []

    # Setup loss function
    criterion = nn.CrossEntropyLoss()

    # 2. Disable gradient computation
    with torch.no_grad():
        # 3. Make predictions on test data
        for batch_X, batch_y in test_loader:
            # Move batch to device
            batch_X = batch_X.to(device)
            outputs = model(batch_X)

            _, predicted = torch.max(outputs.data, 1)

            # Store predictions and labels for confusion matrix
            all_predictions.extend(predicted.cpu().numpy())

    return all_predictions

def embed(tk_doc , w2v_model):

    embeddings = []
    for doc in tk_doc:
        doc_embedding = np.zeros(w2v_model.vector_size)
        count = 0
        for word in doc:
            if word in w2v_model.wv:
                doc_embedding += w2v_model.wv[word]
                count += 1
        if count > 0:
            doc_embedding /= count
        embeddings.append(doc_embedding)

    return embeddings

def w2v_train(train_data, vector_size=100, window=5, min_count=5, workers=4, epochs=10, sg=1):

    model = Word2Vec(train_data, min_count = min_count,
                                   vector_size = vector_size,
                                   window = window,
                                   workers = workers,
                                   epochs = epochs,
                                   sg = sg)
    return model

def load_data(file_path):
    with open(file_path, 'r',encoding="utf8") as inp:
        # Read input data
        data = {
            'id': [],
            'title': [],
            'text': [],
            'label': []
        }
        for i in inp:
            i = json.loads(i)
            data['id'].append(''.join(i['postId']))
            data['title'].append(''.join(i['postText']))
            data['text'].append(''.join(i['targetParagraphs']))
            data['label'].append(''.join(i['tags']))
        # random.seed(156)
        # random.shuffle(train_data)

    return data

def tokenize(text):
    """Tokenize the input text by removing punctuation and splitting into words."""

    tokenized_text = []

    for word in re.findall(r'\w+', text):
        tokenized_text.append(word)

    return tokenized_text

def load_stopword():
    stopword_set =[]
    with open('../stopwords.txt', 'r', encoding='utf-8') as f:
        for line in f:
            stopword_set.append(line.strip())
    res = []
    for item in stopword_set:
        for word in re.findall(r'\w+', item):
            # Key: \w matches any alphanumeric character and underscore
            res.append(word)
    res = set(res)
    # if debug:
    #     print(res)
    return res

def remove_stopwords(tokens, stopwords):
    cleaned_tokens = [word for word in tokens if word.lower() not in stopwords]

    return cleaned_tokens


def train_model(train_data, train_labels, use_bigrams=False, use_unigrams=True):
    """Train a Multinomial Naive Bayes classifier."""
    # Config
    ngram_range = (1, 1) if use_unigrams and not use_bigrams else (2, 2) if not use_unigrams and use_bigrams else (1, 2)
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    MNB = MultinomialNB()

    # Create a pipeline with the vectorizer and classifier
    pipe = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', MNB)
    ])

    # Train the model
    pipe.fit(train_data, train_labels)

    return pipe

def predict(model, data):
    """Predict labels for the given data using the trained model."""
    predictions = model.predict(data)
    return predictions

def evaluate(predictions, test_labels):
    """Evaluate the model's predictions against the true labels."""
    correct = sum(predictions == test_labels)
    accuracy = correct / len(test_labels)
    return accuracy

def save_model(model, filename):
    """Save the trained model to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    if debug:
        print(f"Model saved to {filename}")

if __name__ == '__main__':
    args = parse_args()

    

    run_baseline(args.input, args.output)

