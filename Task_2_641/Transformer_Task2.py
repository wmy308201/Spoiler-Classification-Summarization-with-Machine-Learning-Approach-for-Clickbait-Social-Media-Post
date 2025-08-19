#!/usr/bin/env python3
import argparse
import json
import string
import random
import re
import os
import string
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset


# attention layer
# switch embedding layer to W2V

# nltk.download('stopwords')

debug = True
retrain = True
stopword_removal = False  # True, False
stopword_ref = 'nltk'  # nltk, pre
ref_x = 'text'  # title, text


model_name = 't5-large'  # t5-small, t5-base, t5-large
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

LEARNING_RATE = 5e-4
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 16
NUM_EPOCHS = 2
WEIGHT_DECAY = 1.2
LOGGING_STEPS = 10

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='This is a baseline for task 1 that predicts that each clickbait post warrants a passage spoiler.')
    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The classified output in jsonl format.', required=False)
    return parser.parse_args()


def run_baseline(input_file, output_file):
    print(os.path.isdir("logs"))
    print(os.path.isdir("results"))

    train_data_dir = '../data/train.jsonl'
    val_data_dir = '../data/val.jsonl'
    test_data_dir = '../data/test.jsonl'

    train_dataset = load_preprocess_data(train_data_dir, test_flag=False)
    val_dataset = load_preprocess_data(val_data_dir, test_flag=False)

    # apply tokenization and preprocessing of dataset
    tokenized_train = train_dataset.map(preprocess_dataset, batched=True)
    tokenized_val = val_dataset.map(preprocess_dataset, batched=True)

    if debug:
        print(f"Sample tokenized train input: {tokenized_train[0]['input_text']}")
        print(f"Sample tokenized train target: {tokenized_train[0]['target_text']}")

    # =========================================================================
    # training

    training_args = TrainingArguments(
        output_dir="results",
        eval_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=VAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        logging_dir="logs",
        logging_steps=LOGGING_STEPS,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
    )

    # Fine-tune the model
    trainer.train()

    # Evaluate the model on the validation dataset
    eval_results = trainer.evaluate()

    # Print the evaluation results
    print(f"Evaluation results: {eval_results}")

    # ==========================================================================

    # testing
    with open(test_data_dir, 'r', encoding="utf8") as inp, open(output_dir, 'w', encoding="utf8") as out:
        # Read input data
        data = {
            'id': [],
            'title': [],
            'text': [],
            'label': [],
            'target': []
        }

        # extract data from jsonl file
        for i in inp:
            i = json.loads(i)
            data['id'].append(i['id'])
            data['title'].append(''.join(i['postText']))
            data['text'].append(''.join(i['targetParagraphs']))

        texts = [preprocess_text(text) for text in data['text']]
        titles = [preprocess_text(title) for title in data['title']]
        id = [id for id in data['id']]

        # prepare inputs for the model
        inputs = [f"Question: What is the key spoiler that '{title}' is inferring in passage? Passage: {text}" for
                  title, text in zip(titles, texts)]

        # Tokenize inputs
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')

        # Generate predictions
        predictions = model.generate(input_ids=torch.tensor(model_inputs['input_ids']).to('cuda'))

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        count = 0
        # Write results to output file
        for i, pred in enumerate(decoded_preds):
            if count == 0:
                print(f'input sample: {inputs[0]}')
                print(f'prediction sample: {pred}')
                count = 1
            out.write(json.dumps({'id': id[i], 'spoiler': pred}) + '\n')

def preprocess_dataset(dataset):

    inputs = [f"Question: What is the key spoiler that '{title}' is inferring in passage? Passage: {text}"
              for title, text in zip(dataset['input_title'], dataset['input_text'])]
    targets = [spoiler for spoiler in dataset['target_text']]

    # prepare data for model
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    model_targets = tokenizer(targets, max_length=30, truncation=True, padding='max_length')
    model_inputs["labels"] = model_targets["input_ids"]

    return  model_inputs

def load_preprocess_data(file_path, test_flag):
    with open(file_path, 'r', encoding="utf8") as inp:
        # Read input data
        data = {
            'id': [],
            'title': [],
            'text': [],
            'label': [],
            'target': []
        }

        # extract data from jsonl file
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
                data['target'].append(''.join(i['spoiler']))

        # assign corresponding value to input, output & id
            # clean text
        texts = [preprocess_text(text) for text in data['text']]
        titles = [preprocess_text(title) for title in data['title']]
        if not test_flag:
          spoilers = [preprocess_text(target) for target in data['target']]


        if debug:
            print(f"Sample spoiler: {spoilers[0:5]}")

        if not test_flag:
          data_dict = {
              "input_text" : texts,
              "input_title" : titles,
              "target_text" : spoilers
          }
        else: # for test set
          data_dict = {
              "input_text" : texts,
              "input_title" : titles,
          }

        dataset = Dataset.from_dict(data_dict)

    return dataset

def preprocess_dataset(dataset):

    inputs = [f"Question: What is {title}  Passage: {text}"
              for title, text in zip(dataset['input_title'], dataset['input_text'])]
    targets = [spoiler for spoiler in dataset['target_text']]

    # prepare data for model
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    model_targets = tokenizer(targets, max_length=30, truncation=True, padding='max_length')
    model_inputs["labels"] = model_targets["input_ids"]

    return  model_inputs


def preprocess_text(text):
    # # Convert to lowercase
    # text = text.lower()
    # # # Remove special characters (keep only letters, numbers, spaces & some special punctuations)
    # text = re.sub(r'[^a-zA-Z0-9\s,\!\"\']', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())

    if stopword_removal:
        # remove stop words
        tokenised = tokenize(text)
        sw = load_stopword()
        cleaned = remove_stopwords(tokenised, sw)
        text = " ".join(token for token in cleaned)

    return text


def tokenize(text):
    """Tokenize the input text by removing punctuation and splitting into words."""

    tokenized_text = []

    for word in re.findall(r'\w+', text):
        tokenized_text.append(word)

    return tokenized_text


def load_stopword():
    res = []
    if stopword_ref == 'nltk':
        res = stopwords.words('english')
    else:
        stopword_set = []
        with open('../stopwords.txt', 'r', encoding='utf-8') as f:
            for line in f:
                stopword_set.append(line.strip())

        for item in stopword_set:
            for word in re.findall(r'\w+', item):
                # Key: \w matches any alphanumeric character and underscore
                res.append(word)
        res = set(res)
        # if debug:
        #     print(res)
    return res


def remove_stopwords(tokens, stopwords):
    # reduce word to simplest root form
    porter = PorterStemmer()
    cleaned_tokens = [porter.stem(word) for word in tokens if word.lower() not in stopwords]

    return cleaned_tokens


def save_model(model, filename):
    """Save the trained model to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    if debug:
        print(f"Model saved to {filename}")


if __name__ == '__main__':
    args = parse_args()

    run_baseline(args.input, args.output)

