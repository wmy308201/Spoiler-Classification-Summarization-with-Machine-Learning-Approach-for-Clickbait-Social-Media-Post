#!/usr/bin/env python3
import argparse
import json
import string
import random
import re
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

# attention layer
# switch embedding layer to W2V

nltk.download('stopwords')

debug = True
retrain = True
stopword_removal = False  # True, False
stopword_ref = 'nltk'  # nltk, pre
ref_x = 'title'  # title, text
w2v = False # use word2vec embedding

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Train Hyperparameters
DROPOUT_RATE = 0.3  # on overfitting
LEARNING_RATE = 0.05  # on overfitting
NUM_EPOCHS = 25
L2_REG = 0.005  # on overfitting
EARLY_STOPPING_PATIENCE = 6
KERNAL_SIZE = 5  # for CNNv3

# Model Hyperparameters
MAX_LENGTH = 100  # for padding/truncating sequences
BATCH_SIZE = 32
W2V_EMBED_DIM = 64
NUM_FILTERS = 100
FILTER_SIZES = [2, 3, 4, 5]  # different n-gram sizes
NUM_CLASSES = 3

# Word2Vec parameters
EMBED_WINDOW = 3
EMBED_MINCOUNT = 1
EMBED_WORKER = 16

tuning_param = [MAX_LENGTH, BATCH_SIZE, W2V_EMBED_DIM, NUM_FILTERS, FILTER_SIZES, NUM_CLASSES, DROPOUT_RATE, LEARNING_RATE,
                NUM_EPOCHS, L2_REG, EARLY_STOPPING_PATIENCE]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='This is a baseline for task 1 that predicts that each clickbait post warrants a passage spoiler.')
    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The classified output in jsonl format.', required=False)
    return parser.parse_args()


def run_baseline(input_file, output_file):
    train_data_dir = '../data/train.jsonl'
    val_data_dir = '../data/val.jsonl'

    train, train_labels, train_id = load_preprocess_data(train_data_dir, test_flag=False)
    val, val_labels, val_id = load_preprocess_data(val_data_dir, test_flag=False)

    if debug:
        print("Training data loaded.")
        print(f"Number of training samples: {len(train)}")
        print(f"Sample text: {train[0:5]}")
        print(f"Sample label: {train_labels[0:5]}")

    # =====================================================================================================
    # Data up to this point is loaded and preprocessed,
    # i.e. removed stopwords, label encoded, full length text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train)

    word2idx = tokenizer.word_index
    word2idx['<PAD>'] = 0  # manually add if you want 0 reserved

    vocab_size = len(tokenizer.word_index) + 1

    if w2v: # using word2vec embedding
        # Logic
            # tokenizer created from training data
            # w2v model created from training data

            # convert all data to sequences
        # create Word2Vec model
        w2v_md = create_embed_model(train)

        train = tokenizer.texts_to_sequences(train)
        val = tokenizer.texts_to_sequences(val)

        # padding sequences
        max_length = MAX_LENGTH
        train = pad_sequences(train, maxlen=max_length, padding='post')
        val = pad_sequences(val, maxlen=max_length, padding='post')



        model = CNN_w2v(
            vocab_size=vocab_size,
            embed_dim=W2V_EMBED_DIM,
            word2idx=word2idx,
            w2v_model=w2v_md,
            max_length=MAX_LENGTH,
            num_classes=NUM_CLASSES
        ).to(device)

        if debug:
            print("Word2Vec embedding created and model initialized.")

        train_dataset = TensorDataset(torch.tensor(train, dtype=torch.long), torch.tensor( train_labels, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(val, dtype=torch.long), torch.tensor( val_labels, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    else: # using auto embedding

        # build vocab
        vocab = build_vocabulary(train)
        vocab_size = len(vocab)
        print(f"Vocabulary size: {vocab_size}")

        # =====================================================================================================

        train_dataset = TextDataset(train, train_labels, vocab, MAX_LENGTH)
        val_dataset = TextDataset(val, val_labels, vocab, MAX_LENGTH)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # use auto embedding
        model = CNN(
            vocab_size=len(vocab),
            embed_dim=W2V_EMBED_DIM,
            num_filters=NUM_FILTERS,
            filter_sizes=FILTER_SIZES,
            num_classes=NUM_CLASSES,
            dropout_rate=DROPOUT_RATE
        ).to(device)

    if debug:
        print("loaded")

    if retrain:
        # ==================================================================================================
        # Start of model training

        print("Train Start.................")
        print("=" * 40)

        # Train the model
        train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, L2_REG
        )

        print("completed")
        print("=" * 40)

        # Display final results
        final_train_acc = train_accs[-1]
        final_val_acc = val_accs[-1]
        best_val_acc = max(val_accs)
        best_epoch = val_accs.index(best_val_acc) + 1

        print(f"\n📊 Final Training Results:")
        print(f"Final Training Accuracy: {final_train_acc:.2f}%")
        print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
        print(f"Param setting: {tuning_param}")

        if w2v:
            torch.save(model.state_dict(), "CNN_w2v.pth")
        else:
            torch.save(model.state_dict(), "CNN.pth")

    # ==================================================================================================
    # ==================================================================================================
    if debug:
        print("moving on to test prediction")
    with open(input_file, 'r', encoding="utf8") as inp, open(output_file, 'w') as out:

        if w2v:  # using word2vec embedding
            test, _, test_id = load_preprocess_data(input_file, test_flag=True)
            test = tokenizer.texts_to_sequences(test)
            test = pad_sequences(test, maxlen=max_length, padding='post')

            test_dataset = TensorDataset(torch.tensor(test, dtype=torch.long), torch.tensor(train_labels, dtype=torch.long))
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            w2v_md = pickle.load(open("w2v_model.pkl", 'rb'))

            model = CNN_w2v(
                vocab_size=vocab_size,
                embed_dim=W2V_EMBED_DIM,
                word2idx=word2idx,
                w2v_model=w2v_md,
                max_length=MAX_LENGTH,
                num_classes=NUM_CLASSES,
                dropout_rate=DROPOUT_RATE
            ).to(device)
            model.load_state_dict(torch.load('CNN.pth', weights_only=True))

        else:  # using auto embedding
            # Read input data
            test, _, test_id = load_preprocess_data(input_file, test_flag=True)

            test_dataset = TestTextDataset(test, vocab, MAX_LENGTH)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            model = CNN(
                vocab_size=len(vocab),
                embed_dim=W2V_EMBED_DIM,
                num_filters=NUM_FILTERS,
                filter_sizes=FILTER_SIZES,
                num_classes=NUM_CLASSES,
                dropout_rate=DROPOUT_RATE
            ).to(device)

            model.load_state_dict(torch.load('CNN.pth', weights_only=True))
            print('loaded CNN.pth')

        test_pred, test_prob = evaluate_model(model, test_loader)
        if debug:
            print("test predicted")
        # 1 : passage, 2: phrase, 0: multi
        for i in range(0, len(test_pred)):
            if test_pred[i] == 1:
                res = 'passage'
            elif test_pred[i] == 2:
                res = 'phrase'
            else:
                res = 'multi'
            prediction = {'id': test_id[i], 'spoilerType': res}
            out.write(json.dumps(prediction) + '\n')
        if debug:
            print("completed")

def build_vocabulary(texts, min_freq=2):
    """Build vocabulary from texts"""
    word_counts = Counter()
    for text in texts:
        words = text.split()
        word_counts.update(words)

    # Create vocabulary with special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2

    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab



class CNN(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes,
                 num_classes, dropout_rate=0.5, pretrained_embeddings=None):
        super(CNN, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        # Convolutional layers with different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len)

        # Embedding: (batch_size, seq_len, embed_dim)
        embedded = self.embedding(x)

        # Transpose for conv1d: (batch_size, embed_dim, seq_len)
        embedded = embedded.transpose(1, 2)



        # Apply convolutions and max-pooling
        conv_outputs = []
        for conv in self.convs:
            # Convolution: (batch_size, num_filters, conv_seq_len)
            conv_out = F.relu(conv(embedded))
            # Max-pooling over time: (batch_size, num_filters)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)

        # Concatenate all pooled outputs: (batch_size, len(filter_sizes) * num_filters)
        concatenated = torch.cat(conv_outputs, dim=1)

        # Apply dropout
        dropped = self.dropout(concatenated)

        # Final classification layer
        output = self.fc(dropped)

        return output

class CNNv3(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes,
                 num_classes, dropout_rate=0.5, pretrained_embeddings=None):
        super(CNNv3, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        self.dropout1 = nn.Dropout(dropout_rate)

        # Block 1: Conv1d(50) → Conv1d(50) → MaxPool1d → Dropout
        self.conv1_1 = nn.Conv1d(embed_dim, 50, kernel_size=KERNAL_SIZE, stride=1, padding=1)
        self.conv1_2 = nn.Conv1d(50, 50, kernel_size=KERNAL_SIZE, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=KERNAL_SIZE)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Block 2: Conv1d(100) → Conv1d(100) → MaxPool1d → Dropout
        self.conv2_1 = nn.Conv1d(50, 100, kernel_size=KERNAL_SIZE, stride=1, padding=1)
        self.conv2_2 = nn.Conv1d(100, 100, kernel_size=KERNAL_SIZE, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Block 3: Conv1d(200) → Conv1d(200) → GlobalMaxPool1d → Dropout
        self.conv3_1 = nn.Conv1d(100, 200, kernel_size=KERNAL_SIZE, stride=1, padding=1)
        self.conv3_2 = nn.Conv1d(200, 200, kernel_size=KERNAL_SIZE, stride=1, padding=1)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout4 = nn.Dropout(dropout_rate)

        # Fully connected layers
        self.fc1 = nn.Linear(200, 200)
        self.relu = nn.ReLU()
        self.dropout5 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(200, num_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        # x = self.dropout1(x)
        x = x.transpose(1, 2)  # (batch_size, embed_dim, seq_len)

        # Block 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = self.dropout2(x)

        # Block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = self.dropout3(x)

        # Block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.global_pool(x)  # (batch_size, 200, 1)
        x = x.squeeze(2)  # (batch_size, 200)
        x = self.dropout4(x)

        # FC layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout5(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x


class CNN_w2v(nn.Module):
    def __init__(self, vocab_size, embed_dim,word2idx,w2v_model, max_length, num_classes=3, dropout_rate=0.5):

        super(CNN_w2v, self).__init__()

        # Build embedding matrix from w2v model
        embedding_matrix = np.zeros((vocab_size, embed_dim))
        for word, idx in word2idx.items():
            if word in w2v_model.wv:
                embedding_matrix[idx] = w2v_model.wv[word]
            else:
                embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embed_dim,))

        # Embedding layer using pretrained weights (Word2Vec)
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=True,
            padding_idx=0
        )

        self.dropout1 = nn.Dropout(dropout_rate)

        # Block 1: Conv1d(50) → Conv1d(50) → MaxPool1d → Dropout
        self.conv1_1 = nn.Conv1d(embed_dim, 50, kernel_size=KERNAL_SIZE, stride=1, padding=1)
        self.conv1_2 = nn.Conv1d(50, 50, kernel_size=KERNAL_SIZE, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=KERNAL_SIZE)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Block 2: Conv1d(100) → Conv1d(100) → MaxPool1d → Dropout
        self.conv2_1 = nn.Conv1d(50, 100, kernel_size=KERNAL_SIZE, stride=1, padding=1)
        self.conv2_2 = nn.Conv1d(100, 100, kernel_size=KERNAL_SIZE, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Block 3: Conv1d(200) → Conv1d(200) → GlobalMaxPool1d → Dropout
        self.conv3_1 = nn.Conv1d(100, 200, kernel_size=KERNAL_SIZE, stride=1, padding=1)
        self.conv3_2 = nn.Conv1d(200, 200, kernel_size=KERNAL_SIZE, stride=1, padding=1)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout4 = nn.Dropout(dropout_rate)

        # Fully connected layers
        self.fc1 = nn.Linear(200, 200)
        self.relu = nn.ReLU()
        self.dropout5 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(200, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = self.dropout1(x)
        x = x.transpose(1, 2)  # (batch_size, embed_dim, seq_len)

        # Block 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = self.dropout2(x)

        # Block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = self.dropout3(x)

        # Block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.global_pool(x)  # (batch_size, 200, 1)
        x = x.squeeze(2)  # (batch_size, 200)
        x = self.dropout4(x)

        # FC layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout5(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, l2_reg=0.01):
    # Initialize tracking variables
    best_val_accuracy = 0.0
    best_model_state = None
    early_stopping_patience = EARLY_STOPPING_PATIENCE
    patience_counter = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        for batch_texts, batch_labels in train_loader:
            batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_texts)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()

        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_texts, batch_labels in val_loader:
                batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)
                outputs = model(batch_texts)
                loss = criterion(outputs, batch_labels)

                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch_labels.size(0)
                correct_val += (predicted == batch_labels).sum().item()

        # Calculate averages
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_acc = 100 * correct_train / total_train
        val_acc = 100 * correct_val / total_val

        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Calculate validation metrics
        epoch_val_loss = avg_val_loss
        epoch_val_acc = val_acc

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
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

    return train_losses, val_losses, train_accuracies, val_accuracies


def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    model.eval()
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for batch_texts in test_loader:
            batch_texts = batch_texts.to(device)
            outputs = model(batch_texts)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())

            all_probabilities.extend(probabilities.cpu().numpy())

    return all_predictions, all_probabilities


def load_preprocess_data(file_path, test_flag):
    with open(file_path, 'r', encoding="utf8") as inp:
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

        X = [preprocess_text(text) for text in X_raw]

        if debug:
            print(f"Sample label: {y_raw[0:5]}")

        le = LabelEncoder()
        y = le.fit_transform(y_raw)

    return X, y, data_id

def preprocess_text(text):
    """Simple text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters (keep only letters, numbers, spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())

    if stopword_removal:
        # remove stop words
        tokenised = tokenize(text)
        sw = load_stopword()
        cleaned = remove_stopwords(tokenised, sw)
        text = " ".join(token for token in cleaned)

    return text

def create_embed_model(train):
    # Train the Word2Vec model
    if retrain:
        sentences = [sentence.split() for sentence in train]
        w2v_model = Word2Vec(sentences, vector_size=W2V_EMBED_DIM, window=EMBED_WINDOW, min_count=EMBED_MINCOUNT, workers=EMBED_WORKER)
        # Save the model
        pickle.dump(w2v_model, open("w2v_model.pkl",'wb'))
    else:
        # Load the pre-trained model
        with open("w2v_model.pkl", "rb") as f:
            w2v_model = pickle.load(f)

    return w2v_model

def test_embed(tk_doc , w2v_model):

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

def tokenize(text):
    """Tokenize the input text by removing punctuation and splitting into words."""

    tokenized_text = []

    for word in re.findall(r'\w+', text):
        tokenized_text.append(word)

    return tokenized_text

class TextDataset(Dataset):
    """Custom dataset for text classification"""

    def __init__(self, texts, labels, vocab, max_length=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Convert text to indices
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in text.split()]

        # Pad or truncate to max_length
        if len(indices) < self.max_length:
            indices.extend([self.vocab['<PAD>']] * (self.max_length - len(indices)))
        else:
            indices = indices[:self.max_length]

        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class TestTextDataset(Dataset):
    """Custom dataset for text classification"""

    def __init__(self, texts, vocab, max_length=100):
        self.texts = texts
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Convert text to indices
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in text.split()]

        # Pad or truncate to max_length
        if len(indices) < self.max_length:
            indices.extend([self.vocab['<PAD>']] * (self.max_length - len(indices)))
        else:
            indices = indices[:self.max_length]

        return torch.tensor(indices, dtype=torch.long)

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

