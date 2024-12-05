import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.layers import Dropout


def get_glove_embeddings(path, vocab_size, tokenizer):
    embedding_vector = {}
    with open(path) as f:
        rf = f.read()
    for line in rf.split("\n"):
        value = line.split(' ')
        word = value[0]
        coef = np.array(value[1:], dtype='float32')
        embedding_vector[word] = coef
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in tokenizer.word_index.items():
        embedding_value = embedding_vector.get(word)
        if embedding_value is not None:
            embedding_matrix[i] = embedding_value
    return [embedding_matrix]

glove_path = "glove.6B.100d.txt" 
embedding_matrix = get_glove_embeddings(glove_path, vocab_size, tokenizer)

df = pd.read_csv("data.tsv", sep="\t")
df = df.head(3000)
X = df.Abstract
y = df.DomainID
number_classes = len(y.unique())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_train = X_train.tolist()
X_test = X_test.tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1  # +1 to account for the padding token
tokenized_X_train = tokenizer.texts_to_sequences(X_train)
tokenized_X_test = tokenizer.texts_to_sequences(X_test)
MAX_LENGTH = max(len(tokenized_text) for tokenized_text in tokenized_X_train)
tokenized_X_train = pad_sequences(tokenized_X_train, maxlen=MAX_LENGTH, padding="post")
tokenized_X_test = pad_sequences(tokenized_X_test, maxlen=MAX_LENGTH, padding="post")

model = Sequential()
model.add(Input(shape=(MAX_LENGTH,)))
model.add(Embedding(input_dim=vocab_size, output_dim=300, weights=[embedding_matrix], input_length=MAX_LENGTH, trainable=False))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.03))
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.23))
model.add(Dense(number_classes, activation="softmax"))
model.add(Dense(150, activation="relu"))
model.add(Dense(175, activation="relu"))
model.compile(loss="crossentropy", optimizer=SGD(learning_rate=0.3))
model.summary()
model.fit(tokenized_X_train, y_train, epochs=30, verbose=1)

y_pred = model.predict(tokenized_X_test)
y_pred = y_pred.argmax(axis=1)
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

print(classification_report(y_test, y_pred))
