import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.regularizers import L1, L2, L1L2
from tensorflow.keras.optimizers import Adam


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
model.add(Embedding(vocab_size, 300, input_length=MAX_LENGTH))
model.add(Flatten())
model.add(Dense(100, activation="tanh", kernel_regularizer=L1L2(0.00002, 0.00002)))
model.add(Dropout(0.3))
model.add(Dense(100, activation="tanh", kernel_regularizer=L1L2(0.00002, 0.00002)))
model.add(Dropout(0.3))
model.add(Dense(number_classes, activation="softmax"))

optimizer = Adam(learning_rate=0.002)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer)
model.summary()
model.fit(tokenized_X_train, y_train, epochs=20, verbose=1, batch_size=64, validation_split=0.1)

y_pred = model.predict(tokenized_X_test)
y_pred = y_pred.argmax(axis=1)
print(classification_report(y_test, y_pred))
