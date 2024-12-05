import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# optimizer adam learning rate 0.01, batch size 32, dropout 20%-- overfitted
# df head first 4000, dropout 50%, class weight balanced
# df.sample frac 30 random 42, dropout 25%, dense 120 + 150, dropout 30
# hm



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

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

df = pd.read_csv("data.tsv", sep="\t")
#df = df.sample(frac=0.1, random_state=42)
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
model.add(Dense(120, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(90, activation="relu"))
model.add(Dense(number_classes, activation="softmax"))
model.add(Dropout(0.5))
model.compile(loss="crossentropy", optimizer=Adam(learning_rate=0.03))
model.summary()
history = model.fit(tokenized_X_train, y_train, epochs=20, batch_size=32, verbose=1)

y_pred = model.predict(tokenized_X_test)
y_pred = y_pred.argmax(axis=1)
print(classification_report(y_test, y_pred))

plt.plot(history.history['loss'], label='Training Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()