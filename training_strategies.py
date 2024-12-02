import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

def get_glove_embeddings(path, vocab_size, tokenizer):
    embedding_vector = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            value = line.split()
            word = value[0]
            coef = np.array(value[1:], dtype='float32')
            embedding_vector[word] = coef
    
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in tokenizer.word_index.items():
        embedding_value = embedding_vector.get(word)
        if embedding_value is not None:
            embedding_matrix[i] = embedding_value
    return embedding_matrix

df = pd.read_csv("data.tsv", sep="\t")
df = df.head(3000) 
X = df['Abstract']
y = df['DomainID']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
number_classes = len(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_train = X_train.tolist()
X_test = X_test.tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1  
tokenized_X_train = tokenizer.texts_to_sequences(X_train)
tokenized_X_test = tokenizer.texts_to_sequences(X_test)
MAX_LENGTH = max(len(seq) for seq in tokenized_X_train)
tokenized_X_train = pad_sequences(tokenized_X_train, maxlen=MAX_LENGTH, padding="post")
tokenized_X_test = pad_sequences(tokenized_X_test, maxlen=MAX_LENGTH, padding="post")

GLOVE_PATH = "glove.6B.300d.txt"  
embedding_matrix = get_glove_embeddings(GLOVE_PATH, vocab_size, tokenizer)

model = Sequential([
    Input(shape=(MAX_LENGTH,)),
    Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=MAX_LENGTH, trainable=False),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(number_classes, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer=SGD(learning_rate=0.01), metrics=["accuracy"])
model.summary()

history = model.fit(tokenized_X_train, y_train, epochs=20, batch_size=32, verbose=1, validation_split=0.1)

y_pred = model.predict(tokenized_X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

report = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_)
print(report)

plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
