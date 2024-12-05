import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

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
    return embedding_matrix

# Daten einlesen
df = pd.read_csv("data.tsv", sep="\t")
df = df.head(3000)
X = df.Abstract
y = df.DomainID
number_classes = len(y.unique())

# Train / Test Daten definieren 
np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.7 * len(X))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = X.iloc[train_indices].tolist()
X_test = X.iloc[test_indices].tolist()
y_train = y.iloc[train_indices].values
y_test = y.iloc[test_indices].values

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1

tokenized_X_train = tokenizer.texts_to_sequences(X_train)
tokenized_X_test = tokenizer.texts_to_sequences(X_test)
MAX_LENGTH = max(len(tokenized_text) for tokenized_text in tokenized_X_train)
tokenized_X_train = pad_sequences(tokenized_X_train, maxlen=MAX_LENGTH, padding="post")
tokenized_X_test = pad_sequences(tokenized_X_test, maxlen=MAX_LENGTH, padding="post")

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Modellerstellung
model = Sequential([
    Input(shape=(MAX_LENGTH,)),
    Embedding(vocab_size, 300, input_length=MAX_LENGTH),
    Flatten(),
    Dense(100, activation="relu"),
    Dense(100, activation="relu"),
    Dense(number_classes, activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.summary()

history = model.fit(tokenized_X_train, y_train, epochs=20, verbose=1, validation_split=0.2)

y_pred = model.predict(tokenized_X_test)
y_pred = y_pred.argmax(axis=1)
y_test = y_test.argmax(axis=1)

# Einheiten berechnen
def calculate_metrics(y_true, y_pred):
    true_positives = np.sum((y_true == y_pred) & (y_true != 0))
    false_positives = np.sum((y_true != y_pred) & (y_pred != 0))
    false_negatives = np.sum((y_true != y_pred) & (y_true != 0))
    
    accuracy = np.mean(y_true == y_pred)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Grapherstellung
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Ergebnis an dieser Stelle
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ embedding (Embedding)                │ (None, 541, 300)            │       7,513,800 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ flatten (Flatten)                    │ (None, 162300)              │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense (Dense)                        │ (None, 100)                 │      16,230,100 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 100)                 │          10,100 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_2 (Dense)                      │ (None, 7)                   │             707 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 23,754,707 (90.62 MB)
#  Trainable params: 23,754,707 (90.62 MB)
#  Non-trainable params: 0 (0.00 B)
#
# Accuracy: 0.5233
# Precision: 0.5380
# Recall: 0.5447
# F1-score: 0.5413
