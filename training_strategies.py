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
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Set the working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load and preprocess data
df = pd.read_csv("data.tsv", sep="\t")
df = df.head(3000)  # Use the first 3000 rows
X = df["Abstract"]
y = df["DomainID"]
number_classes = len(y.unique())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_train = X_train.tolist()
X_test = X_test.tolist()

# Tokenize and pad text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1  # +1 to account for the padding token
tokenized_X_train = tokenizer.texts_to_sequences(X_train)
tokenized_X_test = tokenizer.texts_to_sequences(X_test)
MAX_LENGTH = max(len(tokenized_text) for tokenized_text in tokenized_X_train)
tokenized_X_train = pad_sequences(tokenized_X_train, maxlen=MAX_LENGTH, padding="post")
tokenized_X_test = pad_sequences(tokenized_X_test, maxlen=MAX_LENGTH, padding="post")

# Define the model
model = Sequential()
model.add(Input(shape=(MAX_LENGTH,)))
model.add(Embedding(vocab_size, 300, input_length=MAX_LENGTH))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(number_classes, activation="softmax"))

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer=SGD(learning_rate=0.01), metrics=["accuracy"])

# Summarize the model
model.summary()

# Train the model
history = model.fit(tokenized_X_train, y_train, epochs=20, verbose=1, validation_split=0.2)

# Evaluate the model
y_pred = model.predict(tokenized_X_test)
y_pred_classes = y_pred.argmax(axis=1)

# Compute and display evaluation metrics
accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, average="weighted")
recall = recall_score(y_test, y_pred_classes, average="weighted")
f1 = f1_score(y_test, y_pred_classes, average="weighted")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# Plot training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()