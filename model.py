#!/usr/bin/env python

__description__ = "Predicts an MBTI type based on text"
__author__ = "Matthew Flegg"
__version__ = "0.0.1"
__date__ = "20/02/2022"


"""
MBTI Type Predictor - v0.0.1
Source code put in the public domain by Matthew Flegg, no Copyright
https://github.com/MatthewFlegg
matthewflegg@outlook.com

History:
    20/02/2022: Started project
    20/02/2022: Finished data preparation
    21/02/2022: Built model and started experimenting ~ Network is badly overfitting

TODO:
    Fine-tune the embedding dimension
    Experiment with optimizers and find the right one
    Experiment with the network structure to prevent overfitting

Reference:
    https://www.youtube.com/watch?v=kxeyoyrf2cM - Python Engineer
    https://medium.com/@canerkilinc/padding-for-nlp-7dd8598c916a - @kanerkilinc
    https://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate - cybujan
    https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/ - Jason Brownlee
    https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17 - Susan Li
    https://moviecultists.com/which-optimizer-for-lstm - Electa Brakus
"""


import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import re

from nltk.corpus import stopwords
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Embedding, Dense, SpatialDropout1D, Bidirectional
from keras.models import Sequential
from keras.utils.all_utils import to_categorical


# *Opening the CSV file with the MBTI data*
# Put it into a pandas dataframe called df
with open("data/mbti.csv", encoding="utf8") as file:
    df = pd.read_csv(file)

types = [
    "INTP",
    "INTJ",
    "ENTP",
    "ENTJ",
    "ISTP",
    "ISTJ",
    "ESTP",
    "ESTJ",
    "INFP",
    "INFJ",
    "ENFP",
    "ENFJ",
    "ISFP",
    "ISFJ",
    "ESFP",
    "ESFJ",
]

# Print shape and number of rows and columns in the dataset
print(f"Shape: {df.shape}\n" + f"Rows: {df.shape[0]}, Cols: {df.shape[1]}\n")

# Print how many of each type there are
print(f"Type  Posts")
for i, type_ in enumerate(types):
    print(f"{type_}  {(df.type == type_).sum()}")


# *Data cleaning*
# Stopwords are commonly used words (e.g., "the", "a", "an", "in") that a
# search engine has been programmed to ignore
stops = set(stopwords.words("english"))

# Define data cleaning functions
def remove_url(text):
    """Removes a URL from a string of text"""
    url = re.compile(r"https?://\S+|www.\.\S+")
    return url.sub(r"", text)


def remove_punctuation(text):
    """Removes punctuation from a string of text"""
    return re.sub("[" + string.punctuation + "]", "", text)


def remove_stopwords(text):
    """Removes words like "a", "an", "the", etc. from a string"""
    text = [word.lower() for word in text.split() if word.lower() not in stops]
    return " ".join(text)  # Puts the list of filtered words together


def remove_numbers(text):
    """Removes all numbers from a string of text"""
    return re.sub(r"[0-9]", "", text)


# Remove URLs, punctuation, numbers and stopwords from the posts
df["posts"] = df.posts.map(remove_url)
df["posts"] = df.posts.map(remove_punctuation)
df["posts"] = df.posts.map(remove_stopwords)
df["posts"] = df.posts.map(remove_numbers)

# Shuffle the dataset's rows
df = df.sample(frac=1)

# Check that the data cleaning has worked properly
print(f"\n{df['posts'].head()}")


# *Converting the data into numbers*
# Defining some useful functions
def count_unique_words(df_column):
    """Counts unique words in a dataframe column"""
    count = Counter()

    for line in df_column.values:
        for word in line.split():
            count[word] += 1

    return count


def categorical_label_to_type(label):
    """Converts a categorical label to an MBTI type"""
    index_in_types_list = np.argmax(label) - 1
    return types[index_in_types_list]


# Count each unique word in all of the posts
# Print the 5 most common words to check it's working
unique_word_count = count_unique_words(df.posts)
print(f"\n{unique_word_count.most_common(5)}")

# Number of unique words in the data set
num_unique_words = len(unique_word_count)

# TRAINING_SPLIT specifies the percentage of entries in the dataset
# to use for training. Used to find how many entries are used for training
TRAINING_SPLIT = 0.9
training_set_size = int(df.shape[0] * TRAINING_SPLIT)

# Split the data into a training set and a validation set
training_df = df[:training_set_size]
validation_df = df[training_set_size:]

# Convert the text into vectors
training_sentences = training_df["posts"].to_numpy()
training_labels = training_df["type"].to_numpy()
validation_sentences = validation_df["posts"].to_numpy()
validation_labels = validation_df["type"].to_numpy()

# Convert labels into integers
training_labels = [types.index(type_) for type_ in training_labels]
validation_labels = [types.index(type_) for type_ in validation_labels]

# Convert labels to categorical
training_labels = to_categorical(training_labels)
validation_labels = to_categorical(validation_labels)

# Print sizes of the split data to check that everything's
# working correctly
print(
    f"\nNum training sentences: {training_sentences.shape}"
    + f"\nNum validation sentences: {validation_sentences.shape}"
)


# *Tokenizing the data*
# Gives each word a unique index
tokenizer = Tokenizer(num_words=num_unique_words)
tokenizer.fit_on_texts(training_sentences)  # Only fit to training data

# Each word has a unique index
word_index = tokenizer.word_index

# Convert the tokenized sentences into sequences
# The sequences have the same size as the sentences, but instead of containing the
# words they contain the unique indexes associated with the words
training_sequences = tokenizer.texts_to_sequences(training_sentences)
validation_sequences = tokenizer.texts_to_sequences(validation_sentences)

# Compare examples of the sentences to sequences
print(f"\n{training_sentences[10]}\n{training_sequences[10]}")

# Each sequence has a different length, so apply padding
# Example: [6, 7, 6, 3, 2], [3, 3] becomes [6, 7, 6, 3, 2], [3, 3, 0, 0, 0]
MAX_SEQUENCE_LENGTH = 256

# Pad the training sequences
training_sequences = pad_sequences(
    training_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post"
)

# Pad the validation sequences
validation_sequences = pad_sequences(
    validation_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post"
)

# Check the length of the padded sequences
print(
    f"\nPadded: \n{training_sequences.shape} - Training"
    + f"\n{validation_sequences.shape} - Validation"
)


# *Testing the tokenization*
# Swaps the keys around. Example: {"hello": 10} becomes {10: "hello"}
reversed_word_index = dict([(index, word) for (word, index) in word_index.items()])


def decode(sequence):
    """Converts a sequence of indexes back into a sentence"""
    return " ".join([reversed_word_index.get(index, "?") for index in sequence])


# Decode some training sentences
decoded_training_sequences = decode(training_sequences[10])

# Check that the sentences have encoded correctly by decoding them
print(
    f"\nSequences: {training_sequences[10]}\n"
    + f"Decoded Sequences: {decoded_training_sequences}"
)

# Check the shape of the training and validation sequences
print(f"\nTraining: {training_sequences.shape}, Val: {validation_sequences.shape}")


# *Creating the LSTM model for NLP*
# Max sequence length is MAX_SEQUENCE_LENGTH, the max number of words is num_unique_words
# ! Embedding dimension set arbitrarily
EMBEDDING_DIMENSION = 32

# Use a keras.Sequential model
network = Sequential(
    [
        # Using word embeddings for an efficient, dense representation
        Embedding(
            num_unique_words,
            EMBEDDING_DIMENSION,
            input_length=MAX_SEQUENCE_LENGTH,
        ),
        # ? Add a spatial dropout
        # SpatialDropout1D(0.2),
        # LSTM layers will take an integer matrix input of size (batch, input_length) as input
        # The largest integer (e.g., word index) in the input should be no larger than num_unique_words
        # ?? Using Tanh activation and Sigmoid recurrent activation so the LSTM layer can run on the GPU
        LSTM(
            128,
            # ? Experimenting with reducing regularization techniques
            dropout=0.1,
            recurrent_dropout=0,
            activation="tanh",
            recurrent_activation="sigmoid",
            unroll=False,
        ),
        # Adding two LSTM layers for experimentation purposes
        Dense(16, activation="sigmoid"),
    ]
)

# Show a summary of the LSTM model
print(f"\n{network.summary()}")

# Compile the model
network.compile(
    # ? Using RMSprop optimizer but this is subject to change
    optimizer="rmsprop",
    # Categorical cross entropy because we're dealing with vector-based
    # (categorical) inputs
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS = 16
BATCH_SIZE = 32


# *Training the model*
# Use a batch size of 20, train for
# 15 iterations
training_history = network.fit(
    # Use our sequences and labels
    x=training_sequences,
    y=training_labels,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    # Use our validation sequences as the validation
    # data. TODO: Find out how to get this to work
    validation_data=(validation_sequences, validation_labels)
)


# *Summarizing the model's training history*
# Summarize history for accuracy
plt.plot(training_history.history["accuracy"])
plt.plot(training_history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Training", "Testing"], loc="upper left")
plt.show()

# Summarize history for loss
plt.plot(training_history.history["loss"])
plt.plot(training_history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Training", "Testing"], loc="upper left")
plt.show()


print(f"\n{training_sequences.shape} {training_labels.shape}\n")
print(training_labels[5])
