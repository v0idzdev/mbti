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

TODO:
    Implement RNN using Keras

Reference:
    https://www.youtube.com/watch?v=kxeyoyrf2cM - Python Engineer
    https://medium.com/@canerkilinc/padding-for-nlp-7dd8598c916a - @kanerkilinc
    https://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate - cybujan
"""


import nltk
import numpy as np
import pandas as pd
import string
import re

from nltk.corpus import stopwords
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Embedding, Dense
from keras.models import Sequential


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


# Remove URLs and punctuation from the posts
df["posts"] = df.posts.map(remove_url)
df["posts"] = df.posts.map(remove_punctuation)
df["posts"] = df.posts.map(remove_stopwords)

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


# Count each unique word in all of the posts
# Print the 5 most common words to check it's working
unique_word_count = count_unique_words(df.posts)
print(f"\n{unique_word_count.most_common(5)}")

# Number of unique words in the data set
num_unique_words = len(unique_word_count)

# TRAINING_SPLIT specifies the percentage of entries in the dataset
# to use for training. Used to find how many entries are used for training
TRAINING_SPLIT = 0.8
training_set_size = int(df.shape[0] * TRAINING_SPLIT)

# Split the data into a training set and a validation set
training_df = df[:training_set_size]
validation_df = df[training_set_size:]

# Convert the text into vectors
training_sentences = training_df.posts.to_numpy()
training_labels = training_df.type.to_numpy()
validation_sentences = validation_df.posts.to_numpy()
validation_labels = validation_df.type.to_numpy()

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
LENGTH_OF_LARGEST_SEQUENCE = len(max(training_sequences))

# Pad the training sequences
training_sequences = pad_sequences(
    training_sequences, maxlen=LENGTH_OF_LARGEST_SEQUENCE, padding="post", truncating="post"
)

# Pad the validation sequences
validation_sequences = pad_sequences(
    validation_sequences, maxlen=LENGTH_OF_LARGEST_SEQUENCE, padding="post", truncating="post"
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


# *Creating the LSTM model for NLP*
# Use a keras.Sequential model
network = Sequential([

    # Using word embeddings for an efficient, dense representation
    Embedding(num_unique_words, 32, input_length=LENGTH_OF_LARGEST_SEQUENCE),

    # LSTM layer will take an integer matrix input of size (batch, input_length) as input
    # The largest integer (e.g., word index) in the input should be no larger than num_unique_words
    LSTM(64, dropout=0.1),
    Dense(1, activation="sigmoid")
])

# Show a summary of the LSTM model
print(f"\n{network.summary()}")