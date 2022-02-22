#####################
### PREPROCESSING ###
#####################

import tensorflow as tf # Machine learning
import prep.dataset as dataset # src/dataset.py, loads data (e.g. the dataframe)

from keras.preprocessing.text import Tokenizer # Turns sequences into arrays of numbers
from keras.preprocessing.sequence import pad_sequences # Makes sequences the same length
from keras.utils.all_utils import to_categorical # Num to vector (e.g. 2 to [[0], [0], [1], [0]])


train, test, val = dataset.get_train_test_val()
data = dataset.get_dataframe()

VOCAB_SIZE = 10000
TRUNC_TYPE = "post"
PAD_TYPE = "post"
OOV_TOC = "<OOV>"

# Create a tokenizer and fit it onto the posts
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOC)
tokenizer.fit_on_texts(data.cleaned_text.values)

MAX_LEN = 1500

# Convert training sentences to a sequence of integers and apply padding
train_sequences = tokenizer.texts_to_sequences(train.cleaned_text.values)
train_padded = pad_sequences(train_sequences, maxlen=MAX_LEN, truncating=TRUNC_TYPE, padding=PAD_TYPE)

# Convert validation sentences to a sequence of integers and apply padding
val_sequences = tokenizer.texts_to_sequences(val.cleaned_text.values)
val_padded = pad_sequences(val_sequences, maxlen=MAX_LEN, truncating=TRUNC_TYPE, padding=PAD_TYPE)

# Convert training and val labels to one-hot vectors (e.g. 2 to [[0], [0], [1], [0]])
one_hot_labels = to_categorical(train.type_index.values, num_classes=16)
val_labels = to_categorical(val.type_index.values, num_classes=16)


########################
### CREATING A MODEL ###
########################

from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from keras.optimizers import adam_v2 as a # Adaptive movement estimation optimizer
from keras.callbacks import EarlyStopping # Stops training when a metric doesn't improve


def create_model():
    """Creates a bidirectional LSTM model"""
    op = a.Adam(learning_rate=0.00001)

    # Use a bidirectional LSTM
    # Then use a dense layer as the output
    model = Sequential([
        Embedding(VOCAB_SIZE, 256, input_length=MAX_LEN),
        Dropout(0.3),
        Bidirectional(LSTM(200, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(20)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(16, activation='softmax')
    ])

    # Compile the model, use categorical cross entropy because we're dealing with categorical labels
    model.compile(loss="categorical_crossentropy", optimizer=op, metrics=["accuracy"])

    return model


# Create the LSTM model
model = create_model()
BATCH_SIZE = 2

# Training the model
model.fit(

    # Train for 30 epochs
    train_padded, one_hot_labels,
    epochs=30,
    verbose=1,
    validation_data=(val_padded, val_labels),
    callbacks=[EarlyStopping(patience=3)]
)