#####################
### PREPROCESSING ###
#####################

import tensorflow as tf # Machine learning
import prep.dataset as dataset # src/prep/dataset.py, loads data (e.g. the dataframe)

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

import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, TFBertModel
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import adam_v2 as a # Adaptive movement estimation optimizer
from keras.callbacks import EarlyStopping # Stops training when a metric doesn't improve

# Use a tokenizer from the BERT transformer
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

# Tokenize the training and testing sentences using the BERT transformer
train_ids = [tokenizer.encode(str(i), max_length=MAX_LEN, pad_to_max_length=True) for i in train["cleaned_text"].values]
val_ids = [tokenizer.encode(str(i), max_length=MAX_LEN, pad_to_max_length=True) for i in val["cleaned_text"].values]


def create_model():
    """Creates a BERT model"""
    op = a.Adam(learning_rate=0.00001)

    # Use the BERT transformer
    # Then use a dense layer as the output
    input_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")
    BERT_layer = TFBertModel.from_pretrained("bert-large-uncased")
    BERT_outputs = BERT_layer(input_ids)[0]
    dense = Dense(16, activation="softmax")(BERT_outputs[:, 0, :])
    model = Model(inputs=input_ids, outputs=dense)

    # Compile the model
    model.compile(loss="categorical_crossentropy", optimizer=op, metrics=["accuracy"])

    return model


# Create the LSTM model
model = create_model()
BATCH_SIZE = 2

# Training the model
model.fit(
    np.asarray(train_ids).astype("int32"), np.asarray(one_hot_labels).astype("int32"),
    validation_data=(np.asarray(val_ids).astype("int32"), np.asarray(val_labels).astype("int32")),
    verbose=1,
    epochs=20,
    batch_size=BATCH_SIZE,
    callbacks=[EarlyStopping(patience=5)]
)