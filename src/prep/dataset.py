"""
Classifying twitter posts by MBTI type
Matthew Flegg ~ matthewflegg@outlook.com
Date ~ 21/02/2022
"""


####################
### READING DATA ###
####################

from math import trunc
from turtle import shape
import numpy as np # Linear algebra
import pandas as pd
import transformers # Data processing, CSV file I/0 (e.g. pd.read_csv)


# Import the MBTI dataset CSV
# Get a list of all of the types
data = pd.read_csv("data/mbti.csv")
types = np.unique(data.type.values)


def get_type_index(string):
    """Converts an MBTI type into a number"""
    return list(types).index(string)


# Create a new column called type_index
data["type_index"] = data["type"].apply(get_type_index)


#####################
### DATA CLEANING ###
#####################

import string # Contains string constants (e.g. string.punctuation)
import re # Regular expressions


def clean(text):
    """Cleans a string of text"""
    regex = re.compile("[%s]" % re.escape("|"))
    text = regex.sub(" ", text)

    # Split string into words
    # Convert to lowercase, remove punctuation, URLs
    words = str(text).split()
    words = [i.lower() + " " for i in words]
    words = [i for i in words if not "http" in i]
    words = " ".join(words)
    words = words.translate(words.maketrans("", "", string.punctuation))

    return words


# Clean all the posts in the dataset
data["cleaned_text"] = data["posts"].apply(clean)


def get_dataframe():
    """Returns the un-split dataframe"""
    return data


##########################
### SPLITTING THE DATA ###
##########################

from sklearn.model_selection import train_test_split # Spliting data


# Split into training, testing, val sets
def get_train_test_val():
    """Gets the train, test and val datasets"""
    train, test = train_test_split(data)
    train, val = train_test_split(train)
    return train, test, val