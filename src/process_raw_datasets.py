# This file is for reference only, it is already executed.
# This file cleans up raw csv's and exports new csv's for the server to use.

import numpy as np
import pandas as pd

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

from nltk import word_tokenize
from nltk import PorterStemmer

# Read csv
apis_df = pd.read_csv('../datasets/apis.csv', usecols=[0, 1, 2, 3, 5])
mashups_df = pd.read_csv('../datasets/mashups.csv', usecols=[1, 2, 3, 4, 6])

# Only categories column has null values, we simply fill it with a whitespace.
categories = apis_df['categories']
categories.fillna(' ', inplace=True)

# Rename columns
mashups_df.columns = ['mashup', 'api_list', 'tag_list', 'description', 'url']

# We are not using tag_list in our calculation,
# so it's ok to keep those rows with null values in this column.
# We fill null values with a whitespace so these rows won't be dropped later.
tag_list = mashups_df['tag_list']
tag_list.fillna(' ', inplace=True)

# Drop other null values
mashups_df.dropna(inplace=True)


def process_text(text):
    # Create a text filter list of English stopwords and special characters
    text_filter = [stopwords.words('english')]
    special_characters = [',', '/', '-', '.', ';']
    for char in special_characters:
        text_filter.append(char)

    # Initialize text stemmer
    porter = PorterStemmer()

    # String to be returned
    result = ''

    # Tokenize text
    tokens = word_tokenize(str(text))
    for token in tokens:
        # Remove English stopwords and special characters in each token
        if token not in text_filter:
            # Stem each token
            result += porter.stem(token.lower())
            result += ' '

    return result

# Create bag of words column
apis_df['description_words'] = apis_df['description'].apply(process_text)
mashups_df['description_words'] = mashups_df['description'].apply(process_text)

# Export csv
apis_df.to_csv('../datasets/apis_processed.csv', index=False)
mashups_df.to_csv('../datasets/mashups_processed.csv', index=False)
