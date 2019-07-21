# this file is for reference only, it is already executed
# this file cleans up raw csvs and make new csvs ready for use for the server
import numpy as np
import pandas as pd

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

from nltk import word_tokenize
from nltk import PorterStemmer

# read csv
apis_df = pd.read_csv('../datasets/apis.csv', usecols = [0, 1, 2, 3, 5])
mashups_df = pd.read_csv('../datasets/mashups.csv', usecols = [1, 2, 3, 4, 6])

# only categories column has null values, we simply fill it with a whitespace
categories = apis_df['categories']
categories.fillna(' ', inplace = True)

# rename columns
mashups_df.columns = ['mashup', 'api_list', 'tag_list', 'description', 'url']

# we are not using tag_list in our calculation
# so it's ok to keep those rows with null values in this column
# we fill null values with a whitespace so these rows won't be dropped later
tag_list = mashups_df['tag_list']
tag_list.fillna(' ', inplace = True)

# drop other null values
mashups_df.dropna(inplace = True)

# tokenize and stem text
def text_filter_function(text):
  # create a text filter list of English stopwords and special characters
  text_filter = [stopwords.words('english')]

  special_characters = [',', '/', '-', '.', ';']

  for char in special_characters:
    text_filter.append(char)

  # initialize text stemmer
  porter = PorterStemmer()

  # string to be returned
  result = ''

  # tokenize text
  tokens = word_tokenize(str(text))

  for token in tokens:
    # remove English stopwords and special characters in each token
    if token not in text_filter:
      # stem each token
      result += porter.stem(token.lower())
      result += ' '

  return result

# create bag of words column
apis_df['description_words'] = apis_df['description'].apply(text_filter_function)
mashups_df['description_words'] = mashups_df['description'].apply(text_filter_function)

# export csv
apis_df.to_csv('../datasets/apis_processed.csv', index = False)
mashups_df.to_csv('../datasets/mashups_processed.csv', index = False)