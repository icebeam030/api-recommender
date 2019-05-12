import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# read data from tables
apis_df = pd.read_csv("../datasets/apis.csv", usecols = [0, 1, 2, 3])
mashups_df = pd.read_csv("../datasets/mashups.csv", usecols = [0, 1, 2, 3, 4])

# create text filter list of English stopwords and special characters
text_filter = list(stopwords.words("english"))
special_characters = [",", "/", "-", "."]
for char in special_characters:
  text_filter.append(char)

# initialize text stemmer
porter = PorterStemmer()

# tokenize, apply filter to, and stem text
def text_filter_function(text):
  result = ""
  tokens = word_tokenize(str(text))
  for token in tokens:
    if token not in text_filter:
      result += porter.stem(token.lower())
  return result

# process the description column to create new bag of words column
apis_df["description_words"] = apis_df["description"].apply(text_filter_function)
mashups_df["description_words"] = mashups_df["description"].apply(text_filter_function)

# identify importance of words using the tf-idf scheme
tf_idf = TfidfVectorizer()
api_description_matrix = tf_idf.fit_transform(apis_df["description_words"])
mashup_description_matrix = tf_idf.fit_transform(apis_df["description_words"])

# cosine similarity between api and mashup matrix
cosine_similarity_matrix = cosine_similarity(mashup_description_matrix, api_description_matrix)

# recommend the top k APIs to a mashup
def recommend_apis(mashup_id, k):
  recommendations = []
  top_k_recommendations = pd.Series(cosine_similarity_matrix[mashup_id]).sort_values(ascending = False).iloc[0:k]
  for index in top_k_recommendations.index:
    recommendations.append(apis_df.iloc[index]["api"])
  return recommendations