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
      result += " "
  return result

# process the description column to create new bag of words column
apis_df["description_words"] = apis_df["description"].apply(text_filter_function)
# mashups_df["description_words"] = mashups_df["description"].apply(text_filter_function)

# identify importance of words using the tf-idf scheme
tf_idf = TfidfVectorizer(analyzer = str.split)
api_description_matrix = tf_idf.fit_transform(apis_df["description_words"])
# mashup_description_matrix = tf_idf.fit_transform(mashups_df["description_words"])

# recommend the top k related APIs to a given API
def recommend_apis(api_id, k = 10):
  recommendations = []
  top_k_recommendations = []

  # for all API entries
  for i in range(api_description_matrix.shape[0]):
    # if it's not the same as selected API
    if i != api_id:
      # calculate the cosine similarity between all other APIs and selected API
      cos_sim_score_i = cosine_similarity(api_description_matrix[api_id], api_description_matrix[i])[0][0]
      # append index and related cosine similarity score
      top_k_recommendations.append((i, cos_sim_score_i))

  # sort the array descending using cosine similarity score, and pick the first k elements
  top_k_recommendations = sorted(top_k_recommendations, reverse = True, key = lambda x: x[1])[:k]

  for i in top_k_recommendations:
    # retrieve API's name from apis_df
    api_name = apis_df.iloc[i[0]]["api"]
    cos_sim_score = i[1]
    # append a tuple of API name and its cosine similarity score
    recommendations.append((api_name, cos_sim_score))

  api_name = apis_df.iloc[api_id]["api"]
  print("The top " + str(k) + " recommended APIs and their cosine similarity score for " + api_name + " is:\n")

  return recommendations

# test recommendation
recommend_apis(3)