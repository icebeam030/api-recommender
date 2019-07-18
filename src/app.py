# the algorithm part
import numpy as np
import pandas as pd

import nltk
# download required packages
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

from nltk import word_tokenize
from nltk import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class APIRecommender:

  def __init__(self):
    # create dataframes from csv, used to retrieve relevant info
    self.apis_df = pd.read_csv('../datasets/apis_processed.csv')
    self.mashups_df = pd.read_csv('../datasets/mashups_processed.csv')

    # updated when query is updated, used by recommendation function
    self.mashup_cos_sim_matrix = []


  # tokenize and stem text
  def text_filter_function(self, text):
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


  def add_query(self, query, max_df = 0.85, max_features = 5000):
    # add a new row for the query to the end of the dataframe
    # original df is not changed
    mashups_df = self.mashups_df.append({
      'description_words': self.text_filter_function(query)
    }, ignore_index = True)

    # initialize tf-idf vectorizer
    tf_idf = TfidfVectorizer(analyzer = str.split, max_df = max_df, max_features = max_features)

    # apply tf-idf transformation
    mashup_description_matrix = tf_idf.fit_transform(mashups_df['description_words']).toarray()

    # calculate cosine similarity matrix
    self.mashup_cos_sim_matrix = cosine_similarity(
      mashup_description_matrix,
      mashup_description_matrix
    )

    # return query's index in mashups_df to be passed into recommendation function
    query_index = mashups_df.shape[0] - 1

    return query_index


  # return APIs with top scores in the top k similar mashups of the query
  def recommend_apis_from_mashups(self, query_index, k = 15):
    # sort the top related mashups descending using cosine similarity score
    # then pick the top k elements, skip the first mashup because it's the query itself
    score_series = pd.Series(
      self.mashup_cos_sim_matrix[query_index]
    ).sort_values(ascending = False).iloc[1 : k + 1]

    # get all API id's used in top k related mashups
    apis_in_top_k_mashups = []

    for i in range(k):
      # retrieve a list of API id's from mashups_df
      api_list = self.mashups_df.iloc[score_series.index[i]]['api_list'].split(';')
      api_list = [int(api) for api in api_list]

      apis_in_top_k_mashups.append(api_list)

    # filter out repeated API id's from apis_in_top_k_mashups
    all_apis = []

    for apis in apis_in_top_k_mashups:
      for api in apis:
        if api not in all_apis:
          all_apis.append(api)

    # for each API, if it is used in one of the top k related mashups, increment its count by 1
    count = {}

    for api in all_apis:
      count[api] = 0

      for apis in apis_in_top_k_mashups:
        if api in apis:
          count[api] += 1

    # each element in api_scores will be API's id and its count
    api_scores = []

    for api in all_apis:
      api_scores.append({
        'id': api,
        'count': count[api]
      })

    # sort api_scores by count from highest to lowest
    # only recommend up to 10 APIs
    if (len(api_scores) < 10):
      api_scores = sorted(api_scores, reverse = True, key = lambda x: x['count'])
    else:
      api_scores = sorted(api_scores, reverse = True, key = lambda x: x['count'])[ : 10]

    # return a dictionary to be displayed on web page
    recommendations = []

    # for each element in api_scores
    for api_score in api_scores:
      # retrieve corresponding row from apis_df
      api_row = self.apis_df.iloc[api_score['id'] - 1].values

      recommendations.append({
        'name': api_row[1],
        'categories': api_row[2],
        'description': api_row[3],
        'url': api_row[4],
        'count': api_score['count']
      })

    return recommendations


# the server part
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def home():
  if request.method == "POST":
    max_df = float(request.values.get("max_df"))
    max_features = int(request.values.get("max_features"))

    recommender = APIRecommender()

    query = request.values.get("query")
    query_index = recommender.add_query(query, max_df, max_features)

    if request.values.get("search") == "mashup":
      apis = recommender.recommend_apis_from_mashups(query_index)
      return render_template("index.html", apis = apis, search = "mashup")

  return render_template("index.html")


if __name__ == "__main__":
  print("Starting server...")
  app.run(debug = True)