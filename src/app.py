import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class APIRecommender(object):
  # read data from csv and process it
  def __init__(self):
    # read data from tables
    self.apis_df = pd.read_csv('../datasets/apis.csv', usecols = [0, 1, 2, 3, 5])
    self.mashups_df = pd.read_csv('../datasets/mashups.csv', usecols = [0, 1, 2, 3, 4])

    # drop rows that contain null values
    self.apis_df.dropna(inplace = True)
    self.mashups_df.dropna(inplace = True)

    # correct index numbers
    self.apis_df.reset_index(drop = True, inplace = True)
    self.mashups_df.reset_index(drop = True, inplace = True)

    # process the description column to create new bag of words column
    self.apis_df['description_words'] = self.apis_df['categories'] + ' ' + self.apis_df['description']
    self.apis_df['description_words'] = self.apis_df['description_words'].apply(self.text_filter_function)
    self.mashups_df['description_words'] = self.mashups_df['tagList'] + ' ' + self.mashups_df['description']
    self.mashups_df['description_words'] = self.mashups_df['description_words'].apply(self.text_filter_function)

  # tokenize, apply filter to, and stem text
  def text_filter_function(self, text):
    # create text filter list of English stopwords and special characters
    text_filter = list(stopwords.words("english"))
    special_characters = [',', '/', '-', '.', ';']
    for char in special_characters:
      text_filter.append(char)

    # initialize text stemmer
    porter = PorterStemmer()

    result = ''
    tokens = word_tokenize(str(text))

    for token in tokens:
      if token not in text_filter:
        result += porter.stem(token.lower())
        result += ' '
    return result

  def add_query(self, query):
    # add a new row of the query to the end of the dfs
    mashups_df = self.mashups_df.append({ 'description_words': self.text_filter_function(query) }, ignore_index = True)
    query_id_in_mashups_df = mashups_df.shape[0] - 1

    apis_df = self.apis_df.append({ 'description_words': self.text_filter_function(query) }, ignore_index = True)
    query_id_in_apis_df = apis_df.shape[0] - 1

    tf_idf = TfidfVectorizer(analyzer = str.split, max_df = 0.85, max_features = 5000)

    mashup_description_matrix = tf_idf.fit_transform(mashups_df['description_words']).toarray()
    self.mashup_cos_sim_matrix = cosine_similarity(mashup_description_matrix, mashup_description_matrix)

    self.api_description_matrix = tf_idf.fit_transform(apis_df['description_words']).toarray()

    return query_id_in_mashups_df, query_id_in_apis_df

  # return APIs with top scores in the top k related mashups of the query
  def recommend_apis_from_mashups(self, mashup_id, k = 15):
    # sort the top related mashups descending using cosine similarity score, and pick the first k elements
    score_series = pd.Series(self.mashup_cos_sim_matrix[mashup_id]).sort_values(ascending = False).iloc[1:k+1]

    apis_in_top_k_mashups = []
    for i in range(k):
      # retrieve a list of API id's from mashups_df
      api_list = self.mashups_df.iloc[score_series.index[i]]['apiList'].split(';')
      api_list = [int(api) for api in api_list]
      # append the list of API id's
      apis_in_top_k_mashups.append(api_list)

    # create a list of unrepeated API id's
    all_apis = []
    for i in range(k):
      for api in apis_in_top_k_mashups[i]:
        if api not in all_apis:
          all_apis.append(api)

    # for each API, if it is used in one of the top k related mashups, add its count by 1
    count = {}
    for i in range(len(all_apis)):
      count[all_apis[i]] = 0
      for j in range(k):
        if all_apis[i] in apis_in_top_k_mashups[j]:
          count[all_apis[i]] += 1

    # each element in api_scores will be API id and its ranking score (its count divided by k times 100)
    api_scores = []
    for i in range(len(all_apis)):
      api_scores.append([all_apis[i], count[all_apis[i]]])

    # sort api_scores by ranking score from highest to lowest
    api_scores = sorted(api_scores, reverse = True, key = lambda x: x[1])[:10]

    # return a dictionary of API's name, descrition and url
    recommendations = []
    for i in range(len(api_scores)):
      api = self.apis_df.loc[self.apis_df['id'] == api_scores[i][0]].values[0]
      recommendations.append({ 'name': api[1], 'description': api[3], 'url': api[4], 'count': api_scores[i][1] })

    return recommendations

  # recommend the top k related APIs to the query
  def recommend_apis_from_apis(self, api_id, k = 15):
    api_scores = []

    # for all API entries except the last one (which is our query)
    for i in range(api_id):
      # calculate the cosine similarity between all mashups and the query
      cos_sim_score_i = cosine_similarity([self.api_description_matrix[api_id]], [self.api_description_matrix[i]])[0][0]
      # append index and related cosine similarity score
      api_scores.append([i, cos_sim_score_i])

    # sort the array descending using cosine similarity score, and pick the first k elements
    api_scores = sorted(api_scores, reverse = True, key = lambda x: x[1])[:k]

    recommendations = []
    for i in range(len(api_scores)):
      # retrieve API's name from apis_df
      api = self.apis_df.loc[self.apis_df['id'] == api_scores[i][0]].values[0]
      # append a tuple of API name and its cosine similarity score
      recommendations.append({ 'name': api[1], 'description': api[3], 'url': api[4], 'count': api_scores[i][1] })

    return recommendations


from flask import Flask, render_template, request
app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def home():
  if request.method == "POST":
    query = request.values.get("query")
    query_id_in_mashups_df, query_id_in_apis_df = api_recommender.add_query(query)

    if request.values.get("search") == "mashup":
      apis = api_recommender.recommend_apis_from_mashups(query_id_in_mashups_df)
      return render_template("index.html", apis = apis, search = "mashup")

    if request.values.get("search") == "api":
      apis = api_recommender.recommend_apis_from_apis(query_id_in_apis_df)
      return render_template("index.html", apis = apis, search = "api")

  return render_template("index.html")


if __name__ == "__main__":
  print("Initialising data...")
  api_recommender = APIRecommender()
  print("Starting app...")
  app.run(debug = True)