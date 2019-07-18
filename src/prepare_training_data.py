# this file prepares csv for the neural network to train on
# this file is for reference only, it is already executed
import numpy as np
import pandas as pd

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

from nltk import word_tokenize
from nltk import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# create dataframes from csv
apis_df = pd.read_csv('../datasets/apis_processed.csv')
apis_df.fillna(' ', inplace = True)

mashups_df = pd.read_csv('../datasets/mashups_processed.csv')

# return APIs with top scores in the top k related mashups of the query
def recommend_apis_from_mashups(query_index, k = 15):
  # sort the top related mashups descending using cosine similarity score
  # then pick the top k elements, skip the first mashup because it's the query itself
  score_series = pd.Series(
    mashup_cos_sim_matrix[query_index]
  ).sort_values(ascending = False).iloc[1 : k + 1]

  # get all API id's used in top k related mashups
  apis_in_top_k_mashups = []

  for i in range(k):
    # retrieve a list of API id's from mashups_df
    api_list = mashups_df.iloc[score_series.index[i]]['api_list'].split(';')
    api_list = [int(api) for api in api_list]

    apis_in_top_k_mashups.append(api_list)

  # filter out repeated APIs from apis_in_top_k_mashups
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

  # each element in api_scores will be API's id and its score which is its count divided by k
  api_scores = []

  for api in all_apis:
    api_scores.append({
      'id': api,
      'score': count[api] / k
    })

  return api_scores


# initialize tf-idf vectorizer
tf_idf = TfidfVectorizer(analyzer = str.split, max_df = 0.85, max_features = 5000)

# apply tf-idf transformation
api_description_matrix = tf_idf.fit_transform(apis_df['description_words']).toarray()
mashup_description_matrix = tf_idf.fit_transform(mashups_df['description_words']).toarray()

# calculate cosine similarity matrix
mashup_cos_sim_matrix = cosine_similarity(
  mashup_description_matrix,
  mashup_description_matrix
)

# final dataframe to be exported
training_df = pd.DataFrame(columns = ['mashup_vec', 'api_id', 'api_vec', 'top_15_score', 'top_20_score'])

# for each mashup vector
for i in range(len(mashup_description_matrix)):
  # get APIs and their scores in top 15 and top 20 similar mashups
  api_scores_15 = recommend_apis_from_mashups(i, 15)
  api_scores_20 = recommend_apis_from_mashups(i, 20)

  # pairs of mashup and API vectors will be x
  # top 15 and top 20 scores will be y
  temp_df = pd.DataFrame(columns = ['mashup_vec', 'api_id', 'api_vec', 'top_15_score', 'top_20_score'])

  for api_scores in api_scores_15:
    temp_df = temp_df.append({
      # current mashup vector
      'mashup_vec': mashup_description_matrix[i],
      # each API in api_scores_15 and its vector and score
      'api_id': api_scores['id'],
      'api_vec': api_description_matrix[api_scores['id'] - 1],
      'top_15_score': api_scores['score']
    }, ignore_index = True)

  for api_scores in api_scores_20:
    # if the API is already in temp_df
    if api_scores['id'] in temp_df['api_id'].values:
      # update its top_20_score column
      temp_df.loc[temp_df['api_id'] == api_scores['id'], ['top_20_score']] = api_scores['score']
    else:
      # else create a new row
      temp_df = temp_df.append({
        'mashup_vec': mashup_description_matrix[i],
        'api_id': api_scores['id'],
        'api_vec': api_description_matrix[api_scores['id'] - 1],
        'top_20_score': api_scores['score']
      }, ignore_index = True)

  # now the final df is updated with rows that contain the ith mashup vector
  training_df = pd.concat([training_df, temp_df], ignore_index = True)

# fill missing score value with 0
training_df.fillna(0, inplace = True)

# export csv
training_df.to_csv('../datasets/training_sets.csv', index = False)