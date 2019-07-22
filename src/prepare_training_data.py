# This file is an idea only,
# looks like machine learning is not suitable for this project.
# This file prepares csv for the neural network to train on.

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

# Create dataframes from csv
apis_df = pd.read_csv('../datasets/apis_processed.csv')
apis_df.fillna(' ', inplace=True)

mashups_df = pd.read_csv('../datasets/mashups_processed.csv')


def recommend_apis_from_mashups(query_index, k=15):
    """Return APIs with top scores in top k similar mashups to the query"""

    # Sort the top related mashups descending using cosine similarity score.
    # Then pick the top k elements.
    # Skip the first element because it's the query itself.
    score_series = pd.Series(
        mashup_cos_sim_matrix[query_index]
    ).sort_values(ascending=False).iloc[1 : k+1]

    # Get all API id's used in top k related mashups
    apis_in_top_k_mashups = []
    for i in range(k):
        # Retrieve a list of API id's from mashups_df
        api_list = mashups_df.iloc[
            score_series.index[i]
        ]['api_list'].split(';')
        api_list = [int(api) for api in api_list]

        apis_in_top_k_mashups.append(api_list)

    # Filter out repeated APIs in apis_in_top_k_mashups
    all_apis = []
    for api_list in apis_in_top_k_mashups:
        for api in api_list:
            if api not in all_apis:
                all_apis.append(api)

    # For each API, if it is used in one of the top k related mashups,
    # increment its count by 1.
    count = {}
    for api in all_apis:
        count[api] = 0
        for api_list in apis_in_top_k_mashups:
            if api in api_list:
                count[api] += 1

    # Normalize score by dividing API count by k
    api_scores = []
    for api in all_apis:
        api_scores.append({
            'id': api,
            'score': count[api] / k
        })

    return api_scores

# Initialize tf-idf vectorizer
tf_idf = TfidfVectorizer(analyzer=str.split, max_df=0.85, max_features=5000)

# Apply tf-idf transformation
api_description_matrix = tf_idf.fit_transform(
    apis_df['description_words']
).toarray()
mashup_description_matrix = tf_idf.fit_transform(
    mashups_df['description_words']
).toarray()

# Calculate cosine similarity matrix
mashup_cos_sim_matrix = cosine_similarity(
    mashup_description_matrix,
    mashup_description_matrix
)

# Final dataframe to be exported
training_df = pd.DataFrame(
    columns=['mashup_vec', 'api_id', 'api_vec', 'top_15_score', 'top_20_score']
)

for i in range(len(mashup_description_matrix)):
    # Get APIs and their scores in their top 15 and top 20 similar mashups
    api_scores_15 = recommend_apis_from_mashups(i, 15)
    api_scores_20 = recommend_apis_from_mashups(i, 20)

    temp_df = pd.DataFrame(
        columns=[
            'mashup_vec', 'api_id', 'api_vec', 'top_15_score', 'top_20_score'
        ]
    )

    for api_score in api_scores_15:
        temp_df = temp_df.append({
            # Current mashup vector
            'mashup_vec': mashup_description_matrix[i],
            # Each API in api_scores_15 and its vector and score
            'api_id': api_score['id'],
            'api_vec': api_description_matrix[api_score['id'] - 1],
            'top_15_score': api_score['score']
        }, ignore_index=True)

    for api_score in api_scores_20:
        if api_score['id'] in temp_df['api_id'].values:
            # If the API is already in temp_df,
            # update its top_20_score column.
            temp_df.loc[
                temp_df['api_id'] == api_score['id'], ['top_20_score']
            ] = api_score['score']
        else:
            # Else create a new row
            temp_df = temp_df.append({
                'mashup_vec': mashup_description_matrix[i],
                'api_id': api_score['id'],
                'api_vec': api_description_matrix[api_score['id'] - 1],
                'top_20_score': api_score['score']
            }, ignore_index=True)

    # Now the final df is updated with rows that contain the ith mashup vector
    training_df = pd.concat([training_df, temp_df], ignore_index=True)

# Fill missing score values with 0
training_df.fillna(0, inplace=True)
