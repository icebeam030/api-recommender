# The algorithm part

import numpy as np
import pandas as pd

import nltk
# Download required packages
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

from nltk import word_tokenize
from nltk import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class APIRecommender:
    def __init__(self):
        # Create dataframes from csv, used to retrieve relevant info
        self.apis_df = pd.read_csv('../datasets/apis_processed.csv')
        self.mashups_df = pd.read_csv('../datasets/mashups_processed.csv')

        # Updated when query is updated, used by recommendation function
        self.mashup_cos_sim_vector = []

    def process_text(self, text):
        # Create a text filter list of English stopwords and special characters
        text_filter = [stopwords.words('english')]
        text_filter.extend([',', '/', '-', '.', ';'])

        # Initialize text stemmer
        porter = PorterStemmer()

        processed_text = ''

        # Tokenize text
        tokens = word_tokenize(str(text))
        for token in tokens:
            # Remove English stopwords and special characters from each token
            if token not in text_filter:
                # Stem each token
                processed_text += porter.stem(token.lower())
                processed_text += ' '

        return processed_text

    def add_query(self, query, max_df=0.85, max_features=5000):
        # Add a new row for the query to the end of the dataframe.
        # Original df is unchanged.
        mashups_df = self.mashups_df.append({
            'description_words': self.process_text(query)
        }, ignore_index=True)

        # Initialize tf-idf vectorizer
        tf_idf = TfidfVectorizer(
            analyzer=str.split, max_df=max_df, max_features=max_features
        )

        # Apply tf-idf transformation
        mashup_description_matrix = tf_idf.fit_transform(
            mashups_df['description_words']
        ).toarray()

        # Calculate cosine similarity vector
        self.mashup_cos_sim_vector = []

        query_index = mashups_df.shape[0] - 1
        for i in range(query_index):
            cos_sim_score = cosine_similarity(
                [mashup_description_matrix[i]],
                [mashup_description_matrix[query_index]]
            )
            self.mashup_cos_sim_vector.append(cos_sim_score)

    def recommend_apis_from_mashups(self, k=15):
        """Return APIs with top counts in top k similar mashups to the query"""

        # Sort the top related mashups descending by cosine similarity score.
        # Then pick the top k elements.
        score_series = pd.Series(
            self.mashup_cos_sim_vector
        ).sort_values(ascending=False).iloc[0 : k]

        # Get all API id's used in top k related mashups
        apis_in_top_k_mashups = []
        for i in range(k):
            # Retrieve a list of API id's from mashups_df,
            # and convert it to a set
            api_set = set(
                self.mashups_df.iloc[score_series.index[i]]['api_list'].split(';')
            )
            api_set = {int(api) for api in api_set}

            apis_in_top_k_mashups.append(api_set)

        # Filter out repeated API id's in apis_in_top_k_mashups
        all_apis = set()
        for api_set in apis_in_top_k_mashups:
            all_apis = all_apis | api_set

        # For each API, if it is used in one of the top k related mashups,
        # increment its count by 1.
        count = {}
        for api in all_apis:
            count[api] = 0
            for api_set in apis_in_top_k_mashups:
                if api in api_set:
                    count[api] += 1

        api_counts = []
        for api in all_apis:
            api_counts.append({
                'id': api,
                'count': count[api]
            })

        # Sort api_counts descending by count.
        # Only recommend up to 10 APIs.
        if (len(api_counts) < 10):
            api_counts = sorted(
                api_counts, reverse=True, key=lambda item: item['count']
            )
        else:
            api_counts = sorted(
                api_counts, reverse=True, key=lambda item: item['count']
            )[: 10]

        # Return a dictionary to be displayed on web page
        recommendations = []

        for api_count in api_counts:
            # Retrieve corresponding row from apis_df
            api_row = self.apis_df.iloc[api_count['id'] - 1].values

            recommendations.append({
                'name': api_row[1],
                'categories': api_row[2],
                'description': api_row[3],
                'url': api_row[4],
                'count': api_count['count']
            })

        return recommendations


# The server part

from flask import Flask, render_template, request
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        max_df = float(request.values.get("max_df"))
        max_features = int(request.values.get("max_features"))

        query = request.values.get("query")
        recommender.add_query(query, max_df, max_features)

        apis = recommender.recommend_apis_from_mashups()
        return render_template("index.html", apis=apis, search="mashup")

    return render_template("index.html")


if __name__ == "__main__":
    recommender = APIRecommender()
    print("Starting server...")
    app.run(debug=True)
