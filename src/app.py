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
        self.mashup_cos_sim_matrix = []

    def process_text(self, text):
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
            # Remove English stopwords and special characters from each token
            if token not in text_filter:
                # Stem each token
                result += porter.stem(token.lower())
                result += ' '

        return result

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

        # Calculate cosine similarity matrix
        self.mashup_cos_sim_matrix = cosine_similarity(
            mashup_description_matrix,
            mashup_description_matrix
        )

        # Return query's index to be passed into recommendation function
        query_index = mashups_df.shape[0] - 1

        return query_index

    def recommend_apis_from_mashups(self, query_index, k=15):
        """Return APIs with top counts in top k similar mashups to the query"""

        # Sort the top related mashups descending by cosine similarity score.
        # Then pick the top k elements.
        # Skip the first element because it's the query itself.
        score_series = pd.Series(
            self.mashup_cos_sim_matrix[query_index]
        ).sort_values(ascending=False).iloc[1 : k+1]

        # Get all API id's used in top k related mashups
        apis_in_top_k_mashups = []
        for i in range(k):
            # Retrieve a list of API id's from mashups_df
            api_list = self.mashups_df.iloc[
                score_series.index[i]
            ]['api_list'].split(';')
            api_list = [int(api) for api in api_list]

            apis_in_top_k_mashups.append(api_list)

        # Filter out repeated API id's in apis_in_top_k_mashups
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

        if request.values.get("search") == "mashup":
            recommender = APIRecommender()

            query = request.values.get("query")
            query_index = recommender.add_query(query, max_df, max_features)

            apis = recommender.recommend_apis_from_mashups(query_index)
            return render_template("index.html", apis=apis, search="mashup")

    return render_template("index.html")


if __name__ == "__main__":
    print("Starting server...")
    app.run(debug=True)
