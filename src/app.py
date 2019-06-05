from flask import Flask, render_template, request
app = Flask(__name__)

from src.api_recommender import APIRecommender
api_recommender = APIRecommender()


@app.route("/", methods = ["GET", "POST"])
def home():
  if request.method == "POST":
    query = request.values.get("query")
    api_recommender.add_query(query)

    if request.values.get("search") == "mashup":
      apis = api_recommender.recommend_apis_from_mashups(query)
      return render_template("index.html", apis = apis)

    if request.values.get("search") == "api":
      apis = api_recommender.recommend_apis_from_apis(query)
      return render_template("index.html", apis = apis)

  return render_template("index.html")


if __name__ == "__main__":
  app.run(debug = True)