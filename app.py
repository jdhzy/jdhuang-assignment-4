from flask import Flask, request, render_template
from sklearn.datasets import fetch_20newsgroups
from lsa import LSA

app = Flask(__name__)

# Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data  # These are the documents

# Initialize and fit the LSA model
lsa_model = LSA(n_components=100)
lsa_model.fit(documents)

@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    results = []
    if request.method == "POST":
        query = request.form['query']
        # Get document text, similarity score, and document index from the LSA model
        results = lsa_model.get_top_documents(query, top_n=5)

        # The results now contain (doc_text, similarity_score, doc_index)
        # Example: [('Document 1 text...', 0.89, 102), ('Document 2 text...', 0.76, 205), ...]

    return render_template("index.html", query=query, results=results)

if __name__ == "__main__":
    app.run(debug=True, port=3000)