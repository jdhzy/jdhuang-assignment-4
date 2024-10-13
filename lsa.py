from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

class LSA:
    def __init__(self, n_components=100):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        self.svd = TruncatedSVD(n_components=n_components)
        self.documents = None
        self.X_lsa = None

    def fit(self, documents):
        self.documents = documents
        # Transform the documents into a TF-IDF matrix
        X = self.vectorizer.fit_transform(documents)
        # Apply SVD (LSA) to reduce dimensionality
        self.X_lsa = self.svd.fit_transform(X)

    def query(self, query_text):
        # Transform the query into the same TF-IDF space
        query_vector = self.vectorizer.transform([query_text])
        # Project the query into the LSA space
        query_lsa = self.svd.transform(query_vector)
        # Compute cosine similarity between the query and all documents
        similarities = cosine_similarity(query_lsa, self.X_lsa)[0]
        return similarities

    def get_top_documents(self, query_text, top_n=5):
        # Get cosine similarities between query and all documents
        similarities = self.query(query_text)
        # Get the indices of the top N most similar documents
        top_indices = similarities.argsort()[-top_n:][::-1]

        # Convert the document indices (top_indices) to Python int
        top_docs = [(self.documents[i], similarities[i], int(i)) for i in top_indices]
        return top_docs