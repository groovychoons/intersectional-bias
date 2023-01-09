
from gensim.models import KeyedVectors
import gensim.downloader as api

def main():
    client = KeyedVectors.load("./models/vectors_phrasal.kv")
    return client

def load_google_news():
    client = api.load("word2vec-google-news-300")
    return client
