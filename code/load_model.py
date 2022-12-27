
from gensim.models import KeyedVectors

def main():
    client = KeyedVectors.load("./models/vectors_phrasal.kv")
    return client
