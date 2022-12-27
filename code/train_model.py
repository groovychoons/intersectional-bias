
import string as s
import os
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

def main(tokenized_articles):
    # train model
    model = Word2Vec(tokenized_articles, workers=10, vector_size=300)
    # save the model
    if not os.path.exists('models'):
        print("Creating a models directory")
        os.mkdir('models')
    
    model.save("./models/word2vec_phrasal.model")
    model.save('./models/vectors_phrasal.kv')
