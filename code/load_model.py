
import string as s
import os
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

def create_news_model(tokenized_articles, phrasal):

    # train model
    model = Word2Vec(tokenized_articles, workers=10, vector_size=300)
    # summarize the loaded model
    if not os.path.exists('models'):
        print("Creating a models directory")
        os.mkdir('models')
    
    if phrasal:
        model.save("./models/word2vec_phrasal.model")
        model.save('./models/vectors_phrasal.kv')
    else:
        model.save("./models/word2vec_single.model")
        model.save('./models/vectors_single.kv')

def load_news_model():
    client = KeyedVectors.load("./models/vectors_phrasal.kv")
    return client


create_news_model(True)
create_news_model(False)
