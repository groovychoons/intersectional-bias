
from gensim.models import KeyedVectors
import gensim.downloader as api
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_vectors
import data

def get_effectsize(wd, A, B):
    cos_a = cosine_similarity(wd, A)  # .mean()
    cos_b = cosine_similarity(wd, B)  # .mean()
    cos_ab = np.concatenate((cos_a, cos_b), axis=1)
    delta_mean = cos_a.mean() - cos_b.mean()
    std = np.std(cos_ab, ddof=1)
    return delta_mean/std

client = api.load("word2vec-google-news-300")

# get_effectsize(client, 'table','chair','adam')
af = get_vectors(client, data.mexican_female, True)
am = get_vectors(client, data.mexican_male, True)

vocab = list(client.key_to_index.keys())

biases = []
for word in data.african_female_bias:
    if word in vocab:
        bias = get_effectsize(
            np.array([client.get_vector(word)]), am[0:14], af[0:14])
        biases.append((word, bias))

biases.sort(key=lambda x: x[1])
for bias in biases:
    print(bias)

# print(af_names.lower().split(", "))
