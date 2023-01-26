
from gensim.models import KeyedVectors
import gensim.downloader as api
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_vectors
import data

client = api.load("word2vec-google-news-300")


# Get vectors for all names
# change this to a list of all names lists and for loop
#   which appends them one at a time, then use this to 
#   generate Y too, using the lengths of each sublist
af = get_vectors(client, data.african_female, True)
am = get_vectors(client, data.african_male, True)
ef = get_vectors(client, data.european_female, True)
em = get_vectors(client, data.european_male, True)
mf = get_vectors(client, data.mexican_female, True)
mm = get_vectors(client, data.mexican_male, True)
#mef = get_vectors(client, data.mide_female, True)
mem = get_vectors(client, data.mide_male, True)

X1 = np.concatenate((af, am, ef, em, mf, mm, mem), axis=0)
print(X1.shape)

X2_raw = [data.african_female_bias, data.african_male_bias, 
      data.european_female_bias, data.european_male_bias,
      data.mexican_female_bias, data.mexican_male_bias,
      data.mide_male_bias, ['Adam'], ['Jane']]

print(X2_raw)
X2 = []
X2_key = []
Y2 = []

for i, bias in enumerate(X2_raw):
    vectors, words = get_vectors(client, bias, False, True)
    X2.extend(vectors)
    X2_key.extend(words)
    for j in range(len(vectors)):
        Y2.append(i)

print(len(X2))
print(len(Y2))

def get_effectsize(wd, A, B):
    cos_a = cosine_similarity(wd, A)  # .mean()
    cos_b = cosine_similarity(wd, B)  # .mean()
    cos_ab = np.concatenate((cos_a, cos_b), axis=1)
    delta_mean = cos_a.mean() - cos_b.mean()
    std = np.std(cos_ab, ddof=1)
    return delta_mean/std


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
