
import gensim.downloader as api
import numpy as np
from sklearn import svm
from utils import get_vectors
import data

# import model
client = api.load("word2vec-google-news-300")
print("Client downloaded")

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

# Get Y vector, splitting all names into their categories
Y1 = []

count = -1
for i in range(0,105):
    if i % 15 == 0:
        count += 1
    Y1.append(count)

print(len(Y1))

# Fit the SVM
clf = svm.SVC()
clf.fit(X1, Y1)

# Check correct classification of biases

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
print(X2_key)

predictions = clf.predict(X2)
print(predictions)

predictions_key = ['af_fem', 'af_male', 'eu_fem', 'eu_male', 'mex_fem', 'mex_male', 'mid_east_male',]

for i, word in enumerate(X2_key):
    print(word, predictions[i], predictions_key[predictions[i]])
