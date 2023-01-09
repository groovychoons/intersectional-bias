"""
The names are accumulated from 3 data sources. Duplicates are removed. Names not in
the top 50,000 most frequent words of the google-news Word2Vec vocabulary are removed.
Our cleaning procedure is similar but slightly more sophisticated to that of [7]. 
We train a linear Support Vector Machine [scikit-learn’s LinearSVC, 19, with 
default parameters] to distinguish the input names from an equal number of non-names 
chosen randomly from the most frequent 50,000 words in the embedding. 
We then remove the 20% of names with smallest margin in the direction identified by 
the linear classifier.
We then use K-means++ clustering [from scikit-learn, 19, with default parameters] to 
cluster the normalized word vectors of the names, yielding groups X1 ∪ . . . ∪ Xn = X. 
Finally, we define µ = Í i Xi /n.
"""

import csv
import numpy as np
import gensim.downloader as api
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA

# we have 3623 names and 3623 random non-name words
# import words
with open('../data/names_in_top_50k.csv', newline='') as f:
    reader = csv.reader(f)
    names = list(reader)

print(names[:10])

with open('../data/words_in_top_50k.csv', newline='') as f:
    reader = csv.reader(f)
    words = list(reader)

print(words[:10])

# import model
client = api.load("word2vec-google-news-300")
print("Client downloaded")

# check if words are in vocab

def check_vocab(client, words):
    words2 = []
    if check_vocab:
        vocab = list(client.key_to_index.keys())
        # check if word in vocab
        for a in words:
            if a[0] in vocab:
                words2.append([a[0]])
    return(words2)

# get vectors for each word
def get_vectors(client, words):
    # input - client and list of words
    # output - numpy array of vectors corresponding to list of words

    # get vectors of words
    vectors = []
    for item in words:
        vectors.append(client.get_vector(item[0]))

    # turn words to numpy
    vectors = np.array(vectors, dtype=float)
    print(vectors.shape)
    return vectors

# train a linear Support Vector Machine [scikit-learn’s LinearSVC]
X = np.append(get_vectors(client, names), get_vectors(client, words), axis=0)
print("X shape: ", X.shape)
Y = np.append(np.ones((3263,), dtype=int), np.zeros((3263,), dtype=int))
print("Y shape: ", Y.shape)

clf = svm.LinearSVC()
clf.fit(X,Y)

# Test if it's working:
# print(clf.predict([client.get_vector('Jessica')]))
# print(clf.predict([client.get_vector('chair')]))

# remove the 20% of names with smallest margin in the direction identified by  the linear classifier
margins = []
for name in names:
    vector = [client.get_vector(name[0])]
    margins.append([name[0], clf.predict(vector), clf.decision_function(vector)])

sorted_margins = sorted(margins, key=lambda x: x[2])
print(sorted_margins[0:10])

new_names = [[x[0]] for x in sorted_margins[653:]]
print(len(new_names))
print(new_names[0:10])

# use K-means++ clustering [from scikit-learn]
# to cluster the normalized word vectors of the names
kmeans = KMeans(n_clusters=8)
Z = get_vectors(client, new_names)
kmeans.fit(Z)

distances = kmeans.transform(Z)
print(distances)

y_kmeans = kmeans.predict(Z)
print(y_kmeans[0:10])
print(len(y_kmeans))


# import bias words
with open('../data/biases.csv', newline='') as f:
    reader = csv.reader(f)
    biases = list(reader)

# remove duplicates
new_biases = []
for bias in biases:
    if bias not in new_biases:
        new_biases.append(bias)

new_biases2 = check_vocab(client, new_biases)

print("bias len: ", len(biases))
print(len(new_biases))
print(len(new_biases2))
bias_vectors = get_vectors(client, new_biases2)

B = get_vectors(client, new_names + new_biases2)

# Use t-SNE to visualise clusters
tsne = KernelPCA(n_components=2, kernel='rbf')
Z_tsne = tsne.fit_transform(B)

with open('../data/clusters.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f, delimiter=',')

    for i in range(len(new_names)):
        write.writerow([new_names[i][0], y_kmeans[i], Z_tsne[i][0], Z_tsne[i][1], distances[i][y_kmeans[i]]])


with open('../data/bias_tsne.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f, delimiter=',')

    for i in range(len(new_biases2)):
        write.writerow([new_biases2[i][0], Z_tsne[len(new_names) + i][0], Z_tsne[len(new_names) + i][1]])

