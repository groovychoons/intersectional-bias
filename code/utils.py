
import numpy as np

def get_vectors(client, w_list, names=False, return_words=False):
    # input - client, list of words, whether words are names
    # output - numpy array of vectors corresponding to list of words

    # init empty list
    words = []
    vocab = list(client.key_to_index.keys())

    # check if word in vocab
    for a in w_list:
        if names:
            if a.capitalize() in vocab:
                words.append(a.capitalize())
        else:
            if a in vocab:
                words.append(a)
    #print(words)

    # get vectors of words
    vectors = []
    for item in words:
        vectors.append(client.get_vector(item))

    # turn names to numpy
    vectors = np.array(vectors, dtype=float)
    print(vectors.shape)
    if return_words:
        return vectors, words
    else:
        return vectors
