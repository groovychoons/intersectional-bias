
from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# NAMES

african_female = ['aisha', 'lashelle', 'nichelle', 'shereen', 'temeka', 'ebony', 'latisha', 'shaniqua',
                  'tameisha', 'teretha', 'jasmine', 'latonya', 'shanise', 'tanisha', 'tia', 'lakisha',
                  'latoya', 'sharise', 'tashika', 'yolanda', 'lashandra', 'malika', 'shavonn',
                  'tawanda', 'yvette']

african_male = ['alonzo', 'jamel', 'lerone', 'percell', 'theo', 'alphonse', 'jerome', 'leroy', 'rasaan',
                'torrance', 'darnell', 'lamar', 'lionel', 'rashaun', 'vree', 'deion', 'lamont', 'malik',
                'terrence', 'qrone', 'everol', 'lavon', 'marcellus', 'terryl', 'wardell']

european_female = ['amanda', 'courtney', 'heather', 'melanie', 'sara', 'amber', 'crystal', 'katie',
                   'meredith', 'shannon', 'betsy', 'donna', 'kristin', 'nancy', 'stephanie', 'bobbie-sue',
                   'ellen', 'lauren', 'peggy', 'sue-ellen', 'colleen', 'emily', 'megan', 'rachel', 'wendy']

european_male = ['adam', 'chip', 'harry', 'josh', 'roger', 'alan', 'frank', 'ian', 'justin', 'ryan', 'andrew',
                 'fred', 'jack', 'matthew', 'stephen', 'brad', 'greg', 'jed', 'paul', 'todd', 'brandon', 'hank', 
                 'jonathan', 'peter', 'wilbur']

mexican_female = ['Maria', 'Yesenia', 'Adriana', 'Liset', 'Mayra', 'Alma',
                  'Carolina', 'Iliana', 'Sonia',
                  'Karina', 'Alejandra', 'Brenda', 'Vanessa', 'Diana', 'Ana']

mexican_male = ['Jesús', 'Rigoberto', 'César', 'Rogelio', 'José', 'Pedro',
                'Antonio', 'Alberto', 'Alejandro',
                'Alfredo', 'Juan', 'Miguel', 'Ricardo', 'Angel', 'Jorge']

# BIAS
african_common_bias = ['ghetto', 'unrefined', 'criminals', 'athletic', 'loud',
                       'gangsters', 'poor', 'uneducated', 'dangerous', 'violent', 'tall', 'lazy']

african_male_intersectional_bias = ['darkskinned', 'rapper', 'hypersexual']

african_male_bias = ['athletic', 'loud', 'tall',
                     'violent', 'dangerous', 'poor', 'unintelligent',
                     'gangsters', 'lazy', 'criminals']+african_male_intersectional_bias

african_female_intersectional_bias = ['bigbutt', 'overweight', 'confident', 'darkskinned', 'assertive',
                                      'promiscuous', 'unfeminine', 'aggressive', 'fried-chicken']

african_female_bias = ['loud', 'ghetto', 'unrefined',
                       'athletic', 'unintelligent']+african_female_intersectional_bias

european_common_bias = ['high-status', 'rich', 'intelligent', 'arrogant', 'privileged', 'blond', 'racist', 'all-American',
                        'ignorant', 'red-neck', 'attractive', 'tall', 'patronizing', 'blue-eyes', 'overweight']

european_male_intersectional_bias = ['assertive', 'successful', 'educated']
european_male_bias = ['rich', 'intelligent', 'arrogant', 'high-status', 'blond', 'racist', 'all-American',
                      'leader', 'privileged', 'attractive', 'tall', 'sexist'] + european_male_intersectional_bias

european_female_intersectional_bias = ['ditsy']  # ,'sexually-liberal'
european_female_bias = ['arrogant', 'blond', 'rich', 'attractive', 'petite', 'tall', 'materialistic', 'racist',
                        'intelligent', 'feminine', 'emotional', 'submissive', 'high-status'] + european_female_intersectional_bias

male_common_bias = ['tall', 'respected', 'intelligent', 'high-status', 'leader', 'sexist', 'provider',
                    'aggressive', 'unfaithful', 'ambitious', 'arrogant', 'messy', 'fixer-upper']

female_common_bias = ['emotional', 'caring', 'soft', 'talkative', 'petite', 'submissive', 'dependent', 'motherly',
                      'feminine', 'manipulative', 'attractive', 'materialistic', 'jealous']


mexican_female_intersectional_bias = [
    'feisty', 'curvy', 'cook', 'promiscuous', 'sexy', 'maids']

mexican_female_bias = ['feisty', 'curvy', 'loud', 'attractive', 'cook', 'darkskinned',
                       'uneducated', 'hardworker', 'promiscuous', 'unintelligent', 'short', 'sexy', 'maids']

mexican_male_intersectional_bias = [
    'promiscuous', 'jealous', 'violent', 'drunks']

mexican_male_bias = ['macho', 'poor', 'darkskinned', 'day-laborer', 'promiscuous', 'short', 'hardworker',
                     'jealous', 'uneducated', 'illegal-immigrant', 'arrogant', 'unintelligent', 'aggressive', 'violent', 'drunks']


mexican_common_bias = ['poor', 'illegal-immigrant', 'darkskinned', 'uneducated', 'family-oriented',
                       'lazy', 'day-laborer', 'unintelligent', 'loud', 'gangster', 'short', 'overweight', 'macho', 'hardworker']


def get_effectsize(wd, A, B):
    cos_a = cosine_similarity(wd, A)  # .mean()
    cos_b = cosine_similarity(wd, B)  # .mean()
    cos_ab = np.concatenate((cos_a, cos_b), axis=1)
    delta_mean = cos_a.mean() - cos_b.mean()
    std = np.std(cos_ab, ddof=1)
    return delta_mean/std


def get_vectors(client, names):
    # input - client and list of words
    # output - numpy array of vectors corresponding to list of words

    # init empty list
    words = []
    vocab = list(client.wv.key_to_index.keys())

    # check if word in vocab
    for a in names:
        if a.lower() in vocab:
            words.append(a.lower())
    print(words)

    # get vectors of words
    vectors = []
    for item in words:
        vectors.append(client.wv.get_vector(item))

    # turn names to numpy
    vectors = np.array(vectors, dtype=float)
    print(vectors.shape)
    return vectors


client = KeyedVectors.load("../models/vectors_phrasal.kv")

#get_effectsize(client, 'table','chair','adam')
af = get_vectors(client, african_female)
am = get_vectors(client, african_male)

vocab = list(client.wv.key_to_index.keys())

biases = []
for word in african_male_bias:
    if word in vocab:
        bias = get_effectsize(
            np.array([client.wv.get_vector(word)]), am[0:5], af[0:5])
        biases.append((word, bias))

biases.sort(key=lambda x: x[1])
for bias in biases:
    print(bias)

#print(af_names.lower().split(", "))
