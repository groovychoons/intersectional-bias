
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# NAMES

african_female = ['Yvette', 'Aisha', 'Malika', 'Latisha', 'Keisha', 'Tanisha',
                  'Tamika', 'Yolanda', 'Nichelle', 'Latoya', 'Lakisha', 'Shereen', 'Shaniqua',
                  'Jasmine', 'Tia'][0:12]
# delete:  Kenya

african_male = ['Lionel', 'Wardell',  'Jamel', 'Marcellus',
                'Jamal', 'Leroy',  'Tyrone', 'Alphonse', 'Hakim', 'Terrence', 'Jerome', 'Alonzo'][0:12]
# delete: Deion, Ebony, Kareem,Lamar,Lavon,Malik,Rasheed,Jermaine,
# Tyree,Lamont,Darnell,Torrance,Theo

european_female = ['Melanie', 'Colleen', 'Ellen', 'Emily', 'Sarah', 'Rachel',
                   'Carrie', 'Stephanie', 'Megan', 'Nancy', 'Katie', 'Heather', 'Betsy',
                   'Kristin'][0:12]
#delete: Amanda

european_male = ['Frank',   'Roger', 'Neil', 'Geoffrey',
                 'Brad', 'Stephen', 'Peter',   'Jack',
                 'Matthew', 'Jonathan', 'Josh', 'Andrew', 'Greg',
                 'Justin', 'Alan',    'Adam',
                 'Harry',  'Paul'][0:12]
# delete: Lauren,Jill,Brendan,Meredith,Allison,Todd,Ryan,Courtney,Laurie,Brett,Anne

mexican_female = ['Maria', 'Yesenia', 'Adriana', 'Liset', 'Mayra', 'Alma',
                  'Carolina', 'Iliana', 'Sonia',
                  'Karina', 'Alejandra', 'Brenda', 'Vanessa', 'Diana'][0:12]
# delete: Ana
mexican_male = ['Jesús', 'Rigoberto', 'César', 'Rogelio', 'José', 'Pedro',
                'Antonio', 'Alberto', 'Alejandro',
                'Alfredo', 'Juan', 'Miguel', 'Ricardo'][0:12]
# delete: Angel,Jorge


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


# def get_effectsize(wd, A, B):
#     cos_a = cosine_similarity(wd, A)  # .mean()
#     cos_b = cosine_similarity(wd, B)  # .mean()
#     cos_ab = np.concatenate((cos_a, cos_b), axis=1)
#     delta_mean = cos_a.mean() - cos_b.mean()
#     std = np.std(cos_ab, ddof=1)
#     return delta_mean/std

def get_vectors(client, list):
    pass
    # input - client and list of words
    # output - numpy array of vectors corresponding to list of words

    # init empty numpy array
    # 

def get_effectsize(client, wd, A, B):
    cos_a = client.wv.similarity(wd, A)  # .mean()
    cos_b = client.wv.similarity(wd, B)  # .mean()
    print(cos_a)
    print(cos_b)
    cos_ab = np.concatenate((cos_a, cos_b), axis=1)
    delta_mean = cos_a.mean() - cos_b.mean()
    std = np.std(cos_ab, ddof=1)
    return delta_mean/std

print(african_female)
from gensim.models import KeyedVectors

client = KeyedVectors.load("../models/vectors_phrasal.kv")

get_effectsize(client, 'table','chair','adam')
