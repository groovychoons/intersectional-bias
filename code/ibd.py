
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_effectsize(wd,A,B):
    cos_a = cosine_similarity(wd,A) #.mean()
    cos_b = cosine_similarity(wd,B) #.mean()
    cos_ab = np.concatenate((cos_a,cos_b),axis=1)
    delta_mean =   cos_a.mean() - cos_b.mean()
    std = np.std(cos_ab,ddof=1)
    return delta_mean/std

