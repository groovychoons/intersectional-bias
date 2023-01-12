# Quantifying Intersectional Biases in Static Word Embeddings

This project trains a Word2Vec model, incorporating phrases to represent multi-word identities, such as "black_woman" or "mexican_woman". 

We evaluate whether more bias is detected using these phrases or with names (e.g. European American and African American names) using IBD and EIBD [(Guo and Caliskan, 2021)](https://arxiv.org/abs/2006.03955) as evaluation metrics.

We also build on Guo and Caliskan by looking at Asian American and Middle Eastern American stereotypes. The names for these are found using name clustering, building on the work of [Swinger et. al (2018)](https://arxiv.org/abs/1812.08769).

This repository provides:
- Code to train a Word2Vec model incorporating phrases to repesent intersectional identities
- Code to train a linear SVM and K-means clustering models to cluster names
- Code to implement IBD and EIBD methods on identities represented by both phrases (e.g. "white man") and names (e.g. "Adam")

The results will show which intersectional and emergent intersectional biases there are for Asian Americans, Blacks, Latinos, Middle Eastern Americans, or Whites, and whether the stereotypical bias for these groups is held more strongly using terms such as "white man" or names to represent that group.

## Key Papers

- [Distributional techniques for philosophical enquiry (Herbelot et. al, 2012)](https://aclanthology.org/W12-1008.pdf)
More info about this paper - first to look at phrases over multiplicative model - before SWE were a thing but compares using phrase 'black_woman' built into a word distribution, rather than taking the sum of its parts (black x woman)

- [What are the biases in my word embedding? (Swinger et. al, 2018)](https://arxiv.org/abs/1812.08769)
Name clustering - trains a linear SVM to remove the 20% of names most likely to be misconstrued as words (e.g. April, June). Then uses K-means++ clustering to cluster names into groups.

- [Greenwald]()
Has black, european, japanese, korean names

- [Detecting Emergent Intersectional Biases: Contextualized Word Embeddings Contain a Distribution of Human-like Biases (Guo and Caliskan, 2021)](https://arxiv.org/abs/2006.03955)
Provides EIBD and IBD, and Mexican names

- [An Intersectional Analysis of Gender and Ethnic Stereotypes: Testing Three Hypotheses (Ghavami and Peplau, 2013)](https://journals.sagepub.com/doi/epub/10.1177/0361684312464203)
Provides the stereotypes used for IBD and EIBD

## Corpus

- WMT News Crawl
    - Source: https://data.statmt.org/news-crawl/en/
    - Description: Text extracted from online newspapers
    - Extraction: 2007-2021 combined. Cleaned, tokenized, lemmatized.
    - Version: One sentence per line (add link?)
    - Size: 314m sentences

- Add Wikipedia and UMBC, Common Crawl?

## Installation and setup

[NEED TO REDO THE ENVIRONMENT FOR IT TO WORK]

After you clone this repository:

- Navigate to it using `cd intersectional-bias`
- To create an environment with all necessary dependencies use `conda env create ./environment.yml`
- Activate the environment with `conda activate debias`

## Training the model

This code trains a Word2Vec model on the WMT news dataset of 314 million sentences. 
The training includes terms for identifying the race subspace, such as "black_woman", "african_american" and "black_teenager". [load_model file](debias/load_model_script.py)

```bash
python3 debias
```

## Pretrained model

blah blah blah 

## Lists of names and stereotypes

```bash
python3 name_clustering.py
```

Once the model is trained, we find both the race and gender subspaces using the difference between various pairs, such as he -> she and white_man -> black_man, and then computing their principle components. [find_space file](debias/find_space_kv.py)

We also calculate the race subspace by using the difference between name pairs (e.g. adam -> tyrone) to compare to the phrasal model of calculating the race subspace. [find_space file](debias/find_space_kv.py)

## IBD and EIBD

```bash
python3 ibd.py
```

Success and evaluation metrics

## Results

TBD

## Why are we doing it?

Research questions:
- Can we better represent bias for intersectional identities using phrases (instead of single words/names) within word embedding models?
- Can we better understand intersectional stereotypes through the use of these phrases?
- Can we understand intersectional biases for more groups of people?

Novel contributions:
- Using phrases, allows us to look at identities previously not able to
- Detecting intersectional bias for Asian and Middle Eastern Americans
