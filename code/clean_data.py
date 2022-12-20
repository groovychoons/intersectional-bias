import nltk
import string as s

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = nltk.stem.WordNetLemmatizer()

replace_list = ["white woman", 
                "white man", 
                "caucasian woman", 
                "caucasian man", 
                "black woman", 
                "black man", 
                "african american woman", 
                "african american man", 
                "african-american woman", 
                "african-american man", 
                "south asian woman", 
                "south asian man",
                "east asian woman", 
                "east asian man",
                "asian american woman",
                "asian american man",
                "asian woman", 
                "asian man", 
                "mexican woman", 
                "mexican man", 
                "latina woman",
                "latino woman",
                "latino man",
                "latinx woman",
                "latinx man",
                "middle eastern woman",
                "middle eastern man",
                "arab woman",
                "arab man",
                "japanese woman",
                "japanese man",
                "korean woman",
                "korean man"
                ]


# Tokenises a string and adds lowercase tokens to list
# Lemmatizes
# Adds phrases using underscore
def get_sent_tokens(sentence):
    list_tokens = []
    # make lowercase
    sentence = sentence.lower()
    # tokenize
    list_tokens_sentence = nltk.tokenize.word_tokenize(sentence)
    # lemmatize
    for token in list_tokens_sentence:
        list_tokens.append(lemmatizer.lemmatize(token))

    # join list of tokens as one string
    sentence_joined = " ".join(list_tokens)
    # replace single words with chosen phrases above
    for j, word in enumerate(replace_list):
        word = word.lower()
        if word.lower() in sentence_joined:
            sentence_joined = sentence_joined.replace(word, word.replace(" ", "_"))
    # split back into word tokens
    sentence_split = sentence_joined.split(" ")
    return sentence_split

# Same as above but without phrases
def get_sent_tokens_single(sentence):
    list_tokens = []
    sentence = sentence.lower()
    list_tokens_sentence = nltk.tokenize.word_tokenize(sentence)
    for token in list_tokens_sentence:
        list_tokens.append(lemmatizer.lemmatize(token))
    return list_tokens

# Removes punctuation from list of tokens
def remove_punctuations(lst):
    new_lst = []
    for i in lst:
        for j in s.punctuation:
            if j == "-" or j == "_":
                pass
            else:
                i = i.replace(j, '')
        new_lst.append(i)
    return new_lst


def clean_article(data, phrasal):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    stopwords.update(["", "ha", "said", "wa", "nt", "would", "also", "could"])
    not_stopwords = ["he", "she", "she's", "he's", "herself", "himself", "her", "his", "hers", "him"]
    stopwords.difference_update(not_stopwords)
    final_article = []

    for sentence in data:
        sentence_tokens = []
        if phrasal:
            sentence_tokens = get_sent_tokens(sentence)
        else:
            sentence_tokens = get_sent_tokens_single(sentence)
        sentence_tokens = remove_punctuations(sentence_tokens)
        final_sentence = []
        for word in sentence_tokens:
            if word in stopwords:
                continue
            final_sentence.append(word)
        final_article.append(final_sentence)

    return final_article

def main(phrasal):
    print("Running script")

    rawdata = []
    with open("../data/testset.en.shuffled.deduped") as infile:
        print("Data loaded")
        for line in infile:
            rawdata.append(line)

    print("No. of sentences: ", len(rawdata))

    tokenized_articles = clean_article(rawdata, phrasal)
    print("Articles tokenized")

    return tokenized_articles