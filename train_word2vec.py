import re
import multiprocessing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from datasets import load_dataset


def preprocess_text(text):

    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # tokenize the text
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    wiki_terms = ["wikipedia", "wiki", "encyclopedia"]
    tokens = [token for token in tokens if token not in stop_words and token not in wiki_terms]

    # lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    processed_text = " ".join(tokens)

    return processed_text


def get_wiki_text():
    # Load dataset
    wikipedia_dataset = load_dataset('wikipedia', '20220301.simple')
    articles = wikipedia_dataset['train']['text']
    wikiText = ""
    for idx, article_text in enumerate(articles):
        wikiText += " " + article_text
    return wikiText


wikipedia_data = get_wiki_text()
amazon_data = " "

# Preprocess the data before combining
wikipedia_data = preprocess_text(wikipedia_data)
amazon_data = preprocess_text(amazon_data)

combined_data = wikipedia_data + amazon_data

print("LOG: Preprocessed the data.")

cores = multiprocessing.cpu_count()
w2v_model = Word2Vec(sentences=[combined_data.split()],
                     vector_size=100,
                     window=5,  # context window size
                     min_count=5,  # min frequency to consider a word
                     workers=cores,
                     sg=0)

w2v_model.save("models/custom_word2vec_model.bin")
