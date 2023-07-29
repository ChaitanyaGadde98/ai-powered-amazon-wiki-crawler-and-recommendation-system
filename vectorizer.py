from transformers import AutoTokenizer, AutoModel
import torch
import gensim
from gensim.models import Word2Vec
import numpy as np
import gensim.downloader as api


def get_text_vector(text):
    # loading the pre-trained transformer model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_input)
    vector = model_output.pooler_output.squeeze(0).numpy()

    return vector


def vectorize_text_w2v(text):
    # Load the Word2Vec model
    custom_model_path = "models/custom_word2vec_model.bin"
    model = api.load("word2vec-google-news-300")
    # model = Word2Vec.load(custom_model_path)

    # preprocess the text (tokenize, remove punctuation, lowercase, etc.)
    words = gensim.utils.simple_preprocess(text)
    vectorized_text = [model[word] for word in words if word in model]

    if vectorized_text:
        return np.mean(vectorized_text, axis=0)
    else:
        return None
