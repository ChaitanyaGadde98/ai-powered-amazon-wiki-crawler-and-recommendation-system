from transformers import AutoTokenizer, AutoModel
from sklearn.manifold import TSNE
import torch
import gensim
import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt


def get_bert_vector(text):
    # Loading the pre-trained transformer model and its tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        model_output = model(**encoded_input)

    vector = model_output.pooler_output.squeeze(0).numpy()
    return vector


def vectorize_text_w2v(text):
    # Load the default Word2Vec model
    model = api.load("word2vec-google-news-300")

    # Preprocessing the text (tokenize, remove punctuation, lowercase, etc.)
    words = gensim.utils.simple_preprocess(text)
    vectorized_text = [model[word] for word in words if word in model]

    if vectorized_text:
        return np.mean(vectorized_text, axis=0)
    else:
        return None


def plot_word_vectors(words, vectors, title):
    tsne = TSNE(n_components=2, random_state=42, perplexity=4)
    vectors_tsne = tsne.fit_transform(vectors)

    plt.figure(figsize=(10, 6))
    for i, word in enumerate(words):
        x, y = vectors_tsne[i, :]
        plt.scatter(x, y, color='blue')
        plt.annotate(word, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    plt.title(title)
    plt.savefig("visualizations/{}.png".format(title))
    plt.show()


if __name__ == "__main__":
    # List of words to compare and visualize

    sentences_list = ["good", "not good", "bad", "worst", "better", "best", "not best"]

    # BERT vectors
    bert_vectors = [get_bert_vector(word) for word in sentences_list]

    # Word2Vec vectors
    w2v_vectors = [vectorize_text_w2v(word) for word in sentences_list]

    print("t-SNE Visualization for BERT:")
    plot_word_vectors(sentences_list, np.array(bert_vectors), "BERT")

    print("t-SNE Visualization for Word2Vec:")
    plot_word_vectors(sentences_list, np.array(w2v_vectors), "Word2Vec")
