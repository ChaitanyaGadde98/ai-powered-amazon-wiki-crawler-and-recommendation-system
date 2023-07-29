from bs4 import BeautifulSoup
from selenium import webdriver
import re
import requests
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from vectorizer import get_text_vector, vectorize_text_w2v
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    wiki_terms = ["wikipedia", "wiki", "encyclopedia"]
    # text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = word_tokenize(text)
    # stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in wiki_terms]

    # lemmatizer = WordNetLemmatizer()
    # tokens = [lemmatizer.lemmatize(token) for token in tokens]
    processed_text = " ".join(tokens)

    return processed_text


def preprocess_text_wiki(text):
    # text = text.lower()
    # print("PREPROCESSED", text)
    text = text.split("history[edit]")[0]
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

    # Remove Wikipedia-specific words
    wikipedia_words = ['redirect', 'disambiguation', 'edit', 'external', 'link', 'main', 'article',
                       'navigation', 'search', 'talk', 'redirected', 'redirects', 'jpg', 'png', 'file',
                       'photo', 'image', 'svg', 'category', 'template', 'wikipedia', 'citation',
                       'needed', 'vte', 'portal', 'utc', 'isbn', 'sfn', 'reference', 'ref',
                       'url', 'date', 'archive', 'access', 'wikimedia', 'commons']
    text = ' '.join([word for word in text.split() if word not in wikipedia_words])
    text = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word.isalnum() and word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]

    return ' '.join(text)


tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")


def generate_summary(text, tokenizer=tokenizer, model=model, max_length=400):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


def get_asin(link):
    start_index = link.find("/dp/") + 4
    end_index = link.find("/", start_index)
    asin = link[start_index:end_index]
    return asin


def amazon_scraper(product):
    url = f"https://www.amazon.com/s?k={product}"
    driver = webdriver.Firefox()
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    results = soup.find_all('div', {'data-component-type': 's-search-result'}, limit=20)
    data = []
    counter = 0
    for result in results:
        skip = False
        name_element = result.find('span', {'class': 'a-size-medium a-color-base a-text-normal'})
        name = name_element.text if name_element else None
        link_element = result.find('a', {'class': 'a-link-normal s-underline-text s-underline-link-text s-link-style'})
        link = 'https://www.amazon.com' + link_element['href'].split('?')[0] if link_element else None
        if link is not None:
            driver.get(link)
            time.sleep(2)  # Wait for the page to load
            page_soup = BeautifulSoup(driver.page_source, 'html.parser')
            description_element = page_soup.find('div', {'id': 'productDescription'})
            if description_element:
                description = description_element.text.strip()
                counter += 1
            else:
                skip = True
                description = "No description available"
            try:
                image = result.find('img', {'class': 's-image'})['src']
            except TypeError:
                image = None

            if not skip:
                data.append({
                    'title': name,
                    'link': link,
                    'description': description,
                    'image': image,
                    'score': int(10)
                })

            if counter == 5:
                break

    driver.quit()
    return data


def wikipedia_scraper(product):
    url = f"https://en.wikipedia.org/wiki/{product}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.find('h1', {'class': 'firstHeading'}).text

    # extract main text
    paragraphs = soup.select('.mw-parser-output p')[:3]  # adjust as needed
    text = "\n".join([p.text.strip() for p in paragraphs])
    text = re.sub('\n', ' ', text)

    # extract image
    try:
        image = "https:" + soup.find('table', {'class': 'infobox'}).find('img')['src']
    except AttributeError:
        try:
            image = "https:" + soup.find('img')['src']
        except AttributeError:
            image = None

    text = preprocess_text_wiki(text)
    text = generate_summary(text, max_length=300)

    return {
        'title': title,
        'link': url,
        'image': image,
        'description': text,
    }


def get_similarity_scores(amazon, wiki, threshold=60):
    product_count = 0
    wiki_text_vector = get_text_vector(wiki['description']).reshape(1, -1)
    # t1 = np.array(vectorize_text_w2v(wiki['description'])).reshape(1, -1)
    for item in amazon:
        amazon_text_vector = get_text_vector(item["description"]).reshape(1, -1)
        # t2 = np.array(vectorize_text_w2v(item["description"])).reshape(1, -1)
        item['score'] = int(cosine_similarity(wiki_text_vector, amazon_text_vector)[0][0] * 100)
        # terminating condition
        if item['score'] >= threshold:
            product_count += 1
        if product_count == 20:
            break
        # s2 = int(cosine_similarity(t1, t2)[0][0] * 100)
        # print("SIMILARITY SCORES:", int(cosine_similarity(wiki_text_vector, amazon_text_vector)[0][0] * 100), s2)
    return amazon
