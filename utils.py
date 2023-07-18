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
from vectorizer import get_text_vector
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# def preprocess_text(text):
#     # Convert to lowercase
#     text = text.lower()
#
#     # Remove special characters and numbers
#     text = re.sub(r"[^a-zA-Z]", " ", text)
#
#     # Tokenize the text
#     tokens = word_tokenize(text)
#
#     # Remove stopwords
#     stop_words = set(stopwords.words("english"))
#     tokens = [token for token in tokens if token not in stop_words]
#
#     # Lemmatize the tokens
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(token) for token in tokens]
#
#     # Join the tokens back into a single string
#     processed_text = " ".join(tokens)
#
#     return processed_text

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    wiki_terms = ["wikipedia", "wiki", "encyclopedia"]

    # Remove special characters and numbers
    # text = re.sub(r"[^a-zA-Z]", " ", text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    # stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in wiki_terms]

    # Lemmatize the tokens
    # lemmatizer = WordNetLemmatizer()
    # tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join the tokens back into a single string
    processed_text = " ".join(tokens)

    return processed_text


def generate_summary(text, max_length=100):
    # Load pre-trained T5 model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    # Tokenize the input text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the summary
    summary_ids = model.generate(inputs, max_length=max_length, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


def get_asin(link):
    # Extract the ASIN (Amazon Standard Identification Number) from the product link
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
        print(name)
        link_element = result.find('a', {'class': 'a-link-normal s-underline-text s-underline-link-text s-link-style'})
        print(link_element)
        link = 'https://www.amazon.com' + link_element['href'].split('?')[0] if link_element else None
        print(link)
        # Navigate to the product details page
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
    text = "\n".join([elem.text.strip() for elem in soup.select_one("#bodyContent")])
    text = re.sub('\n', ' ', text)
    try:
        image = "https:" + soup.find('table', {'class': 'infobox'}).find('img')['src']
    except AttributeError:
        image = None
    text = preprocess_text(text)
    text_summary = generate_summary(text, max_length=300)
    return {
        'title': title,
        'link': url,
        'image': image,
        'description': text_summary,
    }


def get_similarity_scores(amazon, wiki):
    wiki_text_vector = get_text_vector(wiki['description']).reshape(1, -1)
    for item in amazon:
        amazon_text_vector = get_text_vector(item["description"]).reshape(1, -1)
        item['score'] = int(cosine_similarity(wiki_text_vector, amazon_text_vector)[0][0] * 100)
    return amazon

