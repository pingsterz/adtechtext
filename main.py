import os
import time
from flask import Flask, jsonify, request
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import nltk
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
from nltk.stem import WordNetLemmatizer

def find(name, path):
    for root, dirs, files in os.walk(path):
        if root.endswith(name):
            return root

def find_nltk_data():
    start = time.time()
    path_to_nltk_data = find('nltk_data', '/')
    #with open('where_is_nltk_data.txt', 'w') as fout:
    #    fout.write(path_to_nltk_data)
    return path_to_nltk_data

def magically_find_nltk_data():
    if os.path.exists('where_is_nltk_data.txt'):
        with open('where_is_nltk_data.txt') as fin:
            path_to_nltk_data = '/nltk_data'
        if os.path.exists(path_to_nltk_data):
            nltk.data.path.append(path_to_nltk_data)
        else:
            nltk.data.path.append(find_nltk_data())
    else:
        path_to_nltk_data = find_nltk_data()
        nltk.data.path.append(path_to_nltk_data)

def explicit_app_engine(project):
    from google.auth import app_engine
    import googleapiclient.discovery
    # Explicitly use App Engine credentials. These credentials are
    # only available when running on App Engine Standard.
    credentials = app_engine.Credentials()

def text_extraction_gambling_metro(url):
    html = requests.get(url)
    soup = BeautifulSoup(html.content,'html.parser')
    headline = soup.find('h1',{'class':'post-title clear'})
    body_content = soup.find('div',{'class':'article-body'})
    body_content_all = body_content.findAll('p')
    body = ""
    for paragraph in range(len(body_content_all)):
        body = body + " " + body_content_all[paragraph].text
    return body.replace("\xa0"," ").strip()

def text_mining_word_proportion(s):
    wordnet_lemmatizer = WordNetLemmatizer()
    stopwords = set([word.strip() for word in open('stopwords.txt')])
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
    word_index_map = {}
    current_index = 0
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1
    token_proportion = np.zeros(len(word_index_map))
    for t in tokens:
        i = word_index_map[t]
        token_proportion[i] += 1
        token_proportion = token_proportion / token_proportion.sum()
    token_proportion_df = pd.Series(data=token_proportion, index=word_index_map).to_frame('token_proportion')
    token_proportion_df['token'] = token_proportion_df.index
    return token_proportion_df.reset_index(drop=True)

magically_find_nltk_data()
test_url = 'https://metro.co.uk/2018/06/03/boy-13-blew-80000-gambling-online-dads-credit-card-7600759/'
similarity_score = pd.read_csv('[3.2] gambling_tf_idf_similarity_score_max_df.csv', encoding='ISO-8859-1')
logistic_regression_coef = pd.read_csv('[5.2] gambling_tf_idf_df_logistic_regression_coef.csv', encoding='ISO-8859-1')
client = language.LanguageServiceClient()
test_output_all = pd.DataFrame()
text = text_extraction_gambling_metro(test_url)
word_proportion = text_mining_word_proportion(text)
token_df = word_proportion.merge(similarity_score, how='left', on='token')
token_df = token_df.merge(logistic_regression_coef, how='left', left_on='token', right_on='Unnamed: 0')
token_df = token_df.dropna()
similarity_index = max(token_df['aggregated'])
logit = sum(token_df['token_proportion']*token_df['Coefficient'])
classification_probability = np.exp(logit)/(1+np.exp(logit))
document = types.Document(content=text, type=enums.Document.Type.PLAIN_TEXT)
sentiment = client.analyze_sentiment(document=document).document_sentiment
sentiment_score = sentiment.score
sentiment_magnitude = sentiment.magnitude

app = Flask(__name__)
scores = [{'url': test_url},
{'Similarity_index': similarity_index},
{'Clssification_probablity': classification_probability},
{'Sentiment_score': sentiment_score},
{'Sentiment_maginitude': sentiment_magnitude}]

@app.route('/',methods=['GET', 'POST'])
def index():
    if (request.method == 'POST'):
        some_json = request.get_json()
        return jsonify({'you sent' : some_json}), 201
    else:
        return jsonify({"about":"Hello World!"})

@app.route('/multi/<int:num>',methods=['GET'])
def get_multiply10(num):
    return jsonify({'result': num*10})

@app.route('/test', methods=['GET'])
def returnScore():
    return jsonify({'scores': scores})

if __name__ == '__main__':
    app.run()
