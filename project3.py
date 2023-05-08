# -*- coding: utf-8 -*-
#from text_normalizer import *

import sys

import pypdf
import os
import io
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import networkx

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pickle

path = os.path.abspath('./smartcity')
common_words = ['Smart City', 'City', 'city', 'page', 'Page', 'challenge', 'challenges', "The City's", 'The City']
output_filename = 'smartcity_predict.tsv'
stop_words = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    #change pos_tag to adapt WordNetLemmatizer
    pos = nltk.pos_tag([word])[0][1][0].upper()
    if pos == 'J':
        return wordnet.ADJ
    elif pos == 'V':
        return wordnet.VERB
    elif pos == 'N':
        return wordnet.NOUN
    elif pos == 'R':
        return wordnet.ADV
    else:
        return wordnet.NOUN

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = nltk.word_tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)

def get_raw_data(filename):
    city = filename[:len(filename) - 4]
    file_path = os.path.join(path, filename)
    raw_data = ''
    with open(file_path, 'rb') as data:
        pdf_reader = pypdf.PdfReader(data)
        for page in pdf_reader.pages:
            raw_data += page.extract_text()
    
    return city, raw_data

def data_clean(raw_data):
    for word in common_words:
        clean_data = raw_data.replace(word, 'M')
    #clean_data = re.sub(r'\n|\r', ' ', clean_data)
    clean_data = re.sub(r' +', ' ', clean_data)
    clean_data = clean_data.strip()
    clean_data = normalize_document(clean_data)
    return clean_data

def summarize(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    sentences = nltk.sent_tokenize(text)
    num_sentences = 1
    num_topics = 1
    
    norm_sentences = normalize_corpus(sentences)
    
    tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
    dt_matrix = tv.fit_transform(norm_sentences)
    dt_matrix = dt_matrix.toarray()

    vocab = tv.get_feature_names()
    td_matrix = dt_matrix.T
    similarity_matrix = np.matmul(dt_matrix, dt_matrix.T)
    np.round(similarity_matrix, 3)
    similarity_graph = networkx.from_numpy_array(similarity_matrix)
    scores = networkx.pagerank(similarity_graph)
    ranked_sentences = sorted(((score, index) for index, score in scores.items()),reverse=True)

    words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for sentence in sentences for word in word_tokenize(sentence) if word not in stop_words]
    
    word_freq = Counter(words)
    keywords = word_freq.most_common(1)[0][0]
    top_sentence_indices = [ranked_sentences[index][1] 
                        for index in range(num_sentences)]
    top_sentence_indices.sort()
    summary = '\n'.join(np.array(sentences)[top_sentence_indices])
    return keywords, summary

def parse_config(argv):
    config = {'filename': '', 'summary': False}
    n = len(argv)
    i = 0
    while i < n:
        if argv[i] == '--document':
            config['filename']  = argv[i + 1]
            i += 2

        elif argv[i] == '--summarize':
            config['summary']  = True
            i += 1

        elif argv[i] == '--keywords':
            config['summary']  = True
            i += 1

        else:
            i += 1

    return config

if __name__ == '__main__':
    
    config = parse_config(sys.argv[1::])
    model = pickle.load(open('model.pkl', 'rb'))
    
    city, raw_data = get_raw_data(config['filename'])
    clean_data = data_clean(raw_data)
    keywords, summary = summarize(raw_data)

   
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = tfidf_vectorizer.fit_transform([clean_data])
    cluster_id = model.predict(X)[0]
    
    df = pd.DataFrame(columns=['city', 'raw text', 'clean text', 'clusterid', 'summary', 'keywords'])
    new_row = {'city': city, 'raw text' : raw_data, 'clean text' : clean_data, 'clusterid' : cluster_id, 'summary' : summary, 'keywords' : keywords}
    df.loc[len(df)] = new_row
    print(df)
    df.to_csv('smartcity_predict.tsv', sep='\t')
    
    

