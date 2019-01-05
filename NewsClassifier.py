#!/usr/bin/python
import numpy as np
import os
import random

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim import corpora, models, similarities
from operator import itemgetter

import Glossaries

CLASSIFIED = 'Classified'


def get_test_data():
    features = [element for element in Glossaries.basketball_glossary] \
               + [element for element in Glossaries.cinema_glossary] \
               + [element for element in Glossaries.pollution_glossary]
    labels = [0] * 15 + [1] * 15 + [2] * 15
    stemmed_count_vect = StemmedCountVectorizer()
    x_train_counts = stemmed_count_vect.fit_transform(features)
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

    return labels, stemmed_count_vect, tfidf_transformer, x_train_tfidf


def check_prediction_and_store(predicted_probability, predicted_category, news):
    if predicted_probability[0][predicted_category[0]] > 0.45:
        if predicted_category[0] == 0:
            prediction = 'Basketball'
        elif predicted_category[0] == 1:
            prediction = 'Cinema'
        else:
            prediction = 'Pollution'
        store_news(news, prediction)
    else:
        store_news(news, 'Mix')


def check_model_accuracy(clf, stemmed_count_vect, tfidf_transformer):
    # Check model accuracy with real articles
    files = os.listdir('News/Cinema')
    features = []
    print(len(files))
    for file in files:
        file_object = open('News/Cinema/' + file, 'r')
        features.append(file_object.read())
    x_test_counts = stemmed_count_vect.transform(features)
    x_test_tfidf = tfidf_transformer.transform(x_test_counts)
    predicted = clf.predict(x_test_tfidf)
    test_labels = np.ones(30, dtype=int)
    # print(type(predicted))
    # print(predicted)
    # print(type(test_labels))
    # print(test_labels)
    print('Accuracy of the model ' + str(np.mean(predicted == test_labels)))


def predict_news(clf, stemmed_count_vect, tfidf_transformer, news):
    test_corpus = [news]
    x_test_counts = stemmed_count_vect.transform(test_corpus)
    x_test_tfidf = tfidf_transformer.transform(x_test_counts)
    predicted_probability = clf.predict_proba(x_test_tfidf)
    predicted_category = clf.predict(x_test_tfidf)
    return predicted_category, predicted_probability


def store_news(news, directory):
    if not os.path.exists(CLASSIFIED):
        os.makedirs(CLASSIFIED)
    if not os.path.exists(CLASSIFIED + '/' + directory):
        os.makedirs(CLASSIFIED + '/' + directory)
    file = open(CLASSIFIED + '/' + directory + '/news' + str(random.randint(0, 1000000)) + '.txt', 'w+')
    file.write(news)
    file.close()
    print('This article is classified as ' + directory + '\n')


def naive_bayes_classifier(news):
    labels, stemmed_count_vect, tfidf_transformer, x_train_tfidf = get_test_data()
    clf = MultinomialNB().fit(x_train_tfidf, labels)

    check_model_accuracy(clf, stemmed_count_vect, tfidf_transformer)

    predicted_category, predicted_probability = predict_news(clf, stemmed_count_vect, tfidf_transformer, news)
    return predicted_probability, predicted_category


stemmer = SnowballStemmer(language="english")


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


def support_vector_machine(news):
    labels, stemmed_count_vect, tfidf_transformer, x_train_tfidf = get_test_data()

    clf = SGDClassifier(loss='log', penalty='l2', alpha=1e-3, max_iter=8, random_state=42, tol=0.19)
    _ = clf.fit(x_train_tfidf, labels)

    check_model_accuracy(clf, stemmed_count_vect, tfidf_transformer)

    predicted_category, predicted_probability = predict_news(clf, stemmed_count_vect, tfidf_transformer, news)
    return predicted_probability, predicted_category


def preprocess_document(doc):
    stopset = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = wordpunct_tokenize(doc)
    clean = [token.lower() for token in tokens if token.lower() not in stopset and len(token) > 2]
    final = [stemmer.stem(word) for word in clean]
    return final


# Creates the dictionary of words contained in the corpus given.
def create_dictionary(processed_docs):
    dictionary = corpora.Dictionary(processed_docs)
    dictionary.save('vsm.dict')
    return dictionary


# Document to bag of words.
# Creates the vectors of each text.
def docs2bows(processed_docs, dictionary):
    vectors = [dictionary.doc2bow(doc) for doc in processed_docs]
    corpora.MmCorpus.serialize('vsm_docs.mm', vectors)
    return vectors


# Creates the tfidf for each concept
def create_tf_idf_model(corpus):
    processed_docs = [preprocess_document(doc) for doc in corpus]
    dictionary = create_dictionary(processed_docs)
    docs2bows(processed_docs, dictionary)
    loaded_corpus = corpora.MmCorpus('vsm_docs.mm')
    tfidf = models.TfidfModel(loaded_corpus)
    return tfidf, dictionary


def launch_query(corpus, q):
    tfidf, dictionary = create_tf_idf_model(corpus)
    loaded_corpus = corpora.MmCorpus('vsm_docs.mm')
    index = similarities.MatrixSimilarity(loaded_corpus, num_features=len(dictionary))
    pq = preprocess_document(q)
    vq = dictionary.doc2bow(pq)
    qtfidf = tfidf[vq]
    sim = index[qtfidf]
    ranking = sorted(enumerate(sim), key=itemgetter(1), reverse=True)
    matched_words = 0
    for doc, score in ranking:
        if score > 0:
            matched_words += 1
    return matched_words


print('¡Bienvenido al clasificador de noticias!\n')
option = ''
while option != '1':
    print('Introduce el número de la utilidad:')
    print('[0] Clasificar noticias')
    print('[1] Salir')
    option = input()
    if option == '0':
        news_option = ''
        while news_option == '':
            news_option = input('Introduce la noticia que quieres que se clasifique:\n')
        print('Elige el método para clasificar la noticia')
        print('[0] Support Vector Machine')
        print('[1] Naïve Bayes')
        print('[2] Vector Space Model')
        classifier_option = input()
        if classifier_option == '0':
            print('Support Vector Machine classification')
            predicted_prob, predicted_cat = support_vector_machine(news_option)
            check_prediction_and_store(predicted_prob, predicted_cat, news_option)
        elif classifier_option == '1':
            print('Naïve Bayes classification')
            predicted_prob, predicted_cat = naive_bayes_classifier(news_option)
            check_prediction_and_store(predicted_prob, predicted_cat, news_option)
        elif classifier_option == '2':
            matches = [('Basketball', launch_query(Glossaries.basketball_glossary, news_option)),
                       ('Cinema', launch_query(Glossaries.cinema_glossary, news_option)),
                       ('Pollution', launch_query(Glossaries.pollution_glossary, news_option))]
            ranking = sorted(matches, key=itemgetter(1), reverse=True)
            print(ranking)
            if ranking[0][1] > 5:
                store_news(news_option, ranking[0][0])
            else:
                store_news(news_option, 'Mix')

    elif option == '1':
        print('¡Adiós! Gracias por utilizar la herramienta')
