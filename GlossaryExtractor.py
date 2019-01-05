from operator import itemgetter

from nltk import wordpunct_tokenize
from nltk import PorterStemmer
from nltk.corpus import stopwords
from gensim import corpora, models, similarities
from os import listdir


def preprocess_document(doc):
    stopset = set(stopwords.words('spanish'))
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


def launch_query(corpus):
    tfidf, dictionary = create_tf_idf_model(corpus)
    loaded_corpus = corpora.MmCorpus('vsm_docs.mm')
    index = similarities.MatrixSimilarity(loaded_corpus, num_features=len(dictionary))
    for i in dictionary:
        pq = preprocess_document(dictionary[i])
        vq = dictionary.doc2bow(pq)
        qtfidf = tfidf[vq]
        sim = index[qtfidf]
        ranking = sorted(enumerate(sim), key=itemgetter(1), reverse=True)
        total_score = 0
        for doc, score in ranking:
            rounded_score = round(score, 3)
            if rounded_score > 0:
                total_score += rounded_score
                # print("[ Score = " + "%.3f" % round(score, 3) + "] " + dictionary[i])
        mean = total_score / len(files)
        if mean > 0.01:
            print(dictionary[i] + ': ' + str(total_score) + ' : ' + str(mean))
        # print()


files = listdir('News/Pollution')
files_text = []
print(len(files))
for file in files:
    file_object = open('News/Pollution/' + file, 'r')
    file_text = file_object.read()
    files_text.append(file_text)
    # print(file_text)

launch_query(files_text)
