from collections import Counter
from random import shuffle

import matplotlib
from gensim.utils import tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler
from toolz import interleave

matplotlib.use('agg')
import matplotlib.pyplot as plt

from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk.corpus import reuters
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
import matplotlib.patches as mpatches

from os import listdir
from os.path import isfile, join

cachedStopWords = stopwords.words("english")
p = re.compile('[a-z]+')


def retrieve_reuters_documents(max_documents=-1, filter_words=True):
    # List of documents
    documents = []

    training_files = [file_id for file_id in reuters.fileids() if 'training/' in file_id]
    test_files = [file_id for file_id in reuters.fileids() if 'test/' in file_id]

    for file_id in interleave([training_files, test_files]):
        if max_documents > -1 and len(documents) >= max_documents:
            return documents

        words = list(reuters.words(fileids=file_id))
        words_filtered = do_filter_words(words)
        document = {
            'words': words,
            'title': reuters.raw(fileids=file_id).split("\n")[0],
            'categories': reuters.categories(fileids=file_id),
            'is_training_example': True if 'training/' in file_id else False,
            'is_test_example': True if 'test/' in file_id else False,
            'words_filtered': words_filtered if filter_words else words,
            'file_id': file_id
        }
        if len(words_filtered) < 30:
            continue
        documents.append(document)


    return documents

def retrieve_newsgroup_articles(max_documents=-1):
    # List of documents
    documents = []

    news_groups_all = fetch_20newsgroups(subset='all')

    train_news_groups = fetch_20newsgroups(subset='train')
    train_news_groups_zipped = zip(train_news_groups.target, train_news_groups.data, train_news_groups.filenames)
    test_news_groups = fetch_20newsgroups(subset='test')
    test_news_groups_zipped = zip(test_news_groups.target, test_news_groups.data, test_news_groups.filenames)


    news_groups = interleave([train_news_groups_zipped, test_news_groups_zipped])
    for news_group_id, article, file_name in news_groups:
        if max_documents > -1 and len(documents) >= max_documents:
            return documents

        words = tokenize(article)
        document = {
            'words': words,
            'categories': [news_groups_all.target_names[news_group_id]],
            'is_training_example': True if 'train/' in file_name else False,
            'is_test_example': True if 'test/' in file_name else False,
            'words_filtered': do_filter_words(words),
            'file_id': file_name
        }
        if len(document['words_filtered']) < 30:
            continue
        documents.append(document)


    return documents

def list_files(dir_name):
    return [join(dir_name, file_name) for file_name in listdir(dir_name) if isfile(join(dir_name, file_name))]

def retrieve_imdb_movie_reviews(max_documents=-1):
    # List of documents
    documents = []
    file_names = interleave([list_files('/data-sets/aclImdb/test/neg'), list_files('/data-sets/aclImdb/test/pos'),
                             list_files('/data-sets/aclImdb/train/neg'), list_files('/data-sets/aclImdb/train/pos'),
                             list_files('/data-sets/aclImdb/train/unsup')])

    for file_name in file_names:
        if max_documents > -1 and len(documents) >= max_documents:
            return documents

        text_file = open(file_name, "r", encoding='utf8')
        lines = text_file.readlines()
        text_file.close()
        words = tokenize("\n".join(lines))
        document = {
            'words': words,
            'categories': ['neg'] if '/neg/' in file_name else ['pos'] if '/pos/' in file_name else [],
            'is_training_example': True if '/train/' in file_name else False,
            'is_test_example': True if '/test/' in file_name else False,
            'words_filtered': do_filter_words(words),
            'file_id': file_name
        }
        if len(document['words_filtered']) < 30:
            continue
        documents.append(document)

    return documents


def do_filter_words(words, min_length=3):
    words = [word.lower() for word in words]
    return [PorterStemmer().stem(word) for word in words if len(word) >= min_length and word not in cachedStopWords and p.match(word)]


def train_doc2vec(documents, doc2vec, num_epochs=10):
    corpus = [TaggedDocument(document['words_filtered'], [document['file_id']]) for document in documents]
    doc2vec.build_vocab(corpus)
    doc2vec.train(corpus, total_examples=len(corpus), epochs=num_epochs)

def buildColorMap(categories):
    colorMap = {}
    cmap = plt.get_cmap('jet')
    uniqueCategories = list(set(categories))
    shuffle(uniqueCategories)
    counter = 0
    for category in uniqueCategories:
        colorMap[category] = cmap(1 / len(uniqueCategories) * counter)
        counter += 1
    return colorMap

def visualize(documents, png_location, most_common_categories=10, colors=True):
    # Documents with exactly one category
    documents = [document for document in documents if len(document['categories']) == 1]

    # Only largest categories
    counter = Counter([document['categories'][0] for document in documents])
    largest_categories = [t[0] for t in counter.most_common(most_common_categories)]
    documents = [document for document in documents if document['categories'][0] in largest_categories]

    X = [document['feature_vector'] for document in documents]

    # First reduce to 35 dimensions using PCA (lot quicker than using TSNE right away)
    X = PCA(n_components=35).fit_transform(X)

    # Reduce dimensions to 2 (to plot on a 2d graph)
    reduced = TSNE(n_components=2).fit_transform(X)

    colorMap = buildColorMap([document['categories'][0] for document in documents])

    # Plot
    plt.figure(1, figsize=(30, 20))
    for document, coord in zip(documents, reduced):
        if colors:
            category = document['categories'][0]
            plt.scatter(coord[0], coord[1], c=colorMap[category], alpha=0.4)
        else:
            plt.scatter(coord[0], coord[1], c="black", alpha=0.4)

    # Category color legend
    if colors:
        patches = [mpatches.Patch(color=colorMap[category], label=category) for category in colorMap]
        plt.legend(handles=patches)

    plt.savefig(png_location)
    plt.clf()
    print("png created {}".format(png_location))


def add_feature_vectors_doc2vec(documents, doc2vec):
    for document in documents:
        document['feature_vector'] = doc2vec.docvecs[document['file_id']]

def create_word_count_vectorizer(documents, max_features=250):
    countVectorizer = CountVectorizer(max_features=max_features)
    countVectorizer.fit([' '.join(document['words_filtered']) for document in documents])
    return countVectorizer

def add_feature_vectors_text_vectorizer(documents, textVectorizer):
    for document in documents:
        document_string = [' '.join(document['words_filtered'])]
        document['feature_vector'] = textVectorizer.transform(document_string).toarray().tolist()[0]

def create_or_load_doc2vec_model(model_location, documents):
    if not isfile(model_location):
        doc2vec = Doc2Vec(dm=0)
        train_doc2vec(documents, doc2vec)
        doc2vec.save(model_location)
        print("doc2vec model trained and saved: {}".format(model_location))
        return doc2vec
    else:
        print("doc2vec model loaded from disk: {}".format(model_location))
        return Doc2Vec.load(model_location)


def classify(documents, categories_to_classify=10):
    # Documents with exactly one category
    documents = [document for document in documents if len(document['categories']) == 1]

    # Only largest categories
    counter = Counter([document['categories'][0] for document in documents])
    most_common_categories = [t[0] for t in counter.most_common(categories_to_classify)]
    documents = [document for document in documents if document['categories'][0] in most_common_categories]

    encoder = LabelEncoder()
    encoder.fit([document['categories'][0] for document in documents])
    y_train = encoder.transform([document['categories'][0] for document in documents if document['is_training_example']])
    y_test = encoder.transform([document['categories'][0] for document in documents if document['is_test_example']])

    scaler = RobustScaler()
    scaler.fit([document['feature_vector'] for document in documents])
    x_train = scaler.transform([document['feature_vector'] for document in documents if document['is_training_example']])
    x_test = scaler.transform([document['feature_vector'] for document in documents if document['is_test_example']])

    classifier = MLPClassifier()
    classifier.fit(x_train, y_train)
    score = classifier.score(x_test, y_test)
    print(score)