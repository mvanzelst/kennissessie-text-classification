import common

# Settings
# -1 = Unlimited
max_documents = 1000
# Choose one or more from: ['word_count', 'doc2vec']
feature_extraction_algorithms = ['word_count', 'doc2vec']
# Choose one or more from: ['reuters', 'imdb', 'newsgroups']
data_sets = ['reuters', 'imdb', 'newsgroups']

for data_set in data_sets:

    if data_set == 'reuters':
        documents = common.retrieve_reuters_documents(max_documents=max_documents)
        print('Loaded reuters documents')
    elif data_set == 'imdb':
        documents = common.retrieve_imdb_movie_reviews(max_documents=max_documents)
        print('Loaded imdb reviews')
    elif data_set == 'newsgroups':
        documents = common.retrieve_newsgroup_articles(max_documents=max_documents)
        print('Loaded newsgroup articles')
    else:
        documents = []

    for feature_extraction_algorithm in feature_extraction_algorithms:
        print('using {} algorithm on data set: {}'.format(feature_extraction_algorithm, data_set))

        if 'doc2vec' == feature_extraction_algorithm:
            doc2vec = common.create_or_load_doc2vec_model('model/{}-doc2vec-{}.bin'.format(data_set, len(documents)), documents)
            common.add_feature_vectors_doc2vec(documents, doc2vec)

        if 'word_count' == feature_extraction_algorithm:
            word_count_vectorizer = common.create_word_count_vectorizer(documents)
            common.add_feature_vectors_text_vectorizer(documents, word_count_vectorizer)

        # Visualize
        png_file_name = 'fig/{}-tsne-{}-{}.png'.format(data_set, feature_extraction_algorithm, len(documents))
        common.visualize(documents, png_file_name)

        # Classify
        print('classify data_set: {}, feature_extraction_algorithm: {}, num_documents: {}'.format(data_set, feature_extraction_algorithm, len(documents)))
        common.classify(documents)
        print()
