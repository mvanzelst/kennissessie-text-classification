# Build docker image 

_(from the project root dir)_

`docker build -t kennissessie-text-classification docker/`

# Run script

_(from the project root dir)_

`docker run -it --rm --name kennissessie-text-classification -v "$PWD":/code kennissessie-text-classification python3 script.py`

# Assignments

##### 1 Add the tf idf vectorizer as a feature extraction algorithm

see: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

hint: The TF IDF vectorizer is very similar to the word count vectorizer

##### 2 Doc2Vec document similarity 

The doc2vec library exposes the top10 similar results based on file id. 
see: https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.Doc2VecKeyedVectors.most_similar

In any of the data sets pick a random document and find the most similar documents using doc2vec. If you read these texts do you agree?  

Hint: you can access the similar docvecs for a document using:
~~~~
document_id = 'training/10335' # (stored in document['file_id'])
document_vector = doc2vec.docvecs[document_id]
print(doc2vec.docvecs.most_similar(positive=[document_vector]))
~~~~

##### 3 Try out another classification algorithm

Sklearn has a lots of of-the-shelve classification algorithm next to one that is currently used in the "classify" function in common.py

Most of these classification algorithms have lots of parameters, you can try to tweak them and see if you get better results.

For an overview see: http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

##### 4 Bonus! Find another labeled text data set and try to add it, visualize it and classify it  