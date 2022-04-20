from util import *
from collections import Counter
import math
import numpy as np
import time

# Add your import statements here
import itertools
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Imports for query expansion using wordnet 
import nltk
from nltk.corpus import wordnet

class InformationRetrieval():

	def __init__(self):
		self.index = None
		self.docIDs = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""
		start = time.time()

		self.docIDs = docIDs
		self.count_vectorizer = CountVectorizer(min_df=1)
		self.tfidf_transformer = TfidfTransformer(use_idf=True, smooth_idf=True)

		self.term_doc_freq = self.count_vectorizer.fit_transform([' '.join(list(itertools.chain.from_iterable(x))) for x in docs])
		self.index = self.term_doc_freq.T
		end = time.time()
		print("Index built in {} seconds".format(end-start))

	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []

		#Fill in code here

		self.tfidf_transformer.fit(self.term_doc_freq)
		self.tfidf = self.tfidf_transformer.transform(self.term_doc_freq)

		start = time.time()

		query_sents = [' '.join(list(itertools.chain.from_iterable(x))) for x in queries]
		query_counts = self.count_vectorizer.transform(query_sents)
		query_tfidf = self.tfidf_transformer.transform(query_counts)

		cos_sim = cosine_similarity(query_tfidf, self.tfidf)

		for cos_similarity_vector in cos_sim:
			top_n_doc_indexes = cos_similarity_vector.argsort()[::-1]
			# convert doc_indexes to docIDs
			top_n_docs = [self.docIDs[doc_index] for doc_index in top_n_doc_indexes]
			doc_IDs_ordered.append(top_n_docs)

		end = time.time()
		print("Ranking complete in {} seconds".format(end-start))
	
		return doc_IDs_ordered

	def rank_by_query_expansion(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
			by doing query expansion.
		"""

		doc_IDs_ordered = []

		nltk.download('wordnet')

		self.tfidf_transformer.fit(self.term_doc_freq)
		self.tfidf = self.tfidf_transformer.transform(self.term_doc_freq)
		start = time.time()

		query_sents = [' '.join(list(itertools.chain.from_iterable(x))) for x in queries]

		updated_query_sents = []

		# For each query using wordnet expand the query and check if is in document vector
		for query in query_sents:
			updated_query = ''
			for text in query.split():
				# Ignore words of length 1
				if len(text) <= 1:
					updated_query += text
					continue
				
				# Capture the synonyms of the words using wordnet
				synonyms = []
				length_synsets = len(wordnet.synsets(text))
				if length_synsets > 0:
					# Consider the most frequently used synset words only
					syn = wordnet.synsets(text)[0]
					synonyms = [l.name() for l in syn.lemmas()]
					synonyms = list(filter(lambda x: x in self.count_vectorizer.get_feature_names(), synonyms))

				synonyms.insert(0, text)
				updated_query += ' '.join(set(synonyms)) + ' '
			updated_query_sents.append(updated_query)
			
		query_sents = updated_query_sents
		query_counts = self.count_vectorizer.transform(query_sents)
		query_tfidf = self.tfidf_transformer.transform(query_counts)

		cos_sim = cosine_similarity(query_tfidf, self.tfidf)

		for cos_similarity_vector in cos_sim:
			top_n_doc_indexes = cos_similarity_vector.argsort()[::-1]
			# convert doc_indexes to docIDs
			top_n_docs = [self.docIDs[doc_index] for doc_index in top_n_doc_indexes]
			doc_IDs_ordered.append(top_n_docs)

		end = time.time()
		print("Ranking complete in {} seconds".format(end-start))
	
		return doc_IDs_ordered

	def rank_by_lsa(self, queries, k):
		"""
		Rank the documents according to relevance for each query using LSA

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		
		arg2 : number
			k important features of LSA

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []

		start = time.time()
		U, S, Vt = np.linalg.svd(self.index.toarray())
		end = time.time()
		print("Time to do SVD computation in {} seconds".format(end - start))

		# Consider the top k features
		Uk = U[:, :k]
		Sk = np.diag(S[:k])
		Vk = Vt[:k, :].T

		self.tfidf_transformer.fit(self.term_doc_freq)
		self.tfidf = self.tfidf_transformer.transform(self.term_doc_freq)

		start = time.time()

		query_sents = [' '.join(list(itertools.chain.from_iterable(x))) for x in queries]
		query_counts = self.count_vectorizer.transform(query_sents)
		query_tfidf = self.tfidf_transformer.transform(query_counts)

		# get the query in latent space 
		query_tfidf = (query_tfidf @ Uk) @ np.linalg.inv(Sk)

		# compute cosine similarity
		cos_sim = cosine_similarity(query_tfidf, Vk)

		for cos_similarity_vector in cos_sim:
			top_n_doc_indexes = cos_similarity_vector.argsort()[::-1]
			# convert doc_indexes to docIDs
			top_n_docs = [self.docIDs[doc_index] for doc_index in top_n_doc_indexes]
			doc_IDs_ordered.append(top_n_docs)

		end = time.time()
		print("Ranking complete in {} seconds".format(end-start))
	
		return doc_IDs_ordered