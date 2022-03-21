from util import *
from collections import Counter
import math
import numpy as np
import time

# Add your import statements here
import itertools
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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