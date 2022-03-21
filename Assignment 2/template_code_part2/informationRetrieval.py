from util import *
from collections import Counter
import math
import numpy as np
import time

# Add your import statements here




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
		idx = {}
		for i, doc in enumerate(docs, 1):
			# docID = docIDs[docs.index(doc)]
			docID = i
			terms = [term for sentence in doc for term in sentence]
			for term, tf in list(Counter(terms).items()):
				try:
					idx[term].append([docID, tf])
				except:
					idx[term] = [[docID,tf]]

		self.index = idx
		self.docIDs = docIDs
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

		idf = np.zeros(len(self.index.keys()))
		n_docs = len(self.docIDs)
		for i, term in enumerate(sorted(self.index.keys())):
			idf[i] = math.log10(n_docs/len(self.index[term]))

		tf_idf = {}
		for id_ in self.docIDs:
			tf_idf[id_] = np.zeros(len(self.index.keys()))

		for i, term in enumerate(sorted(self.index.keys())):
			for doc_id, freq in self.index[term]:
				tf_idf[doc_id][i] = idf[i]*freq

		start = time.time()
		for q in queries:
			q_vector = np.zeros(len(self.index.keys()))
			terms = [term for sent in q for term in sent]
			term_counts = list(set(list(Counter(terms).items())))
			for term, freq in term_counts:
				# there are new terms in the query
				if term in self.index:
					idx = sorted(self.index.keys()).index(term)
					q_vector[idx] = idf[idx]*freq

			start = time.time()

			tf_idf_array = np.array(list(tf_idf.values()))
			cos_sim = np.sum(tf_idf_array * q_vector, axis=1) / (np.linalg.norm(tf_idf_array, axis=1) * np.linalg.norm(q_vector))
			# Replace nan with 0
			cos_sim[np.isnan(cos_sim)] = 0
			doc_IDs_ordered.append(np.argsort(cos_sim)[::-1])
		end = time.time()
		print("Ranking complete in {} seconds".format(end-start))
	
		return doc_IDs_ordered