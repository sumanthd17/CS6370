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

		start = time.time()
		idf = {}
		n_docs = len(self.docIDs)
		for term in self.index.keys():
			idf[term] = math.log10(n_docs/len(self.index[term]))
		end = time.time()
		print("IDF built in {} seconds".format(end-start))

		tf_idf = {}
		for id_ in self.docIDs:
			tf_idf[id_] = {t: 0 for t in self.index}

		start = time.time()
		for term in self.index:
			for doc_id, freq in self.index[term]:
				tf_idf[doc_id][term] = idf[term]*freq
		end = time.time()
		print("TF-IDF built in {} seconds".format(end-start))

		print(f'Length of queruies: {len(queries)}')
		start = time.time()
		for q in queries:
			q_vector = {t: 0 for t in self.index}
			terms = [term for sent in q for term in sent]
			term_counts = list(Counter(terms).items())
			for term, freq in term_counts:
				# there are new terms in the query
				if term in self.index:
					q_vector[term] = idf[term]*freq

			cos_sim = {}
			for doc_id in self.docIDs:
				cos_sim[doc_id] = 0
				cos_sim[doc_id] = sum(tf_idf[doc_id][term]*q_vector[term] for term in self.index.keys()) / np.linalg.norm(list(tf_idf[doc_id].values())) * np.linalg.norm(list(q_vector.values()))
			cos_sim_sorted = dict(sorted(cos_sim.items(), key=lambda x: x[1], reverse=True))
			doc_IDs_ordered.append(list(cos_sim_sorted.keys()))

		end = time.time()
		print("Ranking complete in {} seconds".format(end-start))
	
		return doc_IDs_ordered