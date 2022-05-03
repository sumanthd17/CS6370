from util import *
from collections import Counter
import math
import numpy as np
import time
import os
# Add your import statements here
import itertools
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

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


		index = {}  # dictionary that stores the inverted index
		# they keys are the terms, the values are lists of tuples. Each tuple contains a document ID followed by the term frequency

		doc_lengths = {}
		self.vocabulary = []

		# Each sub-list is a document.
		for i, doc in enumerate(docs):
			# Flatten the list of lists of words into one list of words
			ordered_words_of_doc = [word for sentence in doc for word in sentence]
			self.vocabulary = self.vocabulary + list(set(ordered_words_of_doc))
			# Get the frequency of each word in the document
			word_counts = Counter(ordered_words_of_doc)
			# Unique words in the document.
			unique_words = list(word_counts.keys())
			doc_lengths[docIDs[i]] = len(ordered_words_of_doc)

			for unique_word in unique_words:
				if unique_word in index:
					index[unique_word].append(
						(docIDs[i], word_counts[unique_word]))  # Store the document ID and term frequency in a tuple
				# Create a new key if the term does not exist in the dictionary
				else:
					index[unique_word] = [(docIDs[i], word_counts[unique_word])]

		self.vocabulary = list(set(self.vocabulary))
		if not os.path.exists("Datastore/vocab.txt"):
			with open("Datastore/vocab.txt", "w+") as f:
				for w in self.vocabulary:
					f.write(w + "\n")
			print("Saved vocabulary of preprocessed docs\n")

		sents = []
		for doc in docs:
			ordered_words_of_doc = [word for sentence in doc for word in sentence]
			sents.append(" ".join(ordered_words_of_doc))

		with open("Datastore/word_for_glove_training.txt", 'w') as f:
			f.writelines("\n".join(sents))

		self.aux_index = index
		self.V = len(list(index.keys()))
		self.N = len(docIDs)  # number of documents in the corpus
		self.docIDs = docIDs
		self.len_of_docs = doc_lengths
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

		# Fill in code here

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
		print("Ranking complete in {} seconds".format(end - start))

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
		print("Ranking complete in {} seconds".format(end - start))

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
		SVD_COMPRESSED_FILE_PATH = 'compressedFiles/svd.npz'
		if os.path.exists(SVD_COMPRESSED_FILE_PATH):
			print('Loading SVD...')
			savedSVD = np.load(SVD_COMPRESSED_FILE_PATH)
			U, S, Vt = savedSVD['U'], savedSVD['S'], savedSVD['Vt']
		else:
			print('Computing and saving SVD...')
			os.mkdir('compressedFiles')
			U, S, Vt = np.linalg.svd(self.index.toarray())
			np.savez_compressed(SVD_COMPRESSED_FILE_PATH, U=U, S=S, Vt=Vt)

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
		print("Ranking complete in {} seconds".format(end - start))

		return doc_IDs_ordered


	def wordnet(self, queries, k, preprocess):
		doc_IDs_ordered = []
		start_tfidf = time.time()
		# Calculating IDF for every term in the corpus
		words = sorted(list(self.aux_index.keys()))  # list of words

		# IDF score dictionary
		idf_score = {}
		for term in words:
			# Number of docs in which the term occurs
			n = len(self.aux_index[term])
			idf_score[term] = np.log10(self.N / n)

		tfidf_vectors = {}
		for docid in self.docIDs:
			if self.len_of_docs[docid] > 0:
				# Initialize each document vector as a vector of zeroes
				tfidf_vectors[docid] = np.zeros(self.V)

		wordnet_sim_arr = np.load("Datastore/wordnet_sim_arr.npy")

		for word_index, term in enumerate(words):
			for pair in self.aux_index[term]:
				docid = pair[0]
				freq = pair[1]
				# Calculating Tf*IDF
				tfidf_vectors[docid][word_index] += (freq / self.len_of_docs[docid]) * idf_score[term]
				tfidf_vectors[docid] += (freq / self.len_of_docs[docid]) * idf_score[term] * 0.03 * wordnet_sim_arr[
					word_index]

		end_tfidf = time.time()
		print("Time taken to build the TF-IDF matrix in LSA = {} seconds".format(end_tfidf - start_tfidf))

		######### LSA
		# We have a dictionary of tf-idf vectors, if we arrange them column-wise, we get the Term-Document matrix
		k = int(k)
		print("Number of latent dimensions used = {}".format(k))
		keys, vectors = zip(*tfidf_vectors.items())
		# Use the saved SVD results if they are available:
		if os.path.exists("Datastore/SVD_wordnet_weighted.npz") and not preprocess:
			start_svd = time.time()
			data_dict = np.load("Datastore/SVD_wordnet_weighted.npz")
			print("Found SVD_wordnet_weighted results stored in the folder.")
			U = data_dict["arr_0"]
			S = data_dict["arr_1"]
			Vt = data_dict["arr_2"]
			end_svd = time.time()
			print("Time taken to load SVD results = {} seconds".format(end_svd - start_svd))
		else:
			print("Calculating SVD at run-time.")
			all_vectors_matrix = np.array(vectors).T
			# Singular value decomposition
			start_svd = time.time()
			U, S, Vt = np.linalg.svd(all_vectors_matrix)
			end_svd = time.time()
			print("Time taken to do SVD = {} seconds".format(end_svd - start_svd))
			np.savez_compressed("Datastore/SVD_wordnet_weighted.npz", U, S, Vt)  # Save the results

		Uk = U[:, :k]
		# np.save("Datastore/Singular_values.npy", S)
		Sk = np.diag(S[:k])
		# Dictionary of document vectors in the latent space (k dimensions)
		lsa_doc_vectors = dict(zip(keys, Vt[:k, :].T))

		# Each sub-list is a query.
		for query in queries:
			# Flatten the list of lists of words into one list of words
			flat_query = [word for sentence in query for word in sentence]
			# Get the frequency of each word in the query
			word_counts = Counter(flat_query)
			# Unique words in the query.
			# unique_words = list(word_counts.keys())
			unique_words = [word for word in word_counts.keys() if word in words]
			# Find the length of the query
			query_length = len(flat_query)
			# Initialize the query vector as a vector of zeros
			query_vector = np.zeros(self.V)
			for unique_word in unique_words:
				word_index = words.index(unique_word)
				# TF-IDF scores for the query words
				query_vector[word_index] += (word_counts[unique_word] / query_length) * idf_score[unique_word]
				query_vector += (word_counts[unique_word] / query_length) * idf_score[unique_word] * 0.03 * \
								wordnet_sim_arr[word_index]

			# Approach 1
			ordered_docs = []
			# Map the query vector to the latent space
			query_vector = (query_vector.reshape(1, self.V) @ Uk) @ np.linalg.inv(Sk)

			for key in lsa_doc_vectors:
				# Calculate cosine similarity between each document and the query
				sim = np.sum(np.multiply(lsa_doc_vectors[key], query_vector)) / (
						np.linalg.norm(lsa_doc_vectors[key]) * np.linalg.norm(query_vector))
				ordered_docs.append((sim, key))

			# Sorting the documents based on the cosine_similarity
			ordered_docs.sort(reverse=True)
			doc_IDs_ordered.append([pair[1] for pair in ordered_docs])

		return doc_IDs_ordered


	def glove(self, queries):
		"""
		Rank the documents according to relevance for each query.
		TF-IDF vectors with query expansion using GloVe similarity is used.
		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		with open("Datastore/vocab_for_glove.txt") as f:
			vocab = f.readlines()

		doc_IDs_ordered = []
		start_tfidf = time.time()
		# Calculating IDF for every term in the corpus
		with open('Datastore/glove_aux_index.json', 'r') as fp:
			self.aux_index = json.load(fp)
		terms = sorted(list(self.aux_index.keys()))  # list of terms

		# IDF score dictionary
		idf_score = {}
		for term in terms:
			# Number of docs in which the term occurs
			n = len(self.aux_index[term])
			idf_score[term] = np.log10(self.N / n)

		tfidf_vectors = {}
		for docid in self.docIDs:
			if self.len_of_docs[docid] > 0:
				# Initialize each document vector as a vector of zeroes
				tfidf_vectors[docid] = np.zeros(len(vocab))

		glove_arr = np.load('Datastore/glove_arr.npy')
		for arr_pos in range(len(glove_arr)):
			glove_arr[arr_pos] = glove_arr[arr_pos] / np.linalg.norm(glove_arr[arr_pos])

		glove_sim_arr = np.zeros((glove_arr.shape[0], glove_arr.shape[0]))
		for i in range(len(glove_arr)):
			glove_sim_arr[i] = np.dot(glove_arr, glove_arr[i])

		for term_index, term in enumerate(terms):
			for pair in self.aux_index[term]:
				docid = pair[0]
				freq = pair[1]
				# Calculating Tf*IDF
				vec = (freq / self.len_of_docs[docid]) * idf_score[term] * 0.01 * glove_sim_arr[term_index]
				vec[vec < 0.5] = 0
				tfidf_vectors[docid] += vec
				tfidf_vectors[docid][term_index] += (freq / self.len_of_docs[docid]) * idf_score[term]
		end_tfidf = time.time()
		print("Time taken to build the TF-IDF matrix = {} seconds".format(end_tfidf - start_tfidf))

		# Each sub-list is a query.
		for query in queries:
			# Flatten the list of lists of words into one list of words
			flat_query = [word for sentence in query for word in sentence]
			# Get the frequency of each word in the query
			word_counts = Counter(flat_query)
			# Unique words in the query.
			# unique_words = list(word_counts.keys())
			unique_words = [word for word in word_counts.keys() if word in terms]
			# Find the length of the query
			query_length = len(flat_query)
			# Initialize the query vector as a vector of zeros
			query_vector = np.zeros(len(vocab))
			for unique_word in unique_words:
				term_index = terms.index(unique_word)
				# TF-IDF scores for the query terms with query expansion using GloVe similarity
				vec = (word_counts[unique_word] / query_length) * idf_score[unique_word] * 0.01 * glove_sim_arr[
					term_index]
				vec[vec < 0.5] = 0
				query_vector += vec
				query_vector[term_index] += (word_counts[unique_word] / query_length) * idf_score[unique_word]

			# Approach 1
			ordered_docs = []

			for key in tfidf_vectors:
				# Calculate cosine similarity between each document and the query
				sim = np.sum(np.multiply(tfidf_vectors[key], query_vector)) / (
						np.linalg.norm(tfidf_vectors[key]) * np.linalg.norm(query_vector))
				ordered_docs.append((sim, key))
			ordered_docs.sort(reverse=True)
			doc_IDs_ordered.append([pair[1] for pair in ordered_docs])

		return doc_IDs_ordered