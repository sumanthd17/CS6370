from util import *

# Add your import statements here
from nltk.tokenize import sent_tokenize
import re

class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		# return [x.strip() for x in text.split('.') if len(x) >= 1]
		return [x.strip() for x in re.split('[?.\n!]', text) if len(x) >= 1]


	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""
		
		return sent_tokenize(text)