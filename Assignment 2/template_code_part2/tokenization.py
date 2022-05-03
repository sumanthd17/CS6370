from util import *
import string
# Add your import statements here
from nltk.tokenize import TreebankWordTokenizer
import re

class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		# return [sentence.split(' ') for sentence in text]
		output = []
		for sentence in text:
			sent = []
			sent = re.split('[, !/]', sentence)
			output.append(sent)
		return output


	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = [TreebankWordTokenizer().tokenize(sentence) for sentence in text]

		# Convert all tokens to lower case and remove tokens that are stand-alone punctuation marks
		tokenizedText = [[word.lower() for word in sentence if not word in string.punctuation] for sentence in tokenizedText]
		return tokenizedText