from util import *

# Add your import statements here
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stop_words = set(stopwords.words('english'))
		return [[word for word in sentence if word.lower() not in stop_words] for sentence in text]