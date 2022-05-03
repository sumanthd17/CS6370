import itertools
from Levenshtein import distance
import string
import re
import time
from collections import Counter

class BigramSpellCheck:
    def __init__(self, docs):
        
        # A list of all the unique words in the corpus (vocabulory)
        words = []
        for doc in docs:
            words.extend(re.sub("[^a-z ]+", " ", doc).split())
        self.vocabulary = list(set(words))

        # Generate all possible bigrams from aa to zz
        self.bigrams = ["".join(tup) for tup in itertools.product(string.ascii_lowercase, repeat=2)]

        # Create the bigram reverse index
        self.bigram_reverse_index = {}
        for bigram in self.bigrams:
            self.bigram_reverse_index[bigram] = [word for word in self.vocabulary if bigram in word]

    def correct_words_in_query(self, query):
        """
        Returns the corrected query.
        """
        # Updated query with correct spellings
        updated_query = ""
        for word in query.split():
            if word not in self.vocabulary:
                updated_query += " " + self.correct_word(word)
            else:
                updated_query += " " + word

        return updated_query.strip()

    def correct_word(self, query_word):
        """
        Returns the correct word which has the lowest euclidean distance
        """
        # query word is in vocabulary with correct spelling
        if query_word in self.vocabulary:
            return query_word

        # get all the bigrams for the input query word
        bigrams_of_query = [query_word[i:i+2] for i in range(len(query_word)-1)]

        # Get all possible candidates for the query word
        candidates = []
        for qb in bigrams_of_query:
            candidates.extend(self.bigram_reverse_index.get(qb, []))
        candidates = list(set(candidates))

        # Initially try to get bigrams which have atleast 50% of common bigrams
        # If not found add bigrams that have atleast 10% of common bigrams
        good_candidates = []
        thresholds = [0.5, 0.1]
        for threshold in thresholds:
            for candidate in candidates:
                bigrams_of_candidate = [candidate[i:i+2] for i in range(len(candidate)-1)]
                common_bigrams = [bigram for bigram in bigrams_of_query if bigram in bigrams_of_candidate]
                if len(common_bigrams)/len(bigrams_of_query) >= threshold:
                    good_candidates.append(candidate)
            if len(good_candidates) != 0:
                break
                
        # return query word if no correct word found
        if len(good_candidates)==0:
            return query_word

        # Compute edit distances to all the candidates and pick the nearest word
        edit_distances = [(distance(query_word, candidate), candidate) for candidate in good_candidates]
        edit_distances.sort()
        return edit_distances[0][1]


class OneEditSpellCheck:
    def __init__(self, docs):
    
        # A list of all the unique words in the corpus (vocabulory)
        words = []
        for doc in docs:
            words.extend(re.sub("[^a-z ]+", " ", doc).split())

        self.total_no_of_words = len(words)
        self.word_count_dict = Counter(words)
        self.vocabulary = list(set(words))

    def one_edit(self, word):
        """
        return all words at an edit distance 1. 
        """
        letters = string.printable[10:36]
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes    = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts    = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def probability_of_word(self, word):
        """
        probability of the word in the documents
        """
        return self.word_count_dict[word]/self.total_no_of_words

    def correct_word(self, word):
        """
        Returns the corrected word if exists with maximum probability
        """
        # Generates the candidates at edit distance 1 and exist in docs
        words_one_edit_away = self.one_edit(word)
        candidates = [w for w in words_one_edit_away if w in self.vocabulary]

        probability_of_words = [(self.probability_of_word(w), w) for w in candidates]
        if probability_of_words:
            _, most_probable_word = max(probability_of_words)
            return most_probable_word

        return word

    def correct_words_in_query(self, query):
        """
        Returns the corrected query.
        """
        # Updated query with correct spellings
        updated_query = ""
        for word in query.split():
            if word not in self.vocabulary:
                updated_query += " " + self.correct_word(word)
            else:
                updated_query += " " + word

        return updated_query.strip()
