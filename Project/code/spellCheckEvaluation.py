import pandas as pd
import numpy as np
import json
import string
import random
import re
from SpellCheck import BigramSpellCheck, OneEditSpellCheck

# Load the Cranfield dataset documents
docs_json = json.load(open("../cranfield/cran_docs.json", 'r'))[:]
doc_ids, docs = [item["id"] for item in docs_json], [item["body"] for item in docs_json]

# Form the vocabulary
words = []
for doc in docs:
    words = words + re.sub("[^a-z ]+", " ", doc).split()
vocabulary = list(set(words))

letters = string.ascii_lowercase
words_generated_using_corpus = []
correct_words = []

print("Computed the vocabulory...")

# Generate random words with some mistake
for word in vocabulary:
    length = len(word)
    # Ignore words of length less than 2
    if length < 2:
        continue

    index = random.randint(0, length - 1)

    # Add Random insertion, substitution, deletion and Transposition
    words_generated_using_corpus.append(word[:index] + random.choice(string.ascii_lowercase) + word[index:])
    words_generated_using_corpus.append(word[:index] + random.choice(string.ascii_lowercase) + word[index+1:])

    index = random.randint(0, length - 2)
    words_generated_using_corpus.append(word[:index]+word[index+1:])
    words_generated_using_corpus.append(word[:index] + word[index+1] + word[index] + word[index+2:])
    correct_words = correct_words + [word]*4

# Save words to dataframe
df = pd.DataFrame(data={"correct_word": correct_words, "generated_words": words_generated_using_corpus})
df.to_csv("Output/spellcheck_eval_list.csv", index= False)

print('Genearated words...')

# Evaluate the generated words
df = pd.read_csv("Output/spellcheck_eval_list.csv")
words_generated_using_corpus = list(df["generated_words"])
correct_words = list(df["correct_word"])
obj = BigramSpellCheck(docs)

words_corrected_to = []
for word in words_generated_using_corpus:
    try:
        words_corrected_to.append(obj.correct_word(word))
    except:
        words_corrected_to.append("-")

accuracy = np.mean(np.array(words_corrected_to)==np.array(correct_words))

incorrect_spelling_corrections = df.iloc[np.where(np.array(words_corrected_to) != np.array(correct_words))]
incorrect_spelling_corrections['model_pred'] = [words_corrected_to[i] for i in np.where(np.array(words_corrected_to)!=np.array(correct_words))[0].tolist()]
incorrect_spelling_corrections.to_csv("Output/incorrect_spellcheck_predictionss.csv")

print("\nAccuracy of spelling correction = {} %\n".format(accuracy * 100.0))
