# Code for generating Wu-Palmer WordNet similarity scores
# The similarity scores have been pre-computed in the Large_files/wordnet_sim_arr.npy file
# Hence, this code need not be run. Only for verification purpose

from nltk.corpus import wordnet
import os, json
import numpy as np

# Get all vocabulary words

        
with open("./Datastore/vocab.txt",'r') as f:
    vocab = [word.strip() for word in f.readlines()]

wordnet_sim_arr = np.zeros((len(vocab), len(vocab)))
for i in range(len(vocab)):
    syn_i = wordnet.synsets(vocab[i]) # Obtain synsets of each term
    if len(syn_i)==0:
        wordnet_sim_arr[i][i]=1
        continue
    for j in range(len(vocab)):
        syn_j = wordnet.synsets(vocab[j])
        if len(syn_j)==0:
            continue
        wordnet_sim_arr[i][j] = syn_i[0].wup_similarity(syn_j[0]) # Find similarity

# Store similarity scores in an array
np.save("./Datastore/wordnet_sim_arr.npy", wordnet_sim_arr)