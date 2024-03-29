import itertools
import gensim.downloader
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
from gensim.models import Word2Vec


class Word2VecIndex():
    def __init__(self, train):
        self.index = None
        self.docIDs = None
        self.train = train
        # self.model = gensim.downloader.load('word2vec-google-news-300')
        self.model = None
        
    def buildIndex(self, docs, docIDs):
        self.docIDs = docIDs
        index = []

        docs_ = list(itertools.chain.from_iterable(docs))
        if self.train:
            self.model = Word2Vec(docs_, min_count=1)
            vocab = self.model.wv.index_to_key
        else:
            self.model = gensim.downloader.load('word2vec-google-news-300')
            vocab = self.model.index_to_key

        if not os.path.exists('word2vec.npy') or self.train:
            for doc in tqdm(docs):
                count = 0
                doc_emb = 0
                if len(doc) == 0:
                    continue
                for sent in doc:
                    for word in sent:
                        if self.train:
                            if word != '.' and word in vocab:
                                doc_emb += self.model.wv.get_vector(word)
                                count += 1
                        else:
                            if word != '.' and word in vocab:
                                doc_emb += self.model[word]
                                count += 1
                doc_emb /= count
                index.append(doc_emb)

            np.save('word2vec.npy', index)
        else:
            index = np.load('word2vec.npy')

        print(index[0].shape)

        self.index = index

    def rank(self, queries):
        doc_IDs_ordered = []

        query_embs = []
        queries_ = []

        for i in queries:
            q = list(itertools.chain.from_iterable(i))
            queries_.append(q)

        if self.train:
            vocab = self.model.wv.index_to_key
        else:
            vocab = self.model.index_to_key

        for q in tqdm(queries_):
            count = 0
            q_emb = 0
            # for sent in q:
            for word in q:
                if self.train:
                    if word != '.' and word in vocab:
                        q_emb += self.model.wv.get_vector(word)
                        count += 1
                else:
                    if word != '.' and word in vocab:
                        q_emb += self.model[word]
                        count += 1
            q_emb /= count
            query_embs.append(q_emb)

        print(query_embs[0].shape)
        cos_sim = cosine_similarity(query_embs, self.index)
        for cos_similarity_vector in cos_sim:
            top_n_doc_indexes = cos_similarity_vector.argsort()[::-1]
            # convert doc_indexes to docIDs
            top_n_docs = [self.docIDs[doc_index] for doc_index in top_n_doc_indexes]
            doc_IDs_ordered.append(top_n_docs)

        return doc_IDs_ordered