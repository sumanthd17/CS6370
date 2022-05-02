import itertools
import time
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import torch
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer 

class BERTIndex():

    def __init__(self):
        self.index = None
        self.docIDs = None
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def buildIndex(self, docs, docIDs):
        self.docIDs = docIDs
        sent_docs = [' '.join(x) for x in docs]
        
        index = []

        if not os.path.exists('index.pt'):
            with torch.no_grad():
                start = time.time()
                for text in tqdm(sent_docs):
                    # encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
                    # output = self.model(**encoded_input)
                    # index.append(output['pooler_output'].numpy().reshape(768))
                    embedding = self.model.encode(text)
                    index.append(embedding)
                end = time.time()
            print(f'Time taken to build Index: {end - start} seconds')
            print(f'Length of Index: {len(index)}')
            torch.save(index, 'index.pt')
            print(f'Saved successfully')
        else:
            index = torch.load('index.pt')
            print(f'Length of Index: {len(index)}')
            print(f'Loaded successfully')
        self.index = index

    def rank(self, queries):

        sent_queries = [' '.join(x) for x in queries]
        print(len(queries))

        doc_IDs_ordered = []
        query_embs = []

        if not os.path.exists('queries.pt'):
            with torch.no_grad():
                start = time.time()
                for text in tqdm(sent_queries):
                    # encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
                    # output = self.model(**encoded_input)
                    # query_embs.append(output['pooler_output'].numpy().reshape(768))
                    embedding = self.model.encode(text)
                    query_embs.append(embedding)
                end = time.time()

            print(f'Time taken to build queries: {end - start} seconds')
            print(f'Length of queries: {len(query_embs)}')
            torch.save(query_embs, 'queries.pt')
            print(f'Saved successfully')
        else:
            query_embs = torch.load('queries.pt')
            print(f'Length of Index: {len(query_embs)}')
            print(f'Loaded successfully')

        start = time.time()
        # cosine_similarity(query_embs[0].detach().numpy(), self.index[0].detach().numpy())
        cos_sim = cosine_similarity(query_embs, self.index)
        for cos_similarity_vector in cos_sim:
            top_n_doc_indexes = cos_similarity_vector.argsort()[::-1]
            # convert doc_indexes to docIDs
            top_n_docs = [self.docIDs[doc_index] for doc_index in top_n_doc_indexes]
            doc_IDs_ordered.append(top_n_docs)

        end = time.time()
        print("Ranking complete in {} seconds".format(end-start))

        return doc_IDs_ordered