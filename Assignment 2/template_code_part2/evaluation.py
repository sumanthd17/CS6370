from util import get_relevant_docs

# Add your import statements here
from itertools import groupby
from operator import itemgetter
import numpy as np
from math import log2

class Evaluation():

    def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
                A list of integers denoting the IDs of documents in
                their predicted order of relevance to a query
        arg2 : int
                The ID of the query in question
        arg3 : list
                The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
                The k value

        Returns
        -------
        float
                The precision value as a number between 0 and 1
        """

        precision = 0

        # Fill in code here
        precision = len(list(set(query_doc_IDs_ordered[:k]).intersection(true_doc_IDs))) / k
        return precision

    def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean precision value as a number between 0 and 1
        """
        # Fill in code here
        precisions = []
        for idx in range(len(query_ids)):
            precisions.append(self.queryPrecision(doc_IDs_ordered[idx], query_ids[idx], get_relevant_docs(query_ids[idx], qrels), k))
        return np.mean(precisions)

    def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The recall value as a number between 0 and 1
        """
        # Fill in code here
        recall_at_k = len(list(set(query_doc_IDs_ordered[:k]).intersection(true_doc_IDs))) / len(true_doc_IDs)
        return recall_at_k

    def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean recall value as a number between 0 and 1
        """

        # Fill in code here
        recalls = []
        for idx, q in enumerate(query_ids):
            recalls.append(self.queryRecall(doc_IDs_ordered[idx], query_ids[idx], get_relevant_docs(query_ids[idx], qrels), k))

        return np.mean(recalls)

    def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The fscore value as a number between 0 and 1
        """
        # Fill in code here
        precison = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)

        if precison == 0 and recall == 0:
            return 0

        return (2 * precison * recall) / (precison + recall)

    def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean fscore value as a number between 0 and 1
        """
        # Fill in code here
        f1_scores = []
        for idx, q in enumerate(query_ids):
            f1_scores.append(self.queryFscore(doc_IDs_ordered[idx], query_ids[idx], get_relevant_docs(query_ids[idx], qrels), k))
        return np.mean(f1_scores)

    def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of nDCG of the Information Retrieval System
        at given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The nDCG value as a number between 0 and 1
        """

        rel_vals = {}
        rel_docs = []
        DCGk = 0
        IDCGk = 0

        # Capture (ground truth) relevance values for queries in question
        for true_doc in true_doc_IDs:
            if int(query_id) == int(true_doc["query_num"]):
                docID = int(true_doc["id"])
                rel_docs.append(docID)
                relevance = 5 - true_doc["position"]
                rel_vals[docID] = relevance

        # Compute DCGk for predicted order of relevance to a query
        for i in range(1, k+1):
            docID = int(query_doc_IDs_ordered[i-1])
            if docID in rel_docs:
                relevance = rel_vals[docID]
                DCGk += (2**relevance - 1) / log2(i+1)

        # Compute IDCGK for ideal order of relevance to a query
        ideal_order = sorted(rel_vals.values(), reverse=True)
        no_of_rel_docs = len(ideal_order)
        for i in range(1, min(no_of_rel_docs, k) + 1):
            relevance = ideal_order[i-1]
            IDCGk += (2**relevance - 1) / log2(i+1)

        nDCGk = DCGk/IDCGk
        return nDCGk

    def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of nDCG of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean nDCG value as a number between 0 and 1
        """

        nDCGs = []
        no_of_queries = len(query_ids)

        # Compute nDCG for each query
        for i in range(no_of_queries):
            query_doc_IDs_ordered = doc_IDs_ordered[i]
            query_id = int(query_ids[i])
            nDCG = self.queryNDCG(query_doc_IDs_ordered, query_id, qrels, k)
            nDCGs.append(nDCG)
        
        return sum(nDCGs)/len(nDCGs)
        
    def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of average precision of the Information Retrieval System
        at a given value of k for a single query (the average of precision@i
        values for i such that the ith document is truly relevant)

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The average precision value as a number between 0 and 1
        """

        # precision_vals = []
        rel_vals = []

        # # Calculate precision at i recommendations for the query
        # for i in range(1, k+1):
        #     precision_at_i = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, i)
        #     precision_vals.append(precision_at_i)

        # # Compute the predicted document is relevant or not
        # for docID in query_doc_IDs_ordered:
        #     rel_vals.append(1 if int(docID) in true_doc_IDs else 0)

        # # Compute product of precision and relevance at i
        # prod_of_precision_and_rel_at_i = []
        # for i in range(k):
        #     prod_of_precision_and_rel_at_i.append(precision_vals[i] * rel_vals[i])

        # averagePrecision = sum(prod_of_precision_and_rel_at_i) / len(true_doc_IDs)
        # return averagePrecision

        for i in range(k):
            if query_doc_IDs_ordered[i] in true_doc_IDs:
                rel_vals.append(i)

        if len(rel_vals) == 0:
            return 0
        
        precisions = []
        for i in rel_vals:
            precisions.append(self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, i+1))

        return np.sum(precisions) / len(rel_vals)

    def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
        """
        Computation of MAP of the Information Retrieval System
        at given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The MAP value as a number between 0 and 1
        """

        averagePrecisions = []
        for i in range(len(query_ids)):
            # Compute average precision for each query
            averagePrecisions.append(self.queryAveragePrecision(doc_IDs_ordered[i], query_ids[i], get_relevant_docs(query_ids[i], q_rels), k))
        
        return np.mean(averagePrecisions)