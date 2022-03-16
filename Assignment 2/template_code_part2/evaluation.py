from util import *

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
        query_doc_IDs_ordered_list = []
        for i in query_doc_IDs_ordered:
            query_doc_IDs_ordered_list.append(str(i))

        true_doc_IDs_list = []
        for i in true_doc_IDs:
            true_doc_IDs_list.append(str(i))

        retrived_at_k_intersection_relavent_count = len(set(query_doc_IDs_ordered_list[:k]).intersection(true_doc_IDs_list))

        # Fill in code here

        return retrived_at_k_intersection_relavent_count / len(query_doc_IDs_ordered)

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
        qrel_dict = {}
        for key, value in groupby(qrels,
                                  key=itemgetter('query_num')):
            ordered_res = [res['id'] for res in sorted(value, key=itemgetter('position'))]
            qrel_dict[key] = ordered_res

        qrel_list = []
        for query_id in query_ids:
            qrel_list.append(qrel_dict[str(query_id)])

        precisons = []
        for retrived, query_id, relavent in zip(doc_IDs_ordered, query_ids, qrel_list):
            precisons.append(self.queryPrecision(retrived, query_id, relavent,k))

        # Fill in code here

        return np.mean(precisons)

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

        query_doc_IDs_ordered_list = []
        for i in query_doc_IDs_ordered:
            query_doc_IDs_ordered_list.append(str(i))

        true_doc_IDs_list = []
        for i in true_doc_IDs:
            true_doc_IDs_list.append(str(i))

        retrived_at_k_intersection_relavent_count = len(set(query_doc_IDs_ordered_list[:k]).intersection(true_doc_IDs_list))

        # Fill in code here

        return retrived_at_k_intersection_relavent_count / len(true_doc_IDs)

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

        qrel_dict = {}
        for key, value in groupby(qrels,
                                  key=itemgetter('query_num')):
            ordered_res = [res['id'] for res in sorted(value, key=itemgetter('position'))]
            qrel_dict[key] = ordered_res

        qrel_list = []
        for query_id in query_ids:
            qrel_list.append(qrel_dict[str(query_id)])

        recalls = []
        for retrived, query_id, relavent in zip(doc_IDs_ordered, query_ids, qrel_list):
            recalls.append(self.queryRecall(retrived, query_id, relavent,k))

        # Fill in code here

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

        precison = self.queryPrecision(query_doc_IDs_ordered,query_id,true_doc_IDs,k)
        recall = self.queryRecall(query_doc_IDs_ordered,query_id,true_doc_IDs,k)

        if precison == 0 and recall == 0:
            return 0
        # Fill in code here

        return 2 * precison * recall / precison + recall

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

        qrel_dict = {}
        for key, value in groupby(qrels,
                                  key=itemgetter('query_num')):
            ordered_res = [res['id'] for res in sorted(value, key=itemgetter('position'))]
            qrel_dict[key] = ordered_res

        qrel_list = []
        for query_id in query_ids:
            qrel_list.append(qrel_dict[str(query_id)])

        f1s = []
        for retrived, query_id, relavent in zip(doc_IDs_ordered, query_ids, qrel_list):
            f1s.append(self.queryFscore(retrived, query_id, relavent,k))

        # Fill in code here

        return np.mean(f1s)

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

        precision_vals = []
        rel_vals = []

        # Calculate precision at i recommendations for the query
        for i in range(1, k+1):
            precision_at_i = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, i)
            precision_vals.append(precision_at_i)

        # Compute the predicted document is relevant or not
        for docID in query_doc_IDs_ordered:
            rel_vals.append(1 if int(docID) in true_doc_IDs else 0)

        # Compute product of precision and relevance at i
        prod_of_precision_and_rel_at_i = []
        for i in range(k):
            prod_of_precision_and_rel_at_i.append(precision_vals[i] * rel_vals[i])

        averagePrecision = sum(prod_of_precision_and_rel_at_i) / len(true_doc_IDs)
        return averagePrecision


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
            query_doc_IDs_ordered = doc_IDs_ordered[i]
            query_id = int(query_ids[i])
            
            # Capture all true relevant documents for this query 
            true_doc_IDs = []
            for q_rel in q_rels:
                if query_id == int(q_rel["query_num"]):
                    true_doc_IDs.append(int(q_rel["id"]))

            # Compute average precision for each query
            averagePrecision = self.queryAveragePrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
            averagePrecisions.append(averagePrecision)
        
        return sum(averagePrecisions)/len(averagePrecisions)