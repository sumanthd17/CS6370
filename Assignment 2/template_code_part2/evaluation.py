from util import *

# Add your import statements here
from itertools import groupby
from operator import itemgetter
import numpy as np


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

        retrived_at_k_intersection_relavent_count = len(set(query_doc_IDs_ordered[:k]).intersection(true_doc_IDs)

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
            qrel_list.append(qrel_dict[query_id])

        precisons = []
        for retrived, query_id, relavent in zip(doc_IDs_ordered, query_ids, qrel_list):
            precisons.append(self.queryPrecision(retrived, query_id, relavent))

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

        retrived_at_k_intersection_relavent_count = len(set(query_doc_IDs_ordered[:k]).intersection(true_doc_IDs))

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
            qrel_list.append(qrel_dict[query_id])

        recalls = []
        for retrived, query_id, relavent in zip(doc_IDs_ordered, query_ids, qrel_list):
            recalls.append(self.queryRecall(retrived, query_id, relavent))

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

        precison = self.queryPrecision(query_doc_IDs_ordered,query_id,true_doc_IDs)
        recall = self.queryRecall(query_doc_IDs_ordered,query_id,true_doc_IDs)


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
                qrel_list.append(qrel_dict[query_id])

            f1s = []
            for retrived, query_id, relavent in zip(doc_IDs_ordered, query_ids, qrel_list):
                f1s.append(self.queryRecall(retrived, query_id, relavent))

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

        nDCG = -1

        # Fill in code here

        return nDCG

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

        meanNDCG = -1

        # Fill in code here

        return meanNDCG

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

        avgPrecision = -1

        # Fill in code here

        return avgPrecision

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

        meanAveragePrecision = -1

        # Fill in code here

        return meanAveragePrecision
