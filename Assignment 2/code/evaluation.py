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

        DCGk = 0
        IDCGk = 0
        docs = query_doc_IDs_ordered[:k]

        for idx in range(1, k + 1):
            doc_rel = 0
            prev_idx = idx - 1
            if docs[prev_idx] in true_doc_IDs:
                doc_rel = self.true_docs_rels[true_doc_IDs.index(docs[prev_idx])]
            DCGk += doc_rel / log2(idx + 1)

        true_rels = self.true_docs_rels.copy()
        if len(true_rels) < k:
            for idx in range(len(true_rels), k):
                true_rels.append(0)

        IDCGk = sum([true_rels[idx - 1] / log2(idx + 1) for idx in range(1, k + 1)])

        if IDCGk == 0:
            return 0

        nDCG = DCGk / IDCGk

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

        NDCG_arr = []
        for idx, query_id in enumerate(query_ids):
            query_doc_IDs_ordered = doc_IDs_ordered[idx]
            # we need to store position to sort the relevant docs by position
            relevant_docs = [
                (d["position"], int(d["id"]))
                for d in qrels
                if d["query_num"] == str(query_id)
            ]
            if len(relevant_docs) == 0:
                print("No relevant docs found for query: ", query_id)

            # sort by position
            relevant_docs.sort()

            self.true_docs_rels = [5 - ele[0] for ele in relevant_docs]
            # we only need the doc_id
            true_doc_IDs = [ele[1] for ele in relevant_docs]

            # get precision@k and add it to the sum
            NDCG_arr.append(self.queryNDCG(query_doc_IDs_ordered, query_id, true_doc_IDs, k))

        meanNDCG = np.sum(NDCG_arr) / len(query_ids)

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

        rel_vals = []
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