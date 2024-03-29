from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from evaluation import Evaluation
from bert import BERTIndex
from word2vec import Word2VecIndex
from SpellCheck import *
import time

from sys import version_info
import argparse
import json
import matplotlib.pyplot as plt

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print("Unknown python version - input function not safe")


class Word2VecEngine:
    def __init__(self, args):
        self.args = args

        self.word2vec = Word2VecIndex(args.train_word2vec)

        self.tokenizer = Tokenization()
        self.sentenceSegmenter = SentenceSegmentation()
        self.inflectionReducer = InflectionReduction()
        self.stopwordRemover = StopwordRemoval()
        self.evaluator = Evaluation()

    def segmentSentences(self, text):
        """
        Call the required sentence segmenter
        """
        if self.args.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)
        elif self.args.segmenter == "punkt":
            return self.sentenceSegmenter.punkt(text)

    def tokenize(self, text):
        """
        Call the required tokenizer
        """
        if self.args.tokenizer == "naive":
            return self.tokenizer.naive(text)
        elif self.args.tokenizer == "ptb":
            return self.tokenizer.pennTreeBank(text)

    def removeStopwords(self, text):
        """
        Call the required stopword remover
        """
        return self.stopwordRemover.fromList(text)

    def preprocessQueries(self, queries):
        """
        Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
        """

        # Segment queries
        segmentedQueries = []
        for query in queries:
            segmentedQuery = self.segmentSentences(query)
            segmentedQueries.append(segmentedQuery)
        json.dump(
            segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", "w")
        )

        # Tokenize queries
        tokenizedQueries = []
        for query in segmentedQueries:
            tokenizedQuery = self.tokenize(query)
            tokenizedQueries.append(tokenizedQuery)
        json.dump(
            tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", "w")
        )

        # Remove stop words
        stopwordRemovedQueries = []
        for query in tokenizedQueries:
            stopwordRemovedQuery = self.removeStopwords(query)
            stopwordRemovedQueries.append(stopwordRemovedQuery)
        json.dump(
            stopwordRemovedQueries,
            open(self.args.out_folder + "stopword_removed_queries.txt", "w"),
        )

        preprocessedQueries = stopwordRemovedQueries
        return preprocessedQueries

    def preprocessDocs(self, docs):
        """
        Preprocess the documents
        """

        # Segment docs
        segmentedDocs = []
        for doc in docs:
            segmentedDoc = self.segmentSentences(doc)
            segmentedDocs.append(segmentedDoc)
        json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", "w"))

        # Tokenize docs
        tokenizedDocs = []
        for doc in segmentedDocs:
            tokenizedDoc = self.tokenize(doc)
            tokenizedDocs.append(tokenizedDoc)
        json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", "w"))

        # Remove stopwords from docs
        stopwordRemovedDocs = []
        for doc in tokenizedDocs:
            stopwordRemovedDoc = self.removeStopwords(doc)
            stopwordRemovedDocs.append(stopwordRemovedDoc)
        json.dump(
            stopwordRemovedDocs,
            open(self.args.out_folder + "stopword_removed_docs.txt", "w"),
        )

        preprocessedDocs = stopwordRemovedDocs
        return preprocessedDocs

    def evaluateDataset(self):
        """
        - preprocesses the queries and documents, stores in output folder
        - invokes the IR system
        - evaluates precision, recall, fscore, nDCG and MAP
          for all queries in the Cranfield dataset
        - produces graphs of the evaluation metrics in the output folder
        """

        # Read queries
        queries_json = json.load(open(args.dataset + "cran_queries.json", "r"))[:]
        query_ids, queries = (
            [item["query number"] for item in queries_json],
            [item["query"] for item in queries_json],
        )
        # Process queries
        processedQueries = self.preprocessQueries(queries)

        # Read documents
        docs_json = json.load(open(args.dataset + "cran_docs.json", "r"))[:]
        if args.use_title:
            doc_ids, docs = (
                [item["id"] for item in docs_json],
                [item["title"] + item["body"] for item in docs_json],
            )
        else:
            doc_ids, docs = (
                [item["id"] for item in docs_json],
                [item["body"] for item in docs_json],
            )
        # Process documents
        processedDocs = self.preprocessDocs(docs)

        # Build document index
        self.word2vec.buildIndex(processedDocs, doc_ids)
        # Rank the documents for each query
        doc_IDs_ordered = self.word2vec.rank(processedQueries)

        # Read relevance judements
        qrels = json.load(open(args.dataset + "cran_qrels.json", "r"))[:]

        # Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
        precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
        for k in range(1, 11):
            precision = self.evaluator.meanPrecision(
                doc_IDs_ordered, query_ids, qrels, k
            )
            precisions.append(precision)
            recall = self.evaluator.meanRecall(doc_IDs_ordered, query_ids, qrels, k)
            recalls.append(recall)
            fscore = self.evaluator.meanFscore(doc_IDs_ordered, query_ids, qrels, k)
            fscores.append(fscore)
            print(
                "Precision, Recall and F-score @ "
                + str(k)
                + " : "
                + str(precision)
                + ", "
                + str(recall)
                + ", "
                + str(fscore)
            )
            MAP = self.evaluator.meanAveragePrecision(
                doc_IDs_ordered, query_ids, qrels, k
            )
            MAPs.append(MAP)
            nDCG = self.evaluator.meanNDCG(doc_IDs_ordered, query_ids, qrels, k)
            nDCGs.append(nDCG)
            print("MAP, nDCG @ " + str(k) + " : " + str(MAP) + ", " + str(nDCG))

        # Plot the metrics and save plot
        plt.plot(range(1, 11), precisions, label="Precision")
        plt.plot(range(1, 11), recalls, label="Recall")
        plt.plot(range(1, 11), fscores, label="F-Score")
        plt.plot(range(1, 11), MAPs, label="MAP")
        plt.plot(range(1, 11), nDCGs, label="nDCG")
        plt.legend()
        plt.title("Evaluation Metrics - Cranfield Dataset")
        plt.xlabel("k")
        plt.savefig(args.out_folder + "eval_plot.png")


class BERTEngine:
    def __init__(self, args):
        self.args = args

        self.bert = BERTIndex()
        self.sentenceSegmenter = SentenceSegmentation()
        self.evaluator = Evaluation()

    def segmentSentences(self, text):
        """
        Call the required sentence segmenter
        """
        if self.args.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)
        elif self.args.segmenter == "punkt":
            return self.sentenceSegmenter.punkt(text)

    def preprocessQueries(self, queries):
        """
        Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
        """

        # Segment queries
        segmentedQueries = []
        for query in queries:
            segmentedQuery = self.segmentSentences(query)
            segmentedQueries.append(segmentedQuery)
        json.dump(
            segmentedQueries,
            open(self.args.out_folder + "bert_segmented_queries.txt", "w"),
        )

        preprocessedQueries = segmentedQueries
        return preprocessedQueries

    def preprocessDocs(self, docs):
        """
        Preprocess the documents
        """

        # Segment docs
        segmentedDocs = []
        for doc in docs:
            segmentedDoc = self.segmentSentences(doc)
            segmentedDocs.append(segmentedDoc)
        json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", "w"))

        preprocessedDocs = segmentedDocs
        return preprocessedDocs

    def evaluateDataset(self):
        """
        - preprocesses the queries and documents, stores in output folder
        - invokes the IR system
        - evaluates precision, recall, fscore, nDCG and MAP
          for all queries in the Cranfield dataset
        - produces graphs of the evaluation metrics in the output folder
        """

        # Read queries
        queries_json = json.load(open(args.dataset + "cran_queries.json", "r"))[:]
        query_ids, queries = (
            [item["query number"] for item in queries_json],
            [item["query"] for item in queries_json],
        )
        # Process queries
        processedQueries = self.preprocessQueries(queries)

        # Read documents
        docs_json = json.load(open(args.dataset + "cran_docs.json", "r"))[:]
        if args.use_title:
            doc_ids, docs = (
                [item["id"] for item in docs_json],
                [item["title"] + item["body"] for item in docs_json],
            )
        else:
            doc_ids, docs = (
                [item["id"] for item in docs_json],
                [item["body"] for item in docs_json],
            )
        # Process documents
        processedDocs = self.preprocessDocs(docs)

        # Build BERT document index
        self.bert.buildIndex(processedDocs, doc_ids)
        doc_IDs_ordered = self.bert.rank(processedQueries)

        # Read relevance judements
        qrels = json.load(open(args.dataset + "cran_qrels.json", "r"))[:]

        # Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
        precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
        for k in range(1, 11):
            precision = self.evaluator.meanPrecision(
                doc_IDs_ordered, query_ids, qrels, k
            )
            precisions.append(precision)
            recall = self.evaluator.meanRecall(doc_IDs_ordered, query_ids, qrels, k)
            recalls.append(recall)
            fscore = self.evaluator.meanFscore(doc_IDs_ordered, query_ids, qrels, k)
            fscores.append(fscore)
            print(
                "Precision, Recall and F-score @ "
                + str(k)
                + " : "
                + str(precision)
                + ", "
                + str(recall)
                + ", "
                + str(fscore)
            )
            MAP = self.evaluator.meanAveragePrecision(
                doc_IDs_ordered, query_ids, qrels, k
            )
            MAPs.append(MAP)
            nDCG = self.evaluator.meanNDCG(doc_IDs_ordered, query_ids, qrels, k)
            nDCGs.append(nDCG)
            print("MAP, nDCG @ " + str(k) + " : " + str(MAP) + ", " + str(nDCG))

        # Plot the metrics and save plot
        plt.plot(range(1, 11), precisions, label="Precision")
        plt.plot(range(1, 11), recalls, label="Recall")
        plt.plot(range(1, 11), fscores, label="F-Score")
        plt.plot(range(1, 11), MAPs, label="MAP")
        plt.plot(range(1, 11), nDCGs, label="nDCG")
        plt.legend()
        plt.title("Evaluation Metrics - Cranfield Dataset")
        plt.xlabel("k")
        plt.savefig(args.out_folder + "eval_plot.png")


class SearchEngine:
    def __init__(self, args):
        self.args = args

        self.tokenizer = Tokenization()
        self.sentenceSegmenter = SentenceSegmentation()
        self.inflectionReducer = InflectionReduction()
        self.stopwordRemover = StopwordRemoval()

        self.informationRetriever = InformationRetrieval()
        self.evaluator = Evaluation()

    def segmentSentences(self, text):
        """
        Call the required sentence segmenter
        """
        if self.args.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)
        elif self.args.segmenter == "punkt":
            return self.sentenceSegmenter.punkt(text)

    def tokenize(self, text):
        """
        Call the required tokenizer
        """
        if self.args.tokenizer == "naive":
            return self.tokenizer.naive(text)
        elif self.args.tokenizer == "ptb":
            return self.tokenizer.pennTreeBank(text)

    def reduceInflection(self, text):
        """
        Call the required stemmer/lemmatizer
        """
        return self.inflectionReducer.reduce(text)

    def removeStopwords(self, text):
        """
        Call the required stopword remover
        """
        return self.stopwordRemover.fromList(text)

    def preprocessQueries(self, queries):
        """
        Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
        """

        # Segment queries
        segmentedQueries = []
        for query in queries:
            segmentedQuery = self.segmentSentences(query)
            segmentedQueries.append(segmentedQuery)
        json.dump(
            segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", "w")
        )

        # Tokenize queries
        tokenizedQueries = []
        for query in segmentedQueries:
            tokenizedQuery = self.tokenize(query)
            tokenizedQueries.append(tokenizedQuery)
        json.dump(
            tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", "w")
        )

        # Stem/Lemmatize queries
        reducedQueries = []
        for query in tokenizedQueries:
            reducedQuery = self.reduceInflection(query)
            reducedQueries.append(reducedQuery)
        json.dump(
            reducedQueries, open(self.args.out_folder + "reduced_queries.txt", "w")
        )

        # Remove stopwords from queries
        stopwordRemovedQueries = []
        for query in reducedQueries:
            stopwordRemovedQuery = self.removeStopwords(query)
            stopwordRemovedQueries.append(stopwordRemovedQuery)
        json.dump(
            stopwordRemovedQueries,
            open(self.args.out_folder + "stopword_removed_queries.txt", "w"),
        )

        preprocessedQueries = stopwordRemovedQueries
        return preprocessedQueries

    def preprocessDocs(self, docs):
        """
        Preprocess the documents
        """

        # Segment docs
        segmentedDocs = []
        for doc in docs:
            segmentedDoc = self.segmentSentences(doc)
            segmentedDocs.append(segmentedDoc)
        json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", "w"))

        # Tokenize docs
        tokenizedDocs = []
        for doc in segmentedDocs:
            tokenizedDoc = self.tokenize(doc)
            tokenizedDocs.append(tokenizedDoc)
        json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", "w"))

        # Stem/Lemmatize docs
        reducedDocs = []
        for doc in tokenizedDocs:
            reducedDoc = self.reduceInflection(doc)
            reducedDocs.append(reducedDoc)
        json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", "w"))

        # Remove stopwords from docs
        stopwordRemovedDocs = []
        for doc in reducedDocs:
            stopwordRemovedDoc = self.removeStopwords(doc)
            stopwordRemovedDocs.append(stopwordRemovedDoc)
        json.dump(
            stopwordRemovedDocs,
            open(self.args.out_folder + "stopword_removed_docs.txt", "w"),
        )

        preprocessedDocs = stopwordRemovedDocs
        return preprocessedDocs

    def evaluateDataset(self):
        """
        - preprocesses the queries and documents, stores in output folder
        - invokes the IR system
        - evaluates precision, recall, fscore, nDCG and MAP
          for all queries in the Cranfield dataset
        - produces graphs of the evaluation metrics in the output folder
        """

        # Read queries
        queries_json = json.load(open(args.dataset + "cran_queries.json", "r"))[:]
        query_ids, queries = (
            [item["query number"] for item in queries_json],
            [item["query"] for item in queries_json],
        )

        # Read documents
        docs_json = json.load(open(args.dataset + "cran_docs.json", "r"))[:]
        if args.use_title:
            doc_ids, docs = (
                [item["id"] for item in docs_json],
                [item["title"] * 3 + item["body"] for item in docs_json],
            )
        else:
            doc_ids, docs = (
                [item["id"] for item in docs_json],
                [item["body"] for item in docs_json],
            )
        
        if self.args.spell_correction == 'bigrams':
            print('doing bigram spellcheck...')
            bigramspellcheck = BigramSpellCheck(docs)
            start = time.time()
            for i, query in enumerate(queries):
                queries[i] = bigramspellcheck.correct_words_in_query(query)
            end = time.time()
            print("Time to correct all queries {} seconds".format(end-start))
        elif self.args.spell_correction == 'oneedit':
            print('doing one edit spellcheck...')
            bigramspellcheck = OneEditSpellCheck(docs)
            start = time.time()
            for i, query in enumerate(queries):
                queries[i] = bigramspellcheck.correct_words_in_query(query)
            end = time.time()
            print("Time to correct all queries {} seconds".format(end-start))

        # Process queries
        processedQueries = self.preprocessQueries(queries)

        # Process documents
        processedDocs = self.preprocessDocs(docs)

        # Build document index
        self.informationRetriever.buildIndex(processedDocs, doc_ids)
        # Rank the documents for each query
        print("The method is: {}".format(self.args.method))
        if self.args.method == "lsa":
            print("K is: {}".format(self.args.k))
            doc_IDs_ordered = self.informationRetriever.rank_by_lsa(
                processedQueries, self.args.k
            )
        elif self.args.method == "query_expansion":
            doc_IDs_ordered = self.informationRetriever.rank_by_query_expansion(
                processedQueries
            )
        else:
            doc_IDs_ordered = self.informationRetriever.rank(processedQueries)

        # Read relevance judements
        qrels = json.load(open(args.dataset + "cran_qrels.json", "r"))[:]

        # Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
        precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
        for k in range(1, 11):
            precision = self.evaluator.meanPrecision(
                doc_IDs_ordered, query_ids, qrels, k
            )
            precisions.append(precision)
            recall = self.evaluator.meanRecall(doc_IDs_ordered, query_ids, qrels, k)
            recalls.append(recall)
            fscore = self.evaluator.meanFscore(doc_IDs_ordered, query_ids, qrels, k)
            fscores.append(fscore)
            print(
                "Precision, Recall and F-score @ "
                + str(k)
                + " : "
                + str(precision)
                + ", "
                + str(recall)
                + ", "
                + str(fscore)
            )
            MAP = self.evaluator.meanAveragePrecision(
                doc_IDs_ordered, query_ids, qrels, k
            )
            MAPs.append(MAP)
            nDCG = self.evaluator.meanNDCG(doc_IDs_ordered, query_ids, qrels, k)
            nDCGs.append(nDCG)
            print("MAP, nDCG @ " + str(k) + " : " + str(MAP) + ", " + str(nDCG))

        # Plot the metrics and save plot
        plt.plot(range(1, 11), precisions, label="Precision")
        plt.plot(range(1, 11), recalls, label="Recall")
        plt.plot(range(1, 11), fscores, label="F-Score")
        plt.plot(range(1, 11), MAPs, label="MAP")
        plt.plot(range(1, 11), nDCGs, label="nDCG")
        plt.legend()
        plt.title("Evaluation Metrics - Cranfield Dataset")
        plt.xlabel("k")
        plt.savefig(args.out_folder + "eval_plot.png")

    def handleCustomQuery(self):
        """
        Take a custom query as input and return top five relevant documents
        """

        # Get query
        print("Enter query below")
        query = input()

        # Read documents
        docs_json = json.load(open(args.dataset + "cran_docs.json", "r"))[:]
        doc_ids, docs = (
            [item["id"] for item in docs_json],
            [item["body"] for item in docs_json],
        )

        if self.args.spell_correction == 'bigrams':
            print('doing bigram spellcheck...')
            bigramspellcheck = BigramSpellCheck(docs)
            query = bigramspellcheck.correct_words_in_query(query)
        elif self.args.spell_correction == 'oneedit':
            print('doing one edit spellcheck...')
            oneEditSpellCheck = OneEditSpellCheck(docs)
            query = oneEditSpellCheck.correct_words_in_query(query)

        # Process documents
        processedQuery = self.preprocessQueries([query])[0]

        # Process documents
        processedDocs = self.preprocessDocs(docs)

        # Build document index
        self.informationRetriever.buildIndex(processedDocs, doc_ids)
        # Rank the documents for the query
        doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0]

        # Print the IDs of first five documents
        print("\nTop five document IDs : ")
        for id_ in doc_IDs_ordered[:5]:
            print(id_)


class WordnetSearchEngine:
    def __init__(self, args):
        self.args = args

        self.tokenizer = Tokenization()
        self.sentenceSegmenter = SentenceSegmentation()
        self.inflectionReducer = InflectionReduction()
        self.stopwordRemover = StopwordRemoval()

        self.informationRetriever = InformationRetrieval()
        self.evaluator = Evaluation()

    def segmentSentences(self, text):
        """
            Call the required sentence segmenter
            """
        if self.args.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)
        elif self.args.segmenter == "punkt":
            return self.sentenceSegmenter.punkt(text)

    def tokenize(self, text):
        """
            Call the required tokenizer
            """
        if self.args.tokenizer == "naive":
            return self.tokenizer.naive(text)
        elif self.args.tokenizer == "ptb":
            return self.tokenizer.pennTreeBank(text)

    def reduceInflection(self, text):
        """
            Call the required stemmer/lemmatizer
            """
        return self.inflectionReducer.reduce(text)

    def removeStopwords(self, text):
        """
            Call the required stopword remover
            """
        return self.stopwordRemover.fromList(text)

    def preprocessQueries(self, queries):
        """
            Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
            """

        # Segment queries
        segmentedQueries = []
        for query in queries:
            segmentedQuery = self.segmentSentences(query)
            segmentedQueries.append(segmentedQuery)
        json.dump(
            segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", "w")
        )
        # Tokenize queries
        tokenizedQueries = []
        for query in segmentedQueries:
            tokenizedQuery = self.tokenize(query)
            tokenizedQueries.append(tokenizedQuery)
        json.dump(
            tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", "w")
        )
        # Stem/Lemmatize queries
        reducedQueries = []
        for query in tokenizedQueries:
            reducedQuery = self.reduceInflection(query)
            reducedQueries.append(reducedQuery)
        json.dump(
            reducedQueries, open(self.args.out_folder + "reduced_queries.txt", "w")
        )
        # Remove stopwords from queries
        stopwordRemovedQueries = []
        for query in reducedQueries:
            stopwordRemovedQuery = self.removeStopwords(query)
            stopwordRemovedQueries.append(stopwordRemovedQuery)
        json.dump(
            stopwordRemovedQueries,
            open(self.args.out_folder + "stopword_removed_queries.txt", "w"),
        )

        preprocessedQueries = stopwordRemovedQueries
        return preprocessedQueries

    def preprocessDocs(self, docs):
        """
            Preprocess the documents
            """

        # Segment docs
        segmentedDocs = []
        for doc in docs:
            segmentedDoc = self.segmentSentences(doc)
            segmentedDocs.append(segmentedDoc)
        json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", "w"))
        # Tokenize docs
        tokenizedDocs = []
        for doc in segmentedDocs:
            tokenizedDoc = self.tokenize(doc)
            tokenizedDocs.append(tokenizedDoc)
        json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", "w"))
        # Stem/Lemmatize docs
        reducedDocs = []
        for doc in tokenizedDocs:
            reducedDoc = self.reduceInflection(doc)
            reducedDocs.append(reducedDoc)
        json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", "w"))
        # Remove stopwords from docs
        stopwordRemovedDocs = []
        for doc in reducedDocs:
            stopwordRemovedDoc = self.removeStopwords(doc)
            stopwordRemovedDocs.append(stopwordRemovedDoc)
        json.dump(
            stopwordRemovedDocs,
            open(self.args.out_folder + "stopword_removed_docs.txt", "w"),
        )

        preprocessedDocs = stopwordRemovedDocs
        return preprocessedDocs

    def evaluateDataset(self):
        """
            - preprocesses the queries and documents, stores in output folder
            - invokes the IR system
            - evaluates precision, recall, fscore, nDCG and MAP
              for all queries in the Cranfield dataset
            - produces graphs of the evaluation metrics in the output folder
            """

        # Read queries
        queries_json = json.load(open(args.dataset + "cran_queries.json", "r"))[:]
        query_ids, queries = (
            [item["query number"] for item in queries_json],
            [item["query"] for item in queries_json],
        )
        # Process queries
        processedQueries = self.preprocessQueries(queries)

        # Read documents
        docs_json = json.load(open(args.dataset + "cran_docs.json", "r"))[:]
        if args.use_title:
            doc_ids, docs = (
                [item["id"] for item in docs_json],
                [item["title"] + item["body"] for item in docs_json],
            )
        else:
            doc_ids, docs = (
                [item["id"] for item in docs_json],
                [item["body"] for item in docs_json],
            )
        # Process documents
        processedDocs = self.preprocessDocs(docs)

        # Build document index
        self.informationRetriever.buildIndex(processedDocs, doc_ids)
        # Rank the documents for each query
        doc_IDs_ordered = self.informationRetriever.wordnet(
            processedQueries, 200, False
        )

        # Read relevance judements
        qrels = json.load(open(args.dataset + "cran_qrels.json", "r"))[:]

        # Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
        precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
        for k in range(1, 11):
            precision = self.evaluator.meanPrecision(
                doc_IDs_ordered, query_ids, qrels, k
            )
            precisions.append(precision)
            recall = self.evaluator.meanRecall(doc_IDs_ordered, query_ids, qrels, k)
            recalls.append(recall)
            fscore = self.evaluator.meanFscore(doc_IDs_ordered, query_ids, qrels, k)
            fscores.append(fscore)
            print(
                "Precision, Recall and F-score @ "
                + str(k)
                + " : "
                + str(precision)
                + ", "
                + str(recall)
                + ", "
                + str(fscore)
            )
            MAP = self.evaluator.meanAveragePrecision(
                doc_IDs_ordered, query_ids, qrels, k
            )
            MAPs.append(MAP)
            nDCG = self.evaluator.meanNDCG(doc_IDs_ordered, query_ids, qrels, k)
            nDCGs.append(nDCG)
            print("MAP, nDCG @ " + str(k) + " : " + str(MAP) + ", " + str(nDCG))

        # Plot the metrics and save plot
        plt.plot(range(1, 11), precisions, label="Precision")
        plt.plot(range(1, 11), recalls, label="Recall")
        plt.plot(range(1, 11), fscores, label="F-Score")
        plt.plot(range(1, 11), MAPs, label="MAP")
        plt.plot(range(1, 11), nDCGs, label="nDCG")
        plt.legend()
        plt.title("Evaluation Metrics - Cranfield Dataset")
        plt.xlabel("k")
        plt.savefig(args.out_folder + "eval_plot.png")

    def handleCustomQuery(self):
        """
            Take a custom query as input and return top five relevant documents
            """

        # Get query
        print("Enter query below")
        query = input()
        # Process documents
        processedQuery = self.preprocessQueries([query])[0]

        # Read documents
        docs_json = json.load(open(args.dataset + "cran_docs.json", "r"))[:]
        doc_ids, docs = (
            [item["id"] for item in docs_json],
            [item["body"] for item in docs_json],
        )
        # Process documents
        processedDocs = self.preprocessDocs(docs)

        # Build document index
        self.informationRetriever.buildIndex(processedDocs, doc_ids)
        # Rank the documents for the query

        doc_IDs_ordered = self.informationRetriever.wordnet([processedQuery])[0]

        # Print the IDs of first five documents
        print("\nTop five document IDs : ")
        for id_ in doc_IDs_ordered[:5]:
            print(id_)

        def handleCustomQuery(self):
            """
            Take a custom query as input and return top five relevant documents
            """

            # Get query
            print("Enter query below")
            query = input()
            # Process documents
            processedQuery = self.preprocessQueries([query])[0]

            # Read documents
            docs_json = json.load(open(args.dataset + "cran_docs.json", "r"))[:]
            doc_ids, docs = (
                [item["id"] for item in docs_json],
                [item["body"] for item in docs_json],
            )
            # Process documents
            processedDocs = self.preprocessDocs(docs)

            # Build document index
            self.informationRetriever.buildIndex(processedDocs, doc_ids)
            # Rank the documents for the query
            doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0]

            # Print the IDs of first five documents
            print("\nTop five document IDs : ")
            for id_ in doc_IDs_ordered[:5]:
                print(id_)


class GloveSearchEngine:
    def __init__(self, args):
        self.args = args

        self.tokenizer = Tokenization()
        self.sentenceSegmenter = SentenceSegmentation()
        self.inflectionReducer = InflectionReduction()
        self.stopwordRemover = StopwordRemoval()

        self.informationRetriever = InformationRetrieval()
        self.evaluator = Evaluation()

    def segmentSentences(self, text):
        """
        Call the required sentence segmenter
        """
        if self.args.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)
        elif self.args.segmenter == "punkt":
            return self.sentenceSegmenter.punkt(text)

    def tokenize(self, text):
        """
        Call the required tokenizer
        """
        if self.args.tokenizer == "naive":
            return self.tokenizer.naive(text)
        elif self.args.tokenizer == "ptb":
            return self.tokenizer.pennTreeBank(text)

    def reduceInflection(self, text):
        """
        Call the required stemmer/lemmatizer
        """
        return self.inflectionReducer.reduce(text)

    def removeStopwords(self, text):
        """
        Call the required stopword remover
        """
        return self.stopwordRemover.fromList(text)

    def preprocessQueries(self, queries):
        """
        Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
        """

        # Segment queries
        segmentedQueries = []
        for query in queries:
            segmentedQuery = self.segmentSentences(query)
            segmentedQueries.append(segmentedQuery)
        json.dump(
            segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", "w")
        )
        # Tokenize queries
        tokenizedQueries = []
        for query in segmentedQueries:
            tokenizedQuery = self.tokenize(query)
            tokenizedQueries.append(tokenizedQuery)
        json.dump(
            tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", "w")
        )
        # Stem/Lemmatize queries
        reducedQueries = []
        for query in tokenizedQueries:
            reducedQuery = self.reduceInflection(query)
            reducedQueries.append(reducedQuery)
        json.dump(
            reducedQueries, open(self.args.out_folder + "reduced_queries.txt", "w")
        )
        # Remove stopwords from queries
        stopwordRemovedQueries = []
        for query in reducedQueries:
            stopwordRemovedQuery = self.removeStopwords(query)
            stopwordRemovedQueries.append(stopwordRemovedQuery)
        json.dump(
            stopwordRemovedQueries,
            open(self.args.out_folder + "stopword_removed_queries.txt", "w"),
        )

        preprocessedQueries = stopwordRemovedQueries
        return preprocessedQueries

    def preprocessDocs(self, docs):
        """
        Preprocess the documents
        """

        # Segment docs
        segmentedDocs = []
        for doc in docs:
            segmentedDoc = self.segmentSentences(doc)
            segmentedDocs.append(segmentedDoc)
        json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", "w"))
        # Tokenize docs
        tokenizedDocs = []
        for doc in segmentedDocs:
            tokenizedDoc = self.tokenize(doc)
            tokenizedDocs.append(tokenizedDoc)
        json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", "w"))
        # Stem/Lemmatize docs
        reducedDocs = []
        for doc in tokenizedDocs:
            reducedDoc = self.reduceInflection(doc)
            reducedDocs.append(reducedDoc)
        json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", "w"))
        # Remove stopwords from docs
        stopwordRemovedDocs = []
        for doc in reducedDocs:
            stopwordRemovedDoc = self.removeStopwords(doc)
            stopwordRemovedDocs.append(stopwordRemovedDoc)
        json.dump(
            stopwordRemovedDocs,
            open(self.args.out_folder + "stopword_removed_docs.txt", "w"),
        )

        preprocessedDocs = stopwordRemovedDocs
        return preprocessedDocs

    def evaluateDataset(self):
        """
        - preprocesses the queries and documents, stores in output folder
        - invokes the IR system
        - evaluates precision, recall, fscore, nDCG and MAP
          for all queries in the Cranfield dataset
        - produces graphs of the evaluation metrics in the output folder
        """

        # Read queries
        queries_json = json.load(open(args.dataset + "cran_queries.json", "r"))[:]
        query_ids, queries = (
            [item["query number"] for item in queries_json],
            [item["query"] for item in queries_json],
        )
        # Process queries
        processedQueries = self.preprocessQueries(queries)

        # Read documents
        docs_json = json.load(open(args.dataset + "cran_docs.json", "r"))[:]
        if args.use_title:
            doc_ids, docs = (
                [item["id"] for item in docs_json],
                [item["title"] + item["body"] for item in docs_json],
            )
        else:
            doc_ids, docs = (
                [item["id"] for item in docs_json],
                [item["body"] for item in docs_json],
            )
        # Process documents
        processedDocs = self.preprocessDocs(docs)

        # Build document index
        self.informationRetriever.buildIndex(processedDocs, doc_ids)
        # Rank the documents for each query
        doc_IDs_ordered = self.informationRetriever.glove(processedQueries)

        # Read relevance judements
        qrels = json.load(open(args.dataset + "cran_qrels.json", "r"))[:]

        # Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
        precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
        for k in range(1, 11):
            precision = self.evaluator.meanPrecision(
                doc_IDs_ordered, query_ids, qrels, k
            )
            precisions.append(precision)
            recall = self.evaluator.meanRecall(doc_IDs_ordered, query_ids, qrels, k)
            recalls.append(recall)
            fscore = self.evaluator.meanFscore(doc_IDs_ordered, query_ids, qrels, k)
            fscores.append(fscore)
            print(
                "Precision, Recall and F-score @ "
                + str(k)
                + " : "
                + str(precision)
                + ", "
                + str(recall)
                + ", "
                + str(fscore)
            )
            MAP = self.evaluator.meanAveragePrecision(
                doc_IDs_ordered, query_ids, qrels, k
            )
            MAPs.append(MAP)
            nDCG = self.evaluator.meanNDCG(doc_IDs_ordered, query_ids, qrels, k)
            nDCGs.append(nDCG)
            print("MAP, nDCG @ " + str(k) + " : " + str(MAP) + ", " + str(nDCG))

        # Plot the metrics and save plot
        plt.plot(range(1, 11), precisions, label="Precision")
        plt.plot(range(1, 11), recalls, label="Recall")
        plt.plot(range(1, 11), fscores, label="F-Score")
        plt.plot(range(1, 11), MAPs, label="MAP")
        plt.plot(range(1, 11), nDCGs, label="nDCG")
        plt.legend()
        plt.title("Evaluation Metrics - Cranfield Dataset")
        plt.xlabel("k")
        plt.savefig(args.out_folder + "eval_plot.png")

    def handleCustomQuery(self):
        """
        Take a custom query as input and return top five relevant documents
        """

        # Get query
        print("Enter query below")
        query = input()
        # Process documents
        processedQuery = self.preprocessQueries([query])[0]

        # Read documents
        docs_json = json.load(open(args.dataset + "cran_docs.json", "r"))[:]
        doc_ids, docs = (
            [item["id"] for item in docs_json],
            [item["body"] for item in docs_json],
        )
        # Process documents
        processedDocs = self.preprocessDocs(docs)

        # Build document index
        self.informationRetriever.buildIndex(processedDocs, doc_ids)
        # Rank the documents for the query
        doc_IDs_ordered = self.informationRetriever.glove(processedQuery)

        # Print the IDs of first five documents
        print("\nTop five document IDs : ")
        for id_ in doc_IDs_ordered[:5]:
            print(id_)


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="main.py")

    # Tunable parameters as external arguments
    parser.add_argument(
        "--dataset", default="cranfield/", help="Path to the dataset folder"
    )
    parser.add_argument("--out_folder", default="output/", help="Path to output folder")
    parser.add_argument(
        "--segmenter", default="punkt", help="Sentence Segmenter Type [naive|punkt]"
    )
    parser.add_argument("--tokenizer", default="ptb", help="Tokenizer Type [naive|ptb]")
    parser.add_argument(
        "--custom", action="store_true", help="Take custom query as input"
    )
    parser.add_argument("--method", default="vector_space")
    parser.add_argument("--train_word2vec", default=False, required=False)
    parser.add_argument("--use_title", default=False, type=bool)
    parser.add_argument("--spell_correction", default="", help="Spelling correction [bigrams|oneedit]")
    parser.add_argument("-k", default=220, type=int, help="K important features [k]")

    # Parse the input arguments
    args = parser.parse_args()

    # Create an instance of the Search Engine
    if (
        args.method == "vector_space"
        or args.method == "query_expansion"
        or args.method == "lsa"
    ):
        searchEngine = SearchEngine(args)
        if args.custom:
            searchEngine.handleCustomQuery()
        else:
            searchEngine.evaluateDataset()

    elif args.method == "word2vec":
        embeddings = Word2VecEngine(args)
        if args.custom:
            embeddings.handleCustomQuery()
        else:
            embeddings.evaluateDataset()

    elif args.method == "bert":
        embeddings = BERTEngine(args)
        if args.custom:
            embeddings.handleCustomQuery()
        else:
            embeddings.evaluateDataset()

    elif args.method == "wordnet":
        embeddings = WordnetSearchEngine(args)
        if args.custom:
            embeddings.handleCustomQuery()
        else:
            embeddings.evaluateDataset()

    elif args.method == "glove":
        embeddings = GloveSearchEngine(args)
        if args.custom:
            embeddings.handleCustomQuery()
        else:
            embeddings.evaluateDataset()
