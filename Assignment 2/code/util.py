# Add your import statements here


# Add any utility functions here
def get_relevant_docs(query_id, qrels):
    relevant_doc_ids = []
    for query_dict in qrels:
        if int(query_dict["query_num"]) == int(query_id) and int(query_dict["position"]) <= 4:
            relevant_doc_ids.append(int(query_dict["id"]))
    return relevant_doc_ids
