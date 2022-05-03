# Add your import statements here
import os
import numpy as np
import matplotlib.pyplot as plt

# Add any utility functions here
def get_relevant_docs(query_id, qrels):
    relevant_doc_ids = []
    for query_dict in qrels:
        if int(query_dict["query_num"]) == int(query_id) and int(query_dict["position"]) <= 4:
            relevant_doc_ids.append(int(query_dict["id"]))
    return relevant_doc_ids

def plot_singular_values():
    SVD_COMPRESSED_FILE_PATH = 'compressedFiles/svd.npz'
    if os.path.exists(SVD_COMPRESSED_FILE_PATH):
        print('Loading SVD...')
        savedSVD = np.load(SVD_COMPRESSED_FILE_PATH)
        U, S, Vt = savedSVD['U'], savedSVD['S'], savedSVD['Vt']
        plt.plot(range(1, 1401), S, label="Singular values")
        plt.legend()
        plt.title("Plot of Singular Values vs Component Number")
        plt.xlabel("Component Number")
        plt.ylabel("Singular Values")
        plt.grid()
        plt.savefig("singular_values_vs_component_number.png")

def plot_cummulative_variance():
    SVD_COMPRESSED_FILE_PATH = 'compressedFiles/svd.npz'
    if os.path.exists(SVD_COMPRESSED_FILE_PATH):
        print('Loading SVD...')
        savedSVD = np.load(SVD_COMPRESSED_FILE_PATH)
        U, S, Vt = savedSVD['U'], savedSVD['S'], savedSVD['Vt']

        sum_of_squares_of_S = sum(list(map(lambda x: x ** 2, S)))
        pov_k_list = []
        cov_k_list = []
        cov_k = 0
        for k, sk in enumerate(S):
            current_sk = sk * sk
            pov_k = current_sk/sum_of_squares_of_S
            cov_k += pov_k
            pov_k_list.append(pov_k)
            cov_k_list.append(cov_k)
        plt.plot(range(0, 1400), cov_k_list, label="Cummulative Variance")
        plt.legend()
        plt.title("Plot of Cummulative Variance vs Component Number")
        plt.xlabel("Component Number")
        plt.ylabel("Cummulative Variance")
        plt.grid()
        plt.savefig("cummulative_variance.png")


def plot_with_different_ks():
    p10 = [0.1471111111111111, 0.19555555555555557, 0.2311111111111111, 0.23733333333333337, 0.21333333333333335, 0.16666666666666666, 0.14577777777777778]
    r10 = [0.20692381578804836, 0.2761403618918832, 0.3291043203380701, 0.34299920925834587, 0.32187174516947836, 0.2558931811979924, 0.22147781129756505]
    ndcg10 = [0.20226075908653432, 0.2995897758884247, 0.35281475751099406, 0.3886549625560004, 0.38436530325894974, 0.3122033325035396, 0.2789048638822594]
    kComponents = [20, 50, 100, 200, 500, 1000, 1200]

    plt.plot(kComponents, p10, label="Precision @ 10")
    plt.plot(kComponents, r10, label="Recall @ 10")
    plt.plot(kComponents, ndcg10, label="nDCG @ 10")
    plt.legend()
    plt.title("k for LSA vs Evaluation metric")
    plt.xlabel("k")
    plt.ylabel("Evaluation Metrics - Cranfield Dataset")
    plt.savefig("LSAvsMetrics.png")
