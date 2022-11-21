import pandas as pd
import numpy as np
from random import sample
import pickle


def load_mapping_files():

    term_to_string = pd.read_csv("./data/1_term_ID_to_string.txt", delimiter="\t",
                                 header=None, encoding='latin-1', names=['termID', 'termString'])

    term_to_concept = pd.read_csv("./data/3_term_ID_to_concept_ID.txt", delimiter="\t",
                                  header=None, encoding='latin-1', names=['termID', 'conceptID'])

    concept_to_CUI = pd.read_csv("./data/2b_concept_ID_to_CUI.txt", delimiter="\t",
                                 header=None, encoding='latin-1', names=['conceptID', 'CUI'])

    concept_to_string = pd.read_csv("./data/2a_concept_ID_to_string.txt", delimiter="\t",
                                    header=None, encoding='latin-1', names=['conceptID', 'ConceptString'])

    return term_to_string, term_to_concept, concept_to_CUI, concept_to_string


def load_freqs():

    co_occ = pd.read_csv("./data/cofreqs_terms_perBin_1d.txt", delimiter="\t", header=None, names=[
                         'termID1', 'termID2', 'co_occur_freq'])

    singlet = pd.read_csv("./data/singlets_terms_perBin_1d.txt", delimiter="\t",
                          header=None, names=['termID', 'singleton_freq'])

    return co_occ, singlet


def sample_terms(query_terms_list, rate=0.015):

    num = round(len(query_terms_list) * rate)

    return sample(query_terms_list, num)


def main():
    _, term_to_concept, concept_to_CUI, _ = load_mapping_files()

    co_occ, singlet = load_freqs()

    intermed = co_occ.merge(singlet, left_on="termID1", right_on="termID").rename(
        columns={"singleton_freq": "term1_freq"}).drop(columns='termID')

    intermed = intermed.merge(singlet, left_on="termID2", right_on="termID").rename(
        columns={"singleton_freq": "term2_freq"}).drop(columns='termID')

    intermed = intermed.merge(term_to_concept, left_on="termID1",
                              right_on="termID", how="left").drop(columns='termID')

    intermed = intermed.merge(
        concept_to_CUI, left_on="conceptID", right_on="conceptID", how="left")

    # Filter out terms not mapped to a CUI
    out_df = intermed[intermed['CUI'].notnull()]

    # Remove terms mapped to multiple CUI and keep first
    out_df = out_df.drop_duplicates(
        subset=['termID1', 'termID2'], keep='first')

    out_df['PMI'] = np.log((out_df['co_occur_freq'] * len(singlet)) /
                           (out_df['term1_freq'] * out_df['term2_freq']))

    out_df['PPMI'] = np.where(out_df['PMI'] < 0, 0, out_df['PMI'])

    query_terms_list = list(set(out_df['termID1'].to_list()))

    sampled_query_terms = sample_terms(query_terms_list)
    print('len sampled query terms', len(sampled_query_terms))

    # sample training
    print('sampling training...')
    dataset = {}

    output_list = []
    i = 0
    for singleton in sampled_query_terms:
        if i % 50 == 0:
            print(i)

        out_list = out_df[out_df['termID1'] == singleton][[
            'termID2', 'PPMI', 'co_occur_freq']].values.tolist()
        out_tuples = [tuple(x) for x in out_list]
        dataset[singleton] = out_tuples

        out_list = out_df[out_df['termID1'] ==
                          singleton]['termID2'].values.tolist()
        out_tuples = tuple(out_list)
        output_list.append((singleton, out_tuples))

        i += 1

    with open('./data/sub_neighbors_dict_ppmi_perBin_1.pkl', 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./data/train_multi_perBin_1.pkl', 'wb') as handle:
        pickle.dump(output_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # sample test
    print('sampling test...')
    test_query_terms = [
        x for x in query_terms_list if x not in sampled_query_terms]

    sampled_test_query_terms = sample_terms(test_query_terms, rate=0.04)
    print('sampled test terms', len(sampled_test_query_terms))

    test_output_list = []
    i = 0
    for singleton in sampled_test_query_terms:
        if i % 50 == 0:
            print(i)

        out_list = out_df[out_df['termID1'] ==
                          singleton]['termID2'].values.tolist()
        out_tuples = tuple(out_list)

        test_output_list.append((singleton, out_tuples))

        i += 1

    with open('./data/test_multi_perBin_1.pkl', 'wb') as handle:
        pickle.dump(test_output_list, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
