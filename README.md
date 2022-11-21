# BD4H_FinalProject

This repository contains the code used to train the SurfCon model, which is taken from the original paper: https://arxiv.org/pdf/1906.09285.pdf and inspired by the original code: https://github.com/zhenwang9102/SurfCon.

## Installation

Ensure that all packages in requirements.txt are installed.

## Data folder

The raw datasets are too large to put on github. We have therefore put it on a google drive: https://drive.google.com/drive/u/1/folders/1cgPAdU0fCseDvFzW06raTABW6-de0UWc. Before running any code, ensure that the following files are copied to the data folder, where the data folder exists at the same level as src.

1. 1_term_ID_to_string
2. 2a_concept_ID_to_string
3. 2b_concept_ID_to_CUI
4. 3_term_ID_to_concept_ID
5. cofreqs_terms_perBin_1d
6. singlets_terms_perBin_1d
7. embedding/charNgram.txt
8. embedding/glove.6B.100d.txt
9. embedding/line2d_ttcooc_embedding.txt
10. mappings/concept_term_mapping.pkl
11. mappings/term_concept_mapping.pkl
12. mappings/term_string_mapping.pkl

## Dataset creation

generate_train_data.py generates the required training and testing dataset from the raw dataset. To run generate_train_data.py, run the following command from BD4H_FINALPROJECT.

```
python src/generate_train_data.py
```

This will generate three pickle files under the data folder. Two are required for model training, and the third is the terms used for testing.

1. sub_neighbors_dict_ppmi_perBin_1.pkl
2. train_multi_perBin_1.pkl
3. test_multi_perBin_1.pkl

## Training context prediction module

main_pretrain.py trains the context prediction module. To run main_pretrain.py, run the following command from BD4H_FINALPROJECT.

```
python src/main_pretrain.py
```

This will generate "Bin_1_pretrain_model_dict.pkl" and "snapshot_epoch_99.pt".

## Training SurfCon model

main_dym.py trains the SurfCon model. To run main_dym.py, run the following command from BD4H_FINALPROJECT.

```
python src/main_dym.py
```

This will generate "best_epoch_0.pt".
