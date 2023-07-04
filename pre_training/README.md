# Pre-training on E-Manuals

- Files/Folders

1. `taxonomy_hierarchy` - contains files related to hierarchical classification labels
2. `hier_utils.py` - contains a set of loss function definitions used during pre-training by different models.
3. `only_classfn_rule_based_shards.py` - code for pre-training the `hier.` model
4. `triplet_net_rule_based_shards.py` - code for pre-training the `triplet` model
5. `heirarchical_classes_triplet_net_rule_based_shards.py` - code for pre-training the `triplet + hier.` model
6. `convert_to_HF_model.py` - converting `.pt` model generated during pre-training to a model contemporary with HuggingFace.
7. `models_.py` - contains classes for the pre-training architectures used.

> Before pre-training, please download the folders from https://drive.google.com/drive/folders/1MZW8YjC0NAk2tVioquiw9QCHq6t4iWxB?usp=sharing in this directory, which contain sentence embeddings that are used in the pre-training codes. 

> In order to pre-train RoBERTa-based variants, assign `model_type = 'roberta'` in the code, and for BERT-based variants, assign `model_type = 'bert'`.

> Also, for pre-training, run 

```
python3 <PRE-TRAINING FILENAME> 1
```
## Pre-training on Scientific Papers

Code is similar to that of pre-training on E-Manuals, except that it uses ArXiv Category Taxonomy (https://arxiv.org/category_taxonomy), and the triplets used are a subset of the paper triplets used for pre-training by [SPECTER](https://github.com/allenai/specter), such that all the papers in each triplet are present in ArXiv.
