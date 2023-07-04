# Code for the Arxiv submission - 'FPDM: Domain-Specific Fast Pre-training Technique using Document-Level Metadata'

## Required dependencies

Please run `pip install -r requirements.txt` (`python3` required). For fine-tuning on the TechQA Dataset, use [this](./TechQA_code/requirements.txt).

## Links to models pre-trained on the EManuals Corpus

- Our proposed RoBERTa-based variants

1. [<em>FPDM<sub>RoBERTa</sub> (hier.)</em>](https://huggingface.co/AnonymousSub/rule_based_roberta_only_classfn_epochs_1_shard_1)
2. [<em>FPDM<sub>RoBERTa</sub> (triplet)</em>](https://huggingface.co/AnonymousSub/rule_based_roberta_bert_triplet_epochs_1_shard_1)
3. [<em>FPDM<sub>RoBERTa</sub></em>](https://huggingface.co/AnonymousSub/rule_based_roberta_hier_triplet_epochs_1_shard_1)

- Our proposed BERT-based variants

1. [<em>FPDM<sub>BERT</sub> (hier.)</em>](https://huggingface.co/AnonymousSub/rule_based_only_classfn_epochs_1_shard_1)
2. [<em>FPDM<sub>BERT</sub> (triplet)</em>](https://huggingface.co/AnonymousSub/rule_based_bert_triplet_epochs_1_shard_1)
3. [<em>FPDM<sub>BERT</sub></em>](https://huggingface.co/AnonymousSub/rule_based_hier_triplet_epochs_1_shard_1)

- Baselines

1. [BERT<sub>BASE</sub>](https://huggingface.co/bert-base-uncased)
2. [RoBERTa<sub>BASE</sub>](https://huggingface.co/roberta-base)
3. [Longformer](https://huggingface.co/allenai/longformer-base-4096)
4. [EManuals<sub>BERT</sub>](https://huggingface.co/abhi1nandy2/EManuals_BERT)
5. [EManuals<sub>RoBERTa</sub>](https://huggingface.co/abhi1nandy2/EManuals_RoBERTa)
6. [DeCLUTR](https://huggingface.co/AnonymousSub/declutr-model)
7. [ConSERT](https://huggingface.co/AnonymousSub/unsup-consert-base)
8. [SPECTER](https://huggingface.co/AnonymousSub/specter-bert-model)

## Links to the models pre-trained in the Scientific Domain

- Our Proposed Variants
1. [<em>FPDM(Sci.)<sub>BERT</sub></em>](https://huggingface.co/AnonymousSub/Scientific_FPDM_PRIM_CAT_BERT_HYBRID_epochs_1)
2. [<em>FPDM(Sci.)<sub>BERT</sub> (hier.)</em>](https://huggingface.co/AnonymousSub/Scientific_FPDM_PRIM_CAT_HIER_BERT_HYBRID_epochs_1)
3. [<em>FPDM(Sci.)<sub>BERT</sub> (triplet)</em>](https://huggingface.co/AnonymousSub/Scientific_FPDM_PRIM_CAT_TRIPLET_BERT_HYBRID_epochs_1)

- Baseline

1. SciBERT - Refer to https://github.com/allenai/scibert
<!---
2. [SPECTER](https://huggingface.co/allenai/specter)
3. [SciNCL](https://huggingface.co/malteos/scincl)
4. [BioMedRoBERTa](https://huggingface.co/allenai/biomed_roberta_base)
5. [RoBERTa<sub>BASE</sub>](https://huggingface.co/roberta-base)
6. [RoBERTa<sub>LARGE</sub>](https://huggingface.co/roberta-large)
-->
## Fine-tuning on SQuAD 2.0

- To download the training set, run `wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json`.

- Run `python3 finetune_squad.py <MODEL_TYPE> <MODEL_PATH>`
	- `<MODEL_TYPE>` can be `bert` or `roberta`
	- `<MODEL_PATH>` is the model path/HuggingFace model name.

> To get the models fine-tuned on SQuAD 2.0, follow the following format to get the link - `https://huggingface.co/AnonymousSub/<SUBSTRING AFTER THE LAST '/' IN PRE-TRAINED MODEL LINK>_squad2.0` (For example, the link to the model obtained after fine-tuning <em>FPDM<sub>RoBERTa</sub></em> - https://huggingface.co/AnonymousSub/rule_based_roberta_hier_triplet_epochs_1_shard_1 on SQuAD 2.0 is https://huggingface.co/AnonymousSub/rule_based_roberta_hier_triplet_epochs_1_shard_1_squad2.0)

## Fine-tuning on TechQA Dataset

- Go to [this link](./TechQA_code)

## Fine-tuning on S10 QA Dataset

- Go to [this link](./S10_Code)

## Fine-tuning on some of the SciBERT Paper Datasets

- Check all notebooks [here](./Scibert_datasets_code).

## Fine-tuning on GLUE Benchmark Datasets

- Check all notebooks [here](./GLUE_code).
