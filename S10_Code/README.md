# Fine-tuning on the S10 Dataset

- Files

1. `section_retrieval.py` - This is the code for fine-tuning and evaluating on the S10 (question, section) pairs data, where, for each question, top 10 sections retrieved using BM25 are candidate ections, and if one of them is the section containing the answer, it is labelled 1, and all others are labelled 0.

2. `answer_retrieval_given_section.py` - This is the code for fine-tuning and evaluating on the S10 (question, sentence) pairs data, where, for each question, the sentences are that of the section that contains the ground truth answer. The sentences that are a part of the ground truth answer are labelled as 1, and otherwise 0.

3. `answer_retrieval_using_inference.py` - This is the code for evaluating the fie-tuned section retrieval and answer retrieval (given section) models on the S10 QA Dataset. The input is a list of questions of the S10 dataset, and the output is the list of answers predicted by the model in hand. Note that, the section retrieval and the answer retrieval (given section) models need to be present in `section_retrieval` and `answer_retrieval` folders respectively.

4. `smd.py` - This is taken from https://github.com/eaclark07/sms/tree/master/wmd-relax-master and has been modified. This contains code for calculating the S+WMS Metric.