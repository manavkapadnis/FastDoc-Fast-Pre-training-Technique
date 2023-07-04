import os
import sys
import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs
# from transformers import AutoTokenizer, AutoModel
import numpy as np
from metric_utils import *
import json
import torch
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

MANUAL_SEED = 42
set_seed(MANUAL_SEED)

model_paths = '<LIST OF MODEL DIRECTORY NAMES>'

# if os.path.exists('AnonymousSub_models') == False:
# 	os.mkdir('AnonymousSub_models')

# for path in model_paths:
# 	tok = AutoTokenizer.from_pretrained('')

# then do rest of the stuff

save_dir = 'sr_plus_ar_results'

if os.path.exists(save_dir) == False:
	os.mkdir(save_dir)

data_folder = '<PATH TO SECTION RETRIEVAL GROUND TRUTH DATA FOLDER>'

train_filepath = os.path.join(data_folder, 'df_train.csv') 
valid_filepath = os.path.join(data_folder, 'df_valid.csv')
test_filepath = os.path.join(data_folder, 'df_test.csv')

with open(os.path.join(data_folder, 'test_Q_A.json'), 'r') as f:
	test_qa = json.load(f)

top_10_sections_list = [item['DOC_IDS'] for item in test_qa]

question_list = [item['QUESTION_TEXT'] for item in test_qa]
target_list = [item['ANSWER'] for item in test_qa]

with open(os.path.join(data_folder, 'temp_corpus.json'), 'r') as f:
	corpus_dict = json.load(f)

def get_df(filepath):
	df = pd.read_csv(filepath, usecols=['sentence_1', 'sentence_2', 'label'])
	df = df.rename(columns={'sentence_1': 'text_a', 'sentence_2': 'text_b', 'label': 'labels'}).dropna().reset_index(drop = True)
	return df

def get_predictions(test_model_outputs, model2, num_inps_per_q = 10):

	sent_1_list = []
	sent_2_list = []

	print(test_model_outputs.shape[0])

	df = pd.DataFrame(columns = ['text_a', 'text_b', 'labels'])

	class_1_outputs = test_model_outputs[:, 1]
	num_questions = int(test_model_outputs.shape[0]/num_inps_per_q)

	for i in range(num_questions):
		best_idx = np.argmax(class_1_outputs[i*num_inps_per_q:(i+1)*num_inps_per_q])
		section_id = top_10_sections_list[i][best_idx]

		ans_sent_list = corpus_dict[section_id]['text']

		for sent in ans_sent_list:
			sent_1_list.append(question_list[i])
			sent_2_list.append(sent)

	df['text_a'] = sent_1_list
	df['text_b'] = sent_2_list
	df['labels'] = 0

	test_result, test_model_outputs, test_wrong_predictions = model2.eval_model(df)

	q_pred_ans_dict = {}

	for test_idx in range(len(test_model_outputs)):
		q = sent_1_list[test_idx]
		if q not in q_pred_ans_dict:
			q_pred_ans_dict[q] = []
		if test_model_outputs[test_idx][1] > test_model_outputs[test_idx][0]:
			q_pred_ans_dict[q].append(sent_2_list[test_idx])

	for q in q_pred_ans_dict:
		q_pred_ans_dict[q] = ' '.join(q_pred_ans_dict[q])

	return q_pred_ans_dict

df_train = get_df(train_filepath)
df_valid = get_df(valid_filepath)
df_test = get_df(test_filepath)

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
steps_per_epoch = int(np.ceil(len(df_train) / float(TRAIN_BATCH_SIZE)))

evaluate_during_training_steps = int(steps_per_epoch/2)
save_steps = evaluate_during_training_steps
logging_steps = int(steps_per_epoch/5)
# MANUAL_SEED = 42

'''
train_args = {
	'n_gpu': 1,
	'max_seq_length': 512,
	'num_train_epochs': 4,
	'evaluate_during_training': True,
	'evaluate_during_training_verbose': True,
	# 'regression': True,
	'overwrite_output_dir': True,
	'use_cached_eval_features': True,
	# 'use_multiprocessing_for_evaluation': True,
	# 'reprocess_input_data': True,
	'logging_steps': logging_steps,
	'evaluate_during_training_steps': evaluate_during_training_steps,
	'manual_seed': MANUAL_SEED,
	'train_batch_size': TRAIN_BATCH_SIZE,
	'eval_batch_size': EVAL_BATCH_SIZE,
	'save_steps': save_steps
}

train_args['use_cached_eval_features'] = False
train_args['overwrite_output_dir'] = False
'''
# MODEL_NAME = 'roberta'

train_args = ClassificationArgs()

train_args.n_gpu = 1
train_args.max_seq_length = 512
train_args.evaluate_during_training = True
train_args.evaluate_during_training_steps = evaluate_during_training_steps
train_args.train_batch_size = TRAIN_BATCH_SIZE
train_args.eval_batch_size = EVAL_BATCH_SIZE
train_args.num_train_epochs = 4
train_args.MANUAL_SEED = MANUAL_SEED
train_args.save_steps = save_steps
train_args.learning_rate = 4e-5

print(len(model_paths))

# MODEL_DIR = sys.argv[2]
NUM_LABELS = 2

for MODEL_DIR in model_paths:

	if ('roberta' in MODEL_DIR.lower()) or ('declutr' in MODEL_DIR.lower()) or ('cline' in MODEL_DIR.lower()):
		MODEL_NAME = 'roberta'
	else:
		MODEL_NAME = 'bert'

# print('Loading best model ....')
	if 'best_model' in os.listdir('outputs'):
		model = ClassificationModel(
			MODEL_NAME, "section_retrieval/{}/outputs/best_model".format(MODEL_DIR), num_labels=NUM_LABELS, args=train_args
		)
	else:
		model = ClassificationModel(
			MODEL_NAME, "section_retrieval/{}/outputs".format(MODEL_DIR), num_labels=NUM_LABELS, args=train_args
		)
	if 'best_model' in os.listdir('outputs'):
		model2 = ClassificationModel(
			MODEL_NAME, "answer_retrieval/{}/outputs/best_model".format(MODEL_DIR), num_labels=NUM_LABELS, args=train_args
		)
	else:
		model2 = ClassificationModel(
			MODEL_NAME, "answer_retrieval/{}/outputs".format(MODEL_DIR), num_labels=NUM_LABELS, args=train_args
		)		

	test_result, test_model_outputs, test_wrong_predictions = model.eval_model(df_test)

	print(test_model_outputs[:10])





	q_pred_ans_dict = get_predictions(test_model_outputs, model2)

	q_targets_dict = {}

	for i in range(len(question_list)):
		q_targets_dict[question_list[i]] = target_list[i]

	hypothesis_list = []
	reference_list = []

	save_filepath = os.path.join(save_dir, '{}.txt'.format(MODEL_DIR))

	for q in q_pred_ans_dict:
		hypothesis_list.append(q_pred_ans_dict[q])
		reference_list.append(q_targets_dict[q])

	em = get_em(hypothesis_list, reference_list)

	p, r, f = get_results(hypothesis_list, reference_list)

	with open(save_filepath, 'w') as fw:
		fw.write(', '.join([str(em), str(p), str(r), str(f)]))

	get_swms(hypothesis_list, reference_list)

	os.system('python3 smd.py input_swms.tsv glove s+wms {}'.format(save_filepath))

	os.system('rm input_swms.tsv')

	# exact match, rouge-L p, r, f, s+wms

