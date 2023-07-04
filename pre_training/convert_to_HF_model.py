import torch
from models_ import *
from transformers import AutoTokenizer, AutoModel, BertTokenizer
import sys
import torch.nn as nn
import os
import json

model_type = sys.argv[1]
model_path = sys.argv[2]

if 'rule_based' in model_path:
	save_folder = 'HF_models_rule_based'
else:
	save_folder = 'HF_models'

HF_name = 'bert-base-uncased'

if 'roberta' in model_path:
	save_folder += '_roberta'
	HF_name = 'roberta-base'

if os.path.exists(save_folder) == False:
	os.mkdir(save_folder)

if model_type == 'longformer':
	model = nn.DataParallel(LongformerSiameseNetwork())
	tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')

elif model_type=='bert':
	model = nn.DataParallel(BertSiameseNetwork())
	tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
elif (model_type == 'hier') or (model_type == 'onlyclassfn') or (model_type == 'onlyclassfntwostage') or (model_type == 'triplethier') or (model_type == 'twostagesiamesehier') or (model_type == 'twostagetriplethier') or (model_type == 'twostagequadruplethier') or (model_type == 'quadruplethier'):
	taxonomy_path = './taxonomy_hierarchy'
	with open(os.path.join(taxonomy_path, 'taxonomy_hier_classes_labels.json'), 'r') as f:
		taxonomy_dict = json.load(f)	
	num_topics_each_level = []

	for level in taxonomy_dict:
		num_topics_each_level.append(len(taxonomy_dict[level]))     # including 'None'
	if model_type == 'hier':	
		model = nn.DataParallel(HierSiameseNetwork(num_topics_each_level))
	if model_type == 'triplethier':	
		model = nn.DataParallel(HierTripletNetwork(num_topics_each_level, HF_name))		
	elif model_type == 'onlyclassfn':
		model = nn.DataParallel(OnlyClassfnNetwork(num_topics_each_level, HF_name))	
	elif model_type == 'onlyclassfntwostage':
		if 'roberta' in model_path:
			model = nn.DataParallel(OnlyClassfnTwoStage(num_topics_each_level, 'roberta-base'))
		else:
			model = nn.DataParallel(OnlyClassfnTwoStage(num_topics_each_level))
	elif model_type == 'twostagesiamesehier':
		model = nn.DataParallel(HierTwoStageSiamese(num_topics_each_level))
	elif model_type == 'twostagetriplethier':
		if 'roberta' in model_path:
			model = nn.DataParallel(HierTwoStageTriplet(num_topics_each_level, 'distilroberta-base'))
		else:
			model = nn.DataParallel(HierTwoStageTriplet(num_topics_each_level))
	elif model_type == 'twostagequadruplethier':
		if 'roberta' in model_path:
			model = nn.DataParallel(HierTwoStageQuadruplet(num_topics_each_level, 'distilroberta-base'))
		else:
			model = nn.DataParallel(HierTwoStageQuadruplet(num_topics_each_level))		
	elif model_type == 'quadruplethier':
		model = nn.DataParallel(HierQuadrupletNetwork(num_topics_each_level, HF_name))
	if (model_type == 'twostagesiamesehier') or (model_type == 'twostagetriplethier') or (model_type == 'twostagequadruplethier'):
		if 'roberta' in model_path:
			tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
		else:
			tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_6L_768D')
	elif 'roberta' in model_path:
		tokenizer = AutoTokenizer.from_pretrained('roberta-base')
	else:
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')	
	
elif model_type == 'triplet':
	model = nn.DataParallel(TripletNetwork(HF_name))
	if 'roberta' in model_path:
		tokenizer = AutoTokenizer.from_pretrained('roberta-base')
	else:
		tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

elif model_type == 'quadruplet':
	model = nn.DataParallel(QuadrupletNetwork(HF_name))
	if 'roberta' in model_path:
		tokenizer = AutoTokenizer.from_pretrained('roberta-base')
	else:
		tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

elif model_type == 'twostagetriplet':
	
	if 'roberta' in model_path:
		model = nn.DataParallel(TwoStageTriplet('distilroberta-base'))
		tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
	else:
		model = nn.DataParallel(TwoStageTriplet())
		tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_6L_768D')

elif model_type == 'twostagequadruplet':
	model = nn.DataParallel(TwoStageQuadruplet())
	if 'roberta' in model_path:
		model = nn.DataParallel(TwoStageQuadruplet('distilroberta-base'))
		tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
	else:
		model = nn.DataParallel(TwoStageQuadruplet())
		tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_6L_768D')

elif model_type=='twostage':
	model = nn.DataParallel(twostagesiamese())
	if 'roberta' in model_path:
		tokenizer = AutoTokenizer.from_pretrained('roberta-base')
	else:
		tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')		

elif model_type == 'siameselarge':
	model = nn.DataParallel(SiameseLargeNetwork())
	tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')		
elif model_type == 'siameselargesquad':
        model = nn.DataParallel(SiameseLargeNetwork())
        tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
# elif model_type == 'onlyclassfn':
# 	model = nn.DataParallel(OnlyClassfnNetwork())
# 	tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')	

save_path = os.path.join(save_folder, model_path.split('/')[-1].replace('.pt', ''))

model.load_state_dict(torch.load(model_path)) # loaded state dict of model

for child in model.children():
	child_model = child

if 'twostage' in model_type:
	child_model.model_layer.encoder.save_pretrained(save_path)
else:
	child_model.model_layer.save_pretrained(save_path)

print(tokenizer)
tokenizer.save_pretrained(save_path)
