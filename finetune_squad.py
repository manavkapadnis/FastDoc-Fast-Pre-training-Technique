from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
import sys

MODEL_TYPE = sys.argv[1]
MODEL_NAME = sys.argv[2]

import json
with open('train-v2.0.json', 'r') as f:
    train_data = json.load(f)

train_data = [item for topic in train_data['data'] for item in topic['paragraphs'] ]
# print(train_data[0])
# train_data = train_data[:5000]

train_args = QuestionAnsweringArgs()

train_args.max_answer_length = 30    
train_args.learning_rate = 5e-5
train_args.num_train_epochs = 2
train_args.overwrite_output_dir= True
train_args.reprocess_input_data= False
train_args.train_batch_size= 48
train_args.fp16= False
#       'wandb_project': "simpletransformers"


model = QuestionAnsweringModel(MODEL_TYPE, MODEL_NAME, args=train_args)
model.train_model(train_data)

import os

from transformers import BertTokenizer, RobertaTokenizer, AutoModelForQuestionAnswering

def get_model_path():
    parent_path = './outputs'
    for item in os.listdir(parent_path):
        if '-epoch-2' in item:
            return os.path.join(parent_path, item)

model_path = get_model_path()

print()
print(model_path)
print()

if 'roberta' in MODEL_TYPE:
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
else:
    tokenizer = BertTokenizer.from_pretrained(model_path)

model = AutoModelForQuestionAnswering.from_pretrained(model_path)

save_name = MODEL_NAME.split('/')[-1] + '_squad2.0'

os.system('mkdir -p squad_models')

save_path = os.path.join('squad_models', save_name)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

os.system('rm -r outputs'.format(save_name))
os.system('rm -r runs'.format(save_name))