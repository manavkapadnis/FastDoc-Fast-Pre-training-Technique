import os
import sys
import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np
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


data_folder = '<PATH TO DATA FOLDER>'

train_filepath = os.path.join(data_folder, 'train.csv') 
valid_filepath = os.path.join(data_folder, 'valid.csv')
test_filepath = os.path.join(data_folder, 'test.csv')

def get_df(filepath):
    df = pd.read_csv(filepath, usecols = ['question', 'text', 'label'])
    df = df.rename(columns={'question': 'text_a', 'text': 'text_b', 'label': 'labels'}).dropna().reset_index(drop = True)
    print(df.columns)
    return df

df_train = get_df(train_filepath)
df_valid = get_df(valid_filepath)
df_test = get_df(test_filepath)

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
steps_per_epoch = int(np.ceil(len(df_train) / float(TRAIN_BATCH_SIZE)))

evaluate_during_training_steps = int(steps_per_epoch/2)
save_steps = evaluate_during_training_steps
logging_steps = int(steps_per_epoch/5)

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
'''

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

MODEL_NAME = sys.argv[1]
HF_MODEL_PARAMETER = sys.argv[2]
NUM_LABELS = 2

model = ClassificationModel(MODEL_NAME, HF_MODEL_PARAMETER, num_labels=NUM_LABELS, args=train_args)

out_train = model.train_model(train_df = df_train, eval_df = df_valid)

print()
print('Loading best model ....')

# train_args['use_cached_eval_features'] = False
# train_args['overwrite_output_dir'] = False

if 'best_model' in os.listdir('outputs'):
    model = ClassificationModel(
        MODEL_NAME, "outputs/best_model", num_labels=NUM_LABELS, args=train_args
    )
else:
    model = ClassificationModel(
        MODEL_NAME, "outputs", num_labels=NUM_LABELS, args=train_args
    )

test_result, test_model_outputs, test_wrong_predictions = model.eval_model(df_test)

print(test_model_outputs[:10])


# report = classification_report(df_test['labels'].tolist(), np.argmax(test_model_outputs, axis = 1).tolist(), output_dict=True)
# df = pd.DataFrame(report).transpose()

# df.to_csv('similarity_classification_report.csv')

def get_metrics(test_model_outputs, df_test):
    targets = df_test['labels']
    class_1_outputs = test_model_outputs[:, 1]
    class_1_outputs = [1 if item > 0.5 else 0 for item in class_1_outputs]
    q_list = df_test['text_a'].tolist()

    q_target_dict = {}
    q_output_dict = {}
    for i, q in enumerate(q_list):
        if q not in q_target_dict:
            q_target_dict[q] = []
            q_output_dict[q] = []
        q_target_dict[q].append(targets[i])
        q_output_dict[q].append(class_1_outputs[i])


    num_questions = len(q_target_dict)

    avg_prec = 0
    avg_rec = 0
    avg_f1 = 0

    for q in q_target_dict:
        prec, rec, f1, _ = precision_recall_fscore_support(q_target_dict[q], q_output_dict[q], average='macro')
        avg_prec += prec
        avg_rec += rec
        avg_f1 += f1

    avg_prec = avg_prec/num_questions
    avg_rec = avg_rec/num_questions
    avg_f1 = avg_f1/num_questions

    return avg_prec, avg_rec, avg_f1, num_questions

with open('results.txt', 'w') as f:
    avg_prec, avg_rec, avg_f1, num_questions = get_metrics(test_model_outputs, df_test)
    f.write('For {} test QA pairs -\n'.format(num_questions))
    f.write('Avg. Precision: {}\n'.format(np.round(avg_prec, 4)))
    f.write('Avg. Recall: {}\n'.format(np.round(avg_rec, 4)))
    f.write('Avg. F1: {}\n'.format(np.round(avg_f1, 4)))

'''
with open('results.txt', 'w') as f:
    for K in [1, 3, 5]:
        perc, num_correct, num_questions = get_hits(test_model_outputs, df_test, K = K)
        f.write('For K = {}:'.format(K))
        f.write('\n')
        f.write('{}%, {} are correct out of {} QA pairs in test set'.format(perc, num_correct, num_questions))        
        f.write('\n\n')
'''