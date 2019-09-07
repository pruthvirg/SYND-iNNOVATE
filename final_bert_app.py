import os
import json
import torch
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert import BertForSequenceClassification
import logging
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import trange
from tqdm import tqdm
import requests
from flask import Flask, render_template, request
from flask import jsonify


OUTPUT_DIR = "./tmp/"
MODEL_FILE_NAME = "pytorch_model.bin"

labels = [i for i in range(6)]
target_names = list(set(labels))
label2idx = {label: idx for idx, label in enumerate(target_names)}

TARGET_NAME_PATH = os.path.join(os.path.expanduser("~"), "target_names.json")

target_names = list(set(labels))
with open(TARGET_NAME_PATH, "w") as o:
    json.dump(target_names, o)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BERT_MODEL = "bert-base-multilingual-uncased"
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels = 6)
model.to(device)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
MAX_SEQ_LENGTH=100

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        

def convert_examples_to_features(example_texts, example_labels, label2idx, max_seq_length, tokenizer, verbose=0):
    """Loads a data file into a list of `InputBatch`s."""
    
    features = []
    examples = zip(example_texts, example_labels)
    for (ex_index, (text, label)) in enumerate(examples):
        tokens = tokenizer.tokenize(text)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]
            
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)
            
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label2idx[label]
        if verbose and ex_index == 0:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label:" + str(label) + " id: " + str(label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features

def get_data_loader(features, max_seq_length, batch_size): 

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader

BATCH_SIZE = 4

def evaluate(model, dataloader):
    #from tqdm import tqdm_notebook as tqdm
    from tqdm import tqdm
    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []

    for step, batch in enumerate(tqdm(dataloader, desc="Evaluation iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)
        print(logits)
        #outputs = np.argmax(logits, axis=1)
        outputs = logits.argmax(1)
        label_ids = label_ids.to('cpu').numpy()
        
        predicted_labels += list(outputs)
        correct_labels += list(label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    
    correct_labels = np.array(correct_labels)
    predicted_labels = np.array(predicted_labels)
        
    return eval_loss, correct_labels, predicted_labels

with open(TARGET_NAME_PATH) as i:
    target_names = json.load(i)

output_model_file = os.path.join(OUTPUT_DIR, MODEL_FILE_NAME)
model_state_dict = torch.load(output_model_file)
model = BertForSequenceClassification.from_pretrained(BERT_MODEL, state_dict=model_state_dict, num_labels = len(target_names))
model.to(device)
model.eval()

def check_intent(string):
  own_texts = [string]
  own_labels = [0]
  own_features = convert_examples_to_features(own_texts,own_labels,label2idx, MAX_SEQ_LENGTH, tokenizer)
  own_dataloader = get_data_loader(own_features,MAX_SEQ_LENGTH, BATCH_SIZE)
  d, own_correct, own_predicted = evaluate(model, own_dataloader)
  if [i.item() for i in own_predicted][0]==0:
    return 2
  if [i.item() for i in own_predicted][0]==1:
    return 3
  if [i.item() for i in own_predicted][0]==2:
    return 4
  if [i.item() for i in own_predicted][0]==3:
    return 1
  if [i.item() for i in own_predicted][0]==4:
    return 0
  if [i.item() for i in own_predicted][0]==5:
    return 5

pos_list = ["yeah","yes","sure","yes please","ok","please do it","yep","okay","fine ","please proceed","do it","carry on ","yeah sure","surely","please do it","affirmative"]
neg_list = ["no","nope","sorry no","no sorry","stop","please stop","stop it","negetive","cancel it","go back","do not proceed","wait","return","return back"]
quit_list = ["quit","kindly go to main menu","proceed to main menu","main menu","quit all","go to the starting""exit","can you please quit","please go to the main menu","please exit","please proceed to main  menu"]

def check_yes_no_intent(text):
    if text in pos_list:
        return 'yes'
    elif text in neg_list:
        return 'no'
    elif text in neg_list:
        return 'quit'
    else:
        return check_intent(text)
    
app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/bert')
def bert_run():
    text = request.args.get('text')

    if text == None:
        return jsonify(0)
    else:
        intent = check_yes_no_intent(text)
        return jsonify(intent)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=80)