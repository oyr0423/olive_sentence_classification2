import transformers 

from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch
# from pytorch_pretrained_bert import BertTokenizer

DEVICE = "cpu"
MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 10
BERT_PATH = "../input/bert_base_uncased/"
MODEL_PATH = "model.bin"
# MODEL_PATH = "model.py"
# MODEL_PATH = "../input/bert_base_uncased/pytorch_model.bin"
TRAINING_FILE = "../input/imdb.csv"
# TOKENIZER = BertTokenizer.from_pretrained(BERT_PATH, do_lower_case = True)
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')