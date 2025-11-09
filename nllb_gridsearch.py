from transformers import (MBartForConditionalGeneration, MBart50TokenizerFast,
						M2M100ForConditionalGeneration, M2M100Tokenizer,
						AutoTokenizer, AutoModelForSeq2SeqLM,
						AutoModelForCausalLM, MT5ForConditionalGeneration,
						Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer,EarlyStoppingCallback)
from peft import LoraConfig, get_peft_model
from datasets import DatasetDict
from evaluate import load
import numpy as np
import argparse
import torch
import sys

MODEL = None
METRIC = None
TOKENIZER = None
TER_METRIC = None

class Europarl(torch.utils.data.Dataset):
	def __init__(self,source,target,tok,prefix=''):
		self.src = []
		with open(source,'r') as file:
			self.src = [l for l in file]
		self.src = [prefix + l for l in self.src]
		self.tgt = []
		with open(target,'r') as file:
			self.tgt = [l for l in file]
		self.tgt = [l for l in self.tgt]
		self.tok = tok
    
	def __len__(self):
		return len(self.src)
	
	def __getitem__(self,idx):
		return self.tok(self.src[idx], text_target=self.tgt[idx], max_length=128, truncation=True)

def get_device():
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	return device


def load_model(model_name, _dev=None):
	if model_name == 'nllb':
		_mdl = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
	else:
		print('Model not implemented: {0}'.format(model_name))
		sys.exit(1)
	return _mdl


def load_tokenizer(args):
	if args.model_name == 'mbart':
		_tok = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
	elif args.model_name == 'm2m':
		_tok = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
	elif args.model_name == 'flant5':
		_tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
	elif args.model_name == 'mt5':
		_tok = AutoTokenizer.from_pretrained("google/mt5-small")
	elif args.model_name == 'llama3':
		_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
	elif args.model_name == 'nllb':
		_tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
	else:
		print('Model not implemented: {0}'.format(args.model_name))
		sys.exit(1)
	_tok.src_lang = args.source_code
	_tok.tgt_lang = args.target_code
	return _tok


def load_text(file_path):
	with open(file_path) as file:
		data = file.read().splitlines()
	return data


def preprocess_dataset(dataset):
	inputs =  [translation['src'] for translation in dataset['translation']]
	targets = [translation['tgt'] for translation in dataset['translation']]
	model_inputs = TOKENIZER(inputs, text_target=targets, max_length=128, truncation=True)

	return model_inputs


def gen(shards):
	src_data = load_text(shards[0])
	tgt_data = load_text(shards[1])

	for src_line, tgt_line in zip(src_data, tgt_data):
		yield {"translation": {'src': src_line, 'tgt': tgt_line}}


def load_datasets(args):
	if 't5' in args.model_name:
		extend = {'en':'English','fr':'French','de':'German','es':'Spanish'}
		prefix = f'translate from {extend[args.source]} to {extend[args.target]}: '
	else:
		prefix = ''

	tr_shards = [f"{args.folder}/tr.{args.source}", f"{args.folder}/tr.{args.target}"]
	dev_shards = [f"{args.folder}/dev.{args.source}", f"{args.folder}/dev.{args.target}"]
	test_shards = [f"{args.folder}/test.{args.source}", f"{args.folder}/test.{args.target}"]

	training = Europarl(tr_shards[0], tr_shards[1], TOKENIZER, prefix=prefix)
	development = Europarl(dev_shards[0], dev_shards[1], TOKENIZER)
	test = Europarl(test_shards[0], test_shards[1], TOKENIZER)

	import torch.utils.data

	dataset = DatasetDict({
		"train": torch.utils.data.Subset(training, range(len(training))),
		"dev": torch.utils.data.Subset(development, range(len(development))),
		"test": torch.utils.data.Subset(test, range(len(test))),
	})

	return dataset


def postprocess_text(preds, labels):
	preds  = [pred.strip() for pred in preds]
	labels = [[label.strip()] for label in labels]

	return preds, labels

def compute_metrics(eval_preds):
	preds, labels = eval_preds
	if isinstance(preds, tuple):
		preds = preds[0]
	decoded_preds = TOKENIZER.batch_decode(preds, skip_special_tokens=True)

	labels = np.where(labels != -100, labels, TOKENIZER.pad_token_id)
	decoded_labels = TOKENIZER.batch_decode(labels, skip_special_tokens=True)

	decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

	bleu_result = METRIC.compute(predictions=decoded_preds, references=decoded_labels)
	ter_result = TER_METRIC.compute(predictions=decoded_preds, references=decoded_labels)

	result = {
		"bleu": bleu_result["score"],
		"ter": ter_result["score"]
	}

	prediction_lens = [np.count_nonzero(pred != TOKENIZER.pad_token_id) for pred in preds]
	result["gen_len"] = np.mean(prediction_lens)
	result = {k: round(v,4) for k, v in result.items()}
	return result


def check_language_code(code):
	if code=='ar':			# Arabic
		return 'ar_AR'
	elif code == 'en':		# English
		return 'en_XX'
	elif code == 'es':		# Spanish
		return 'es_XX'
	elif code == 'fr':		# French
		return 'fr_XX'
	else:
		print('Code not implemented')
		sys.exit()

def check_parameters(args):
	args.source_code = check_language_code(args.source) if args.model_name == 'mbart' else args.source
	args.target_code = check_language_code(args.target) if args.model_name == 'mbart' else args.target
	return args

def read_parameters():
	parser = argparse.ArgumentParser()
	parser.add_argument("-src", "--source", default="en", help="Source Language")
	parser.add_argument("-trg", "--target", default="es", help="Target Language")
	parser.add_argument("-dir", "--folder", default="./dataset2", help="Folder where is the dataset")
	parser.add_argument('-model','--model_name',default='nllb',choices=['mbart','m2m','flant5','mt5','llama3','nllb'],help='Model to train')
	parser.add_argument('-lora','--lora',action='store_true',help='Whether to use LowRank or not')
	parser.add_argument("-e","--epochs",type=int,default=3,help="Number of epochs")
	parser.add_argument('-bs','--batch_size',type=int,default=32,help='Batch size')

	args = parser.parse_args()
	return args

import itertools
import json
def main():
	global TOKENIZER, METRIC, TER_METRIC, MODEL

	args = read_parameters()
	args = check_parameters(args)
	print(args)

	device = get_device()
	METRIC = load("sacrebleu")
	TER_METRIC = load("ter")
	TOKENIZER = load_tokenizer(args)

	MODEL_BASE = lambda: load_model(args.model_name, device)

	if args.lora:
		lora_config = LoraConfig(
			r=16,
			lora_alpha=16,
			lora_dropout=0.1,
			target_modules='all-linear'
		)

	dataset_full = load_datasets(args)

	from torch.utils.data import Subset
	tr_len = len(dataset_full["train"])
	subset_size = int(0.2 * tr_len)
	subset_dataset = DatasetDict({
		"train": Subset(dataset_full["train"], list(range(subset_size))),
		"dev": Subset(dataset_full["dev"], list(range(int(1 * len(dataset_full["dev"]))))), 
		"test": Subset(dataset_full["test"], list(range(int(1 * len(dataset_full["test"]))))), 
	})

	fp16 = not 't5' in args.model_name

	grid = {
		"learning_rate": [2e-5, 5e-5],
		"batch_size": [8, 16],
		"weight_decay": [0.01, 0.1]
	}
	all_combinations = list(itertools.product(*grid.values()))

	best_score = -1
	best_config = None

	for i, (lr, bs, wd) in enumerate(all_combinations):
		print(f"\n Ejecutando combinación {i+1}/{len(all_combinations)}: lr={lr}, bs={bs}, wd={wd}")
		
		MODEL = MODEL_BASE()
		if args.lora:
			MODEL = get_peft_model(MODEL, lora_config)

		training_args = Seq2SeqTrainingArguments(
			output_dir=f"tmp_{i}",
			save_strategy="no",
			eval_strategy='epoch',
			per_device_train_batch_size=bs,
			per_device_eval_batch_size=bs,
			learning_rate=lr,
			weight_decay=wd,
			num_train_epochs=3, 
			predict_with_generate=True,
			fp16=fp16,
			logging_strategy="no"
		)

		data_collator = DataCollatorForSeq2Seq(TOKENIZER, model=MODEL)

		trainer = Seq2SeqTrainer(
			model=MODEL,
			args=training_args,
			train_dataset=subset_dataset['train'],
			eval_dataset=subset_dataset['dev'],
			data_collator=data_collator,
			tokenizer=TOKENIZER,
			compute_metrics=compute_metrics,
		)

		trainer.train()
		results = trainer.evaluate()
		score = results.get("eval_bleu", 0)

		print(f" BLEU: {score:.2f}")
		if score > best_score:
			best_score = score
			best_config = {"learning_rate": lr, "batch_size": bs, "weight_decay": wd}

	with open(f"best_hyperparams_{args.source}_{args.target}.json", "w") as f:
		json.dump(best_config, f, indent=2)

	print("\n Mejor configuración:")
	print(json.dumps(best_config, indent=2))


if __name__ == '__main__':
	main()
