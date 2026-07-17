Hi Francois. Hope you are well.
For T5, the HuggingFace API doesn't ship a span-corruption collator, so I used DataCollatorForT5MLM from the HF olm-training repo. It builds sentinel tokens by counting down from len(tokenizer). ByT5's tokenizer reports a vocab of 384, but only the first 259 IDs were actually trained, as per the ByT5 paper, it reuses its own last 100 real byte IDs as sentinels rather than adding new ones.

My first ByT5 smoke run (batch size 4, grad accum 256, steps = 3) had a training loss of ~3175,  dividing by the accumulation steps gives ~12.4 per example and eval loss of 8.3. Much higher than the other models, so I wanted to be sure this wasn't a training bug on my end before going further.
I ran a step-0 forward-pass check (no training, just loss on fresh weights) to isolate it:
Original setup (sentinel base 384, span 3): loss 8.09
Sentinel base fixed to 259: loss 4.88
Also fixing span length to 20 bytes (the ByT5 paper's value, not T5's 3): loss 0.81 
Ruled out bf16 precision (same result in fp32) and bad data.
Here's the confusing part: running the same test on nguni-byt5 gives the opposite result. Sentinel base 384 + span 3 gives its lowest loss (4.45), while the ByT5 paper's own convention (base 259, span 20) is worse (5.67).
Do you still have access to the code you used to CPT nguni-byt5 from byt5? I want to check whether your training used the standard T5-style collator (sentinels at the top of the vocab, span length 3) rather than ByT5's own convention. That would explain why nguni-byt5 does best with those settings. I am unsure about this setup and would appreciate any insight.

Separate question, unrelated to the above: I've queued 7 pretraining jobs (24hrs each) with 3 batch sizes for T5 (20, 16, 8) and 2 each for the byt5 models (8, 4). Does that setup seem right to you?

Warm regards

Rakeen

Hi Rakeen



It will be easier to chat about this in a meeting. I'm not sure what the issue is and where sentinel tokens come in. The code I used for Nguni-ByT5 would have been based on the original AfriByT5 code, which is available here: https://github.com/masakhane-io/lafand-mt. If you still have questions after looking at this code, we can have a quick online meeting tomorrow if you want to.



Comparing absolute losses is not ideal - rather compare loss curves over an initial amount of training steps.



Yes, that sounds good. Although to keep things simple initially, I would test hyperparameters on one model only.



Regards,
Francois

#I have pasted the code from this repo here for your understanding claude:
requirement:
* PyTorch, tested with version 1.7 and 1.10. Any version >=1.7 should work.
* Transformers 4.10, install with “pip install transformers==4.10”
* datasets, install with “pip install datasets”
* install apex to use fp16, see https://github.com/NVIDIA/apex
* install fairscale to use distributed training, see https://github.com/facebookresearch/fairscale

preprocess data to the input format:
run preprocess.py
line 74, change the number 258 to the first token id when using mT5.

run bash distribute_train.sh to train

parameters explanation:
* tokenizer_name: checkpoint folder or a huggingface identifier
* model_name: checkpoint folder used for initialization or a huggingface identifier
* data_dir: dataset folder
* max_source_length: maximum source length in the training, longer will be truncated, default 128
* num_labels: number of label classes (2 for cPQA0/USB and 3 for GQA)
* metric_for_best_model: metrics used to select the model, default “accuracy”
* greater_is_better: if true, then select the model with the greatest metric and vice versa
* output_dir: path of the output checkpoint
* per_device_eval_batch_size: batch size for evaluation, default set as 512
* eval_steps: every number of steps to evaluate on the dev set, default set as 1% of whole training steps
* fp16: if true, use fp16, default set as true
* gradient_accumulation_steps: number of gradient accumulation steps, default set as 1
* save_steps: every number of steps to save the model, default set as 1% of whole training steps
* save_total_limit: maximum saved checkpoint, default set as 1
* per_device_train_batch_size: batch size for training, default set as 64
* num_train_epochs: total number of train epochs, default set as 3
* warmup_steps: warm up step, default set as 20% of the whole training steps
* learning_rate: learning rate, default set as 3e-5
* gpus= ... GPU ids used, default as 0,1,2,3,4,5,6,7
* nproc_per_node= .. number of GPUs used, default as 8
* ddp_find_unused_parameters: default false
* sharded_ddp: default zero_dp_3

the data_dir folder should have train.source/train.target, dev.source/dev.target files
//evaluate.py
from datasets import load_metric
import sys
import torch.nn as nn
from tqdm import tqdm
from util import *
from argparse import ArgumentParser
import numpy as np
from transformers import QuestionAnsweringPipeline
import pandas as pd
import pickle
import shutil
import os
import time
from scipy.special import softmax

def eval_outputs(of, reference, batch_size, metric):
    metric = load_metric(metric)
    outs, refs = open(of).readlines(), open(reference).readlines()
    compare_chunk = chunks(list(zip(outs, refs)), batch_size)
    for o in tqdm(list(compare_chunk)):
        os = [m[0] for m in o]
        rs = [m[1] for m in o]
        preds = [l.strip().split() for l in os]
        tgts = [[l.strip().split()] for l in rs]
        metric.add_batch(predictions=preds, references=tgts)
    scores = metric.compute()
    print(scores)

def gen_eval(model, tokenizer, dloader, output_dir, max_len, args, testtype):
    if args.compute_metric:
        metric = load_metric(args.metric)
    outputs = []
    model.config.force_bos_token_to_be_generated = False
    with torch.no_grad():
        for inputs in tqdm(list(dloader)):
            #print('inputs:',inputs)
            #sys.exit()
            inputs = {k: v.cuda() for k, v in inputs.items()}
            if args.decode == 'beam_search':
                preds = model.generate(inputs["input_ids"], num_beams=args.num_beams, num_return_sequences=args.num_samples,max_length=max_len, early_stopping=True, do_sample = False, decoder_start_token_id = model.config.bos_token_id)
            elif args.decode == 'nucleus':
                preds = model.generate(inputs["input_ids"], do_sample=True, top_p = 0.8, num_return_sequences=args.num_samples, max_length=max_len, early_stopping=True, decoder_start_token_id = model.config.bos_token_id)
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip() for g in preds]
            outputs.extend([l + "\n" for l in preds])
            if 'bleu' in args.metric:
                preds = [l.split() for l in preds]
            labels = inputs['labels']
            labels[labels < 0] = tokenizer.pad_token_id
            trgs = [[tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip().split()] for g in labels] if 'bleu' in args.metric else [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip() for g in labels]
            if args.compute_metric:
                metric.add_batch(predictions=preds, references=trgs)
    if args.compute_metric:
        scores = metric.compute()
        outputs.append(str(scores) + '\n')
    with open(Path(output_dir).joinpath('decode.' + testtype), "w") as f:
        f.writelines(outputs)

//main.py
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    ElectraTokenizerFast,
    Seq2SeqTrainingArguments,
    set_seed,
)
from dataclasses import dataclass, field
from transformers.models.electra.modeling_electra import ElectraClassificationHead
from transformers.trainer_utils import EvaluationStrategy
from typing import Optional
import sys
import os
from util import *
from pathlib import Path

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    task_type: str = field(
        metadata={"help": "Task type, can be either generation or classification"}
    )
    num_labels: str = field(
        metadata={"help": "Number of labels, used for sequence classification"}
    )
    mode: str = field(
        metadata={"help": "mode, can be either train, predict, hp_search or output_loss"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_encoder: bool = field(default=False, metadata={"help": "Whether tp freeze the encoder."})
    freeze_embeds: bool = field(default=False, metadata={"help": "Whether  to freeze the embeddings."})

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    test_type: Optional[str] = field(
        default="test", metadata={"help": "The type_path of the test file, test.seen, test.unseen etc."}
    )
    task: Optional[str] = field(
        default="summarization",
        metadata={"help": "Task name, summarization (or summarization_{dataset} for pegasus) or translation"},
    )
    max_source_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=64,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=64,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. "
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=300,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    n_train: Optional[int] = field(default=None, metadata={"help": "# training examples. None means use all."})
    n_val: Optional[int] = field(default=None, metadata={"help": "# validation examples. None means use all."})
    n_test: Optional[int] = field(default=None, metadata={"help": "# test examples. None means use all."})
    eval_beams: Optional[int] = field(default=None, metadata={"help": "# num_beams to use for evaluation."})
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )

@dataclass
class EvalArguments:
    """
    Arguments pertaining to the evaluation of the model.
    """

    decode: Optional[str] = field(
        default='beam_search', metadata={"help": "Decoding method used, take in value of beam_search, nucleus"}
    )
    metric: Optional[str] = field(
        default='bleu', metadata={"help": "The metric used to evaluate the model, takes in value of bleu, rouge, meteor etc"}
    )
    compute_metric: Optional[bool] = field(
        default=False, metadata={"help": "whether to compute metrics while generating the outputs, must be False if num_samples > 1"}
    )
    num_beams: Optional[int] = field(
        default=5, metadata={"help": "beam size used to decode"}
    )
    num_samples: Optional[int] = field(
        default=1, metadata={"help": "Number of decoded sequence for each input"}
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, EvalArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) > 1 and sys.argv[-1].endswith(".json"):
    # If the last argument ends with json then it's the path to a json file,
    # parse it to get our arguments.
        model_args, data_args, eval_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, eval_args, training_args = parser.parse_args_into_dataclasses()

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name,
        cache_dir=model_args.cache_dir,
    )
    if model_args.task_type == 'generation_id':
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name,
            config=config,
            cache_dir=model_args.cache_dir
        )
    
    # Get datasets
    train_dataset = (
        Seq2SeqDataset(
            tokenizer,
            type_path="train",
            task_type = model_args.task_type,
            mode = model_args.mode,
            data_dir=data_args.data_dir,
            n_obs=data_args.n_train,
            max_target_length=data_args.max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
        )
        if model_args.mode == 'train' or model_args.mode == 'hp_search'
        else None
    )
    eval_dataset = (
        Seq2SeqDataset(
            tokenizer,
            type_path="dev",
            task_type = model_args.task_type,
            mode = model_args.mode,
            data_dir=data_args.data_dir,
            n_obs=data_args.n_val,
            max_target_length=data_args.val_max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
        )
        if model_args.mode == 'train' and training_args.evaluation_strategy != EvaluationStrategy.NO
        else None
    )
    test_dataset = (
        Seq2SeqDataset(
            tokenizer,
            type_path=data_args.test_type,
            task_type = model_args.task_type,
            mode = model_args.mode,
            data_dir=data_args.data_dir,
            n_obs=data_args.n_test,
            max_target_length=data_args.test_max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
        )
    ) 

    # Initialize our Trainer

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if training_args.do_predict else eval_dataset,
        data_collator=Seq2SeqDataCollator(tokenizer, config.decoder_start_token_id,model_args.task_type, model_args.mode, data_args),
        tokenizer=tokenizer,
    )
   
    if model_args.mode == 'train':
        print('enter train')
        check_output_dir(training_args)#check if output_dir exists and raises error if it exists over_wirte=False

        set_seed(training_args.seed)#set training seed
        if model_args.freeze_embeds:
            freeze_embeds(model)
        if model_args.freeze_encoder:
            freeze_params(model.get_encoder())
            assert_all_frozen(model.get_encoder())
        print('start train')
        trainer.train()
        trainer.save_model(Path(training_args.output_dir).joinpath("best-epoch"))#save best epoch

    elif model_args.mode == 'predict':
        gen_eval(model, tokenizer, trainer.get_eval_dataloader(), training_args.output_dir, data_args.test_max_target_length, eval_args, 'test')
    elif model_args.mode == 'eval':
        print(trainer.evaluate(eval_dataset=test_dataset))

if __name__ == "__main__":
    main()
//process.py
from transformers import AutoTokenizer
import random
from tqdm import tqdm
import sys

def racha_detection(lista):
    # It returns a list of lists where each sub-list contains the consecutive tokens in the list
    rachas = []
    racha = []
    for i, element in enumerate(lista):
        if (i<len(lista)-1) and (lista[i+1] == element+1):
            racha.append(element)
        else:
            if len(racha)>0:
                rachas.append(racha + [element])          
            else:# (i!=len(lista)-1):
                rachas.append([element])
            racha = []
    return rachas

def masking(tokenized_sentence, rachas):
    # Function to mask a tokenized_sentence (token ids) following the rachas described in rachas
    # Only one sentinel_token per racha
    sent_token_id = 0
    enmascared = tokenized_sentence.copy()
    for racha in rachas:
        sent_token = f'<extra_id_{sent_token_id}>'
        sent_id = tokenizer.encode(sent_token)[0]
        for i, idx in enumerate(racha):
            if i==0:
                enmascared[idx] = sent_id
            else:
                enmascared[idx] = -100
        sent_token_id += 1
    
    enmascared = [t for t in enmascared if t!=-100] 

    return enmascared

def add_noise(sentence, tokenizer, percent=0.15):
    # Function that takes a sentence, tokenizer and a noise percentage and returns
    # the masked input_ids and masked target_ids accordling with the T5 paper and HuggingFace docs
    # To see the process working uncomment all the prints ;)
    tokenized_sentence = tokenizer.encode(sentence)
    #print('PRE-MASKED:')
    #print('INPUT: {}'.format(tokenizer.convert_ids_to_tokens(tokenized_sentence)))

    idxs_2_mask = sorted(random.sample(range(len(tokenized_sentence)), 
                                       int(len(tokenized_sentence)*percent)))
    rachas = racha_detection(idxs_2_mask)
    enmascared_input = masking(tokenized_sentence, rachas)
    #print('RACHAS INPUT: {}'.format(rachas))
    idxs_2_mask = [idx for idx in range(len(tokenized_sentence)) if idx not in idxs_2_mask]
    rachas = racha_detection(idxs_2_mask)
    enmascared_target = masking(tokenized_sentence, rachas)
    #print('RACHAS TARGET: {}'.format(rachas))

    #print('POST-MASKED:')
    #print('INPUT: {}'.format(tokenizer.convert_ids_to_tokens(enmascared_input)))
    #print('TARGET: {}'.format(tokenizer.convert_ids_to_tokens(enmascared_target)))

    return enmascared_input, enmascared_target

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")
    files = ['af',  'am',  'ar',  'en',  'fr',  'ha',  'ig',  'mg',  'ny',  'om',  'pcm',  'rw',  'sn',  'so',  'st',  'sw',  'xh',  'yo',  'zu']
    sources, targets = [], []
    for f in files:
        print(f)
        lines = open('Processed/' + f + '/train.'+f).readlines()
        for line in tqdm(lines):
            line = line.strip()
            source, target = add_noise(line, tokenizer)
            while(target[0]!=258):
                source, target = add_noise(line, tokenizer)
            sources.append(' '.join(list(map(str, source))).strip() + '\n')
            targets.append(' '.join(list(map(str, target))).strip() + '\n')
    with open('train.source' + '.' + str(i), 'w') as f:
        f.writelines(sources)
    with open('train.target' + '.' + str(i), 'w') as f:
        f.writelines(targets)
    #lines = open('Processed/af/eval.af').readlines()
    #for i in range(1000):
    #    s, t = add_noise(lines[i], tokenizer)
    #    sources.append(' '.join(list(map(str, s))).strip() + '\n')
    #    targets.append(' '.join(list(map(str, t))).strip() + '\n')
    #with open('train.source', 'w') as f:
    #    f.writelines(sources)
    #with open('train.target', 'w') as f:
    #    f.writelines(targets)
//util.py
import linecache
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, Sampler
from pathlib import Path
import os
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers import T5TokenizerFast, BartTokenizerFast
import numpy as np
from transformers.file_utils import cached_property
from typing import Callable, Dict, Iterable, List, Tuple, Union
import random
import math

class AbstractSeq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        task_type = "generation",
        n_obs=None,
        source_prefix="",
        target_prefix="",
        **dataset_kwargs
    ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.len_file = Path(data_dir).joinpath(type_path + ".len")
        if os.path.exists(self.len_file):
            self.src_lens = pickle_load(self.len_file)
            self.used_char_len = False
        else:
            self.src_lens = self.get_char_lens(self.src_file)
            self.used_char_len = True
        self.task_type = task_type
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.source_prefix = source_prefix
        self.target_prefix = target_prefix
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.dataset_kwargs = dataset_kwargs

    def __len__(self):
        return len(self.src_lens)

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @cached_property
    def tgt_lens(self):
        """Length in characters of target documents"""
        return self.get_char_lens(self.tgt_file)

    def make_sortish_sampler(self, batch_size, distributed=False, shuffle=True, **kwargs):
        if distributed:
            return DistributedSortishSampler(self, batch_size, shuffle=shuffle, **kwargs)
        else:
            return SortishSampler(self.src_lens, batch_size, shuffle=shuffle)

    def make_dynamic_sampler(self, max_tokens_per_batch=1024, **kwargs):
        assert FAIRSEQ_AVAILABLE, "Dynamic batch size requires `pip install fairseq`"
        assert not self.used_char_len, "You must call  python make_len_file.py before calling make_dynamic_sampler"
        sorted_indices = list(self.make_sortish_sampler(1024, shuffle=False))

        def num_tokens_in_example(i):
            return min(self.src_lens[i], self.max_target_length)

        # call fairseq cython function
        batch_sampler: List[List[int]] = batch_by_size(
            sorted_indices,
            num_tokens_fn=num_tokens_in_example,
            max_tokens=max_tokens_per_batch,
            required_batch_size_multiple=64,
        )
        shuffled_batches = [batch_sampler[i] for i in np.random.permutation(range(len(batch_sampler)))]
        # move the largest batch to the front to OOM quickly (uses an approximation for padding)
        approximate_toks_per_batch = [max(self.src_lens[i] for i in batch) * len(batch) for batch in shuffled_batches]
        largest_batch_idx = np.argmax(approximate_toks_per_batch)
        shuffled_batches[0], shuffled_batches[largest_batch_idx] = (
            shuffled_batches[largest_batch_idx],
            shuffled_batches[0],
        )
        return shuffled_batches

    def __getitem__(self, item):
        raise NotImplementedError("You must implement this")

    def collate_fn(self, batch):
        raise NotImplementedError("You must implement this")


class Seq2SeqDataset(AbstractSeq2SeqDataset):
    """A dataset that calls prepare_seq2seq_batch."""

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.source_prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = self.target_prefix + linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""
        if self.task_type == 'generation':
            batch_encoding: Dict[str, torch.Tensor] = self.tokenizer.prepare_seq2seq_batch(
                [x["src_texts"] for x in batch],
                tgt_texts=[x["tgt_texts"] for x in batch],
                max_length=self.max_source_length,
                max_target_length=self.max_target_length,
                padding="longest",
                return_tensors="pt",
                **self.dataset_kwargs,
            ).data
            batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])
            return batch_encoding
        else:
            batch = self.tokenizer([x["src_texts"] for x in batch], padding=True, max_length=self.max_source_length, truncation=True)
            batch["labels"] = torch.tensor([int(x["tgt_texts"]) for x in batch])
            return batch

class Seq2SeqDataCollator:
    def __init__(self, tokenizer, decoder_start_token_id, task_type, mode, data_args):
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.mode = mode
        self.decoder_start_token_id = decoder_start_token_id
        #if self.task_type == 'generation':
        self.pad_token_id = tokenizer.pad_token_id
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        self.data_args = data_args

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        if self.task_type == 'generation_id':
            sources = [x["src_texts"] for x in batch]
            targets = [x["tgt_texts"] for x in batch]
            input_ids = [list(map(int, s.split())) for s in sources]
            max_len = max([len(i) for i in input_ids])
            if max_len > self.data_args.max_source_length:
                max_len = self.data_args.max_source_length
            attention_mask = [[1]*max_len if len(i) > max_len else [1]*len(i) + [0]*(max_len-len(i)) for i in input_ids]
            input_ids = [i[:max_len] if len(i) > max_len else i + [self.pad_token_id]*(max_len-len(i)) for i in input_ids]
            labels = [list(map(int, s.split())) for s in targets]
            max_len = max([len(i) for i in labels])
            if max_len > self.data_args.max_target_length:
                max_len = self.data_args.max_target_length
            labels = [i[:max_len] if len(i) > max_len else i + [self.pad_token_id]*(max_len-len(i)) for i in labels]
            result_batch = {}
            result_batch["input_ids"] = torch.LongTensor(input_ids)
            result_batch["attention_mask"] = torch.LongTensor(attention_mask)
            result_batch["labels"] = torch.LongTensor(labels)
            return result_batch        

    def _encode(self, batch) -> Dict[str, torch.Tensor]:
        batch_encoding: Dict[str, torch.Tensor] = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.data_args.max_source_length,
            max_target_length=self.data_args.max_target_length,
            padding="longest",
            return_tensors="pt"
            ).data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])
        return batch_encoding


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."

    def __init__(self, data, batch_size, shuffle=True):
        self.data, self.bs, self.shuffle = data, batch_size, shuffle

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(sortish_sampler_indices(self.data, self.bs, shuffle=self.shuffle))


def sortish_sampler_indices(data: List, bs: int, shuffle=True) -> np.array:
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."
    if not shuffle:
        return np.argsort(np.array(data) * -1)

    def key_fn(i):
        return data[i]

    idxs = np.random.permutation(len(data))
    sz = bs * 50
    ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
    sort_idx = np.concatenate([sorted(s, key=key_fn, reverse=True) for s in ck_idx])
    sz = bs
    ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
    max_ck = np.argmax([key_fn(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
    ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
    sort_idx = np.concatenate(np.random.permutation(ck_idx[1:])) if len(ck_idx) > 1 else np.array([], dtype=np.int)
    sort_idx = np.concatenate((ck_idx[0], sort_idx))
    return sort_idx


class DistributedSortishSampler(Sampler):
    """Copied from torch DistributedSampler"""

    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, add_extra_examples=True, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        if add_extra_examples:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(dataset)
            self.num_samples = len(self.available_indices)
        self.batch_size = batch_size
        self.add_extra_examples = add_extra_examples
        self.shuffle = shuffle

    def __iter__(self) -> Iterable:
        g = torch.Generator()
        g.manual_seed(self.epoch)

        sortish_data = [self.dataset.src_lens[i] for i in self.available_indices]
        sortish_indices = sortish_sampler_indices(sortish_data, self.batch_size, shuffle=self.shuffle)
        indices = [self.available_indices[i] for i in sortish_indices]
        assert len(indices) == self.num_samples
        return iter(indices)

    @cached_property
    def available_indices(self) -> np.array:
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        available_indices = indices[self.rank : self.total_size : self.num_replicas]
        return available_indices

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)

def check_output_dir(args, expected_items=0):
    """
    Checks whether to bail out if output_dir already exists and has more than expected_items in it
    `args`: needs to have the following attributes of `args`:
      - output_dir
      - do_train
      - overwrite_output_dir
    `expected_items`: normally 0 (default) - i.e. empty dir, but in some cases a few files are expected (e.g. recovery from OOM)
    """
    if (
        os.path.exists(args.output_dir)
        and len(os.listdir(args.output_dir)) > expected_items
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and "
            f"has {len(os.listdir(args.output_dir))} items in it (expected {expected_items} items). "
            "Use --overwrite_output_dir to overcome."
        )

def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])

# Utilities for freezing parameters and checking whether they are frozen


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    model_type = model.config.model_type

    if model_type == "t5":
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)
    elif model_type == "fsmt":
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    else:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)

def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"

def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())

def any_requires_grad(model: nn.Module) -> bool:
    return any(grad_status(model))