# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain Llama"""

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.llama_dataset import build_train_valid_test_datasets
from megatron.model import LlamaModel
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from collections import OrderedDict
import numpy as np
import os,sys

from mlflow.ksmlflow_runner import mlflowRunner, mlflowAsyncRunner, AsyncType
import os
from pathlib import Path
import json
import shutil

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building Llama model ...')
    model = LlamaModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    if tokens_.shape[-1] // args.seq_length == 2:
        tokens = tokens_[:, :args.seq_length].contiguous()
        labels = torch.cat([tokens[:, 1:], torch.ones_like(tokens[:, 0].unsqueeze(1))], dim=-1).contiguous()
    else:
        labels = tokens_[:, 1:].contiguous()
        tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    if tokens_.shape[-1] // args.seq_length == 2:
        loss_mask = tokens_[:, args.seq_length:].contiguous()
        loss_mask = torch.cat([loss_mask[:, 1:], torch.zeros_like(loss_mask[:, 0].unsqueeze(1))], dim=-1)

    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def no_wd_decay_cond(param_name, param):
    """
    Defines whether the parameter requires weight decay.
    if param do not require weight decay, return 'True', otherwise return 'False'
    """
    # do not regularize biases nor Norm parameters
    if param_name.endswith(".bias") or len(param.shape) == 1:
        no_wd = True
    else:
        no_wd = False
    return no_wd


def get_mlflow_data(runner=None, **kwargs):
    args = get_args()
    # log training info
    save_interval = args.save_interval
    global_step = kwargs['step']
    loss = kwargs['loss']
    grad_norm = kwargs['grad_norm']
    # num_tokens = kwargs['past_num_tokens']
    learning_rate = kwargs['learning_rate']

    dataloader_state_dict = kwargs['dataloader_state']
    # num_tokens = 0
    # num_tokens = 734546901196 # 切换数据集，保存进度 iter_175000
    # num_tokens = 734546901196-20972263552 # 跳过数据，205k-210k
    num_tokens = 734546901196-20972263552-8388608000 # 跳过数据，210k-212k
    for name, state in dataloader_state_dict.items():
        if name == 'np_rng_state':
            continue
        # num_tokens += state['c_token'] + state['past_epoch'] * state['num_tokens']
        num_tokens += state['c_token']
    log = {
        'loss': loss,
        'grad_norm': grad_norm,
        'learning_rate': learning_rate,
        'num_tokens': num_tokens,
    }

    # mlogger.log_mlflow_async(global_step, log, runner)
    runner.setLogMetric(log, global_step, int(num_tokens))

    simple_state_dict = OrderedDict()
    ds_cls_dict = OrderedDict()
    for name, state in dataloader_state_dict.items():
        if name == 'np_rng_state':
            continue
        total_num_tokens = state['num_tokens']
        simple_state_dict[name] = {'progress': None, 'visited_tokens': None, 'total': None}
        simple_state_dict[name]['total'] = total_num_tokens
        # simple_state_dict[name]['visited_tokens'] = state['c_token'] + state['past_epoch'] * state['num_tokens']
        simple_state_dict[name]['visited_tokens'] = state['c_token']
        ds_cls = name.split("__")[0]
        simple_state_dict[name]['cls'] = ds_cls
        simple_state_dict[name]['progress'] = simple_state_dict[name]['visited_tokens'] / simple_state_dict[name]['total']
        if ds_cls not in ds_cls_dict:
            ds_cls_dict[ds_cls] = {'progress': 0, 'visited_tokens': 0, 'total': 0}
        ds_cls_dict[ds_cls]['visited_tokens'] += simple_state_dict[name]['visited_tokens']
        ds_cls_dict[ds_cls]['total'] += simple_state_dict[name]['total']

    # ignore dataset sample ratio
    for ds_cls in ds_cls_dict:
        ds_cls_dict[ds_cls]['progress'] = ds_cls_dict[ds_cls]['visited_tokens'] / ds_cls_dict[ds_cls]['total']
    simple_state_dict.update(ds_cls_dict)

    all_dict = {'progress': 0, 'visited_tokens': 0, 'total': 0}
    all_dict['visited_tokens'] = sum(ds_cls_dict[k]['visited_tokens'] for k in ds_cls_dict)
    all_dict['total'] = sum(ds_cls_dict[k]['total'] for k in ds_cls_dict)
    all_dict['progress'] = all_dict['visited_tokens'] / all_dict['total']
    all_dict = {'all': all_dict}
    simple_state_dict.update(all_dict)

    # print("simple_state_dict: ", simple_state_dict)

    # log simple_state_dict
    skipped_iter = kwargs['skipped_iter']
    # global_step % 500 == 1 to check dataloader progress when restart.
    if global_step % save_interval == 0 or skipped_iter == 1 or global_step % 500 == 1:
        # print("simple_state_dict", simple_state_dict)
        runner.setStrLogMetric({"dataSetMetric": simple_state_dict}, global_step)


def train_valid_test_datasets_provider():
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for Llama ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_impl=args.data_impl,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path)
    print_rank_0("> finished creating Llama datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    save_path = sys.argv[2]
    exp_name = sys.argv[4]
    run_name = sys.argv[6]
    torch.set_printoptions(precision=8)
    log_params = {
        'model': 'llama',
        'arch': 'llama-1.3b',
        'framework': 'megatron',
        'dict': 128000,
        'tp': 2,
        'pp': 4,
        'extra1': '',
        'extra2': '',
        'position_emb': 'alibi'}
    
    mlflow_dict = {}
    mlflow_dict['experiment_name'] = exp_name
    mlflow_dict['run_name'] = run_name
    mlflow_dict['uri'] = "/nlp_group/mlflow_monitor/nlp/pretrain"
    mlflow_dict['save_dir'] = os.path.join(save_path, 'save')
    mlflow_dict['log_params'] = log_params
    pretrain(train_valid_test_datasets_provider,
                model_provider,
                ModelType.encoder_or_decoder,
                forward_step,
                no_wd_decay_cond,
                get_mlflow_data,
                mlflow_dict=mlflow_dict,
                args_defaults={'tokenizer_type': 'GPTSentencePieceTokenizer'})
