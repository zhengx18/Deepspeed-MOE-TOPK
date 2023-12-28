# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Blendable dataset."""

import hashlib
import os
import time

import numpy as np
import torch

from megatron import print_rank_0
from megatron.core import mpu

from megatron import get_args
from collections import OrderedDict
import multiprocessing
import itertools

class BlendableDataset(torch.utils.data.Dataset):


    def __init__(self, datasets, weights, size, *,
                 data_cache_path=None):

        self.datasets = datasets
        num_datasets = len(datasets)
        assert num_datasets == len(weights)

        self.size = size

        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights

        # Build indicies.
        def _build_indices():
            start_time = time.time()
            assert num_datasets < 255
            dataset_index = np.zeros(self.size, dtype=np.uint8)
            dataset_sample_index = np.zeros(self.size, dtype=np.int64)

            from megatron.data import helpers
            helpers.build_blending_indices(dataset_index, dataset_sample_index,
                                           weights, num_datasets, self.size,
                                           torch.distributed.get_rank() == 0)
            print_rank_0('> elapsed time for building blendable dataset indices: '
                         '{:.2f} (sec)'.format(time.time() - start_time))
            return dataset_index, dataset_sample_index

        desc = "Blendable dataset\n\n"
        desc += "Datasets:\n"
        for dataset in datasets:
            desc += dataset.desc + "\n\n"
        desc += f"Weights: {weights}\n"
        desc += f"Size: {size}\n"
        self.desc = desc

        if data_cache_path:
            desc_hash = hashlib.md5(desc.encode('utf-8')).hexdigest()
            desc_path = os.path.join(data_cache_path, desc_hash + ".dsc")
            index_path = os.path.join(data_cache_path, desc_hash + "_index.npy")
            sample_index_path = os.path.join(data_cache_path, desc_hash + "_sample_index.npy")
            cache_hit = os.path.isfile(index_path) and os.path.isfile(sample_index_path)
            cache_success = True
            if torch.distributed.get_rank() == 0 and not cache_hit:
                print(' > WARNING: could not find index map files for blendable'
                      ' dataset, building indices on rank 0 ...', flush=True)
                dataset_index, dataset_sample_index = _build_indices()
                try:
                    os.makedirs(os.path.dirname(index_path), exist_ok=True)
                    with open(desc_path, 'wt') as fd:
                        fd.write(desc)
                        np.save(index_path, dataset_index, allow_pickle=True)
                        np.save(sample_index_path, dataset_sample_index,
                                allow_pickle=True)
                except OSError:
                    print(f'There was an error trying to create the data cache directory ({data_cache_path})')
                    print('or a file in it. This is set with the --data-cache-path argument. Please')
                    print('ensure you have write access to this directory or specify one that you do have')
                    print('write access to.')
                    cache_success = False


            counts = torch.cuda.LongTensor([cache_success])
            torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
            torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
            if counts[0].item() != (
                torch.distributed.get_world_size() //
                torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group()) //
                torch.distributed.get_world_size(group=mpu.get_sequence_parallel_group())):
                print_rank_0("Data index creation unsuccessful, exiting.")
                exit()

            # Load on all ranks.
            print_rank_0(f'> loading blendable dataset index: {index_path}')
            self.dataset_index = np.load(index_path, allow_pickle=True, mmap_mode='r')
            assert self.dataset_index.size == self.size

            print_rank_0(f'> loading blendable dataset sample index: {sample_index_path}')
            self.dataset_sample_index = np.load(sample_index_path, allow_pickle=True, mmap_mode='r')
            assert self.dataset_sample_index.size == self.size
        else:
            self.dataset_index, self.dataset_sample_index = _build_indices()


        # Check size
        _ = self.__getitem__(self.size - 1)
        try:
            _ = self.__getitem__(self.size)
            raise RuntimeError('BlendedDataset size is improperly bounded')
        except IndexError:
            pass
        print_rank_0('> size of blendable dataset: '
                     '{} samples'.format(self.size))


    def __len__(self):
        return self.size


    def __getitem__(self, idx):
        dataset_idx = self.dataset_index[idx]
        sample_idx = self.dataset_sample_index[idx]
        return {
            "dataset_idx" : dataset_idx,
            **self.datasets[dataset_idx][sample_idx],
        }


class BlendableWeightedSamplingDataset(torch.utils.data.IterableDataset):

    def __init__(self, datasets, weights):
        args = get_args()
        self.seed = args.seed
        self.global_bs = args.global_batch_size

        num_datasets = len(datasets)
        assert num_datasets == len(weights)
        self._num_datasets = num_datasets
        self._datasets = {dataset.name: dataset for dataset in datasets}
        self.sizes = [dataset.size for dataset in self._datasets.values()]

        self._dp_rank = mpu.get_data_parallel_rank()
        self._dp_world_size = mpu.get_data_parallel_world_size()
        self._num_workers = args.num_workers if args.num_workers >= 1 else 1
        assert self.global_bs % (self._num_workers * self._dp_world_size) == 0

        self.consume_sample_dict = multiprocessing.Manager().dict()
        self.consume_token_dict = multiprocessing.Manager().dict()
        self.past_epoch_dict = multiprocessing.Manager().dict()
        for dataset in datasets:
            # self.consume_sample_dict[dataset.name] = {'past_epoch': 0, 'c_sample': 0, 'c_token': 0}
            self.consume_sample_dict[dataset.name] = 0
            self.past_epoch_dict[dataset.name] = 0
            self.consume_token_dict[dataset.name] = 0

        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights
        self.weights = weights

        # Check size
        print_rank_0('> total number of tokens of blendable dataset: '
                     '{} tokens'.format(sum(self.sizes)))

    @property
    def size(self):
        return sum(self.sizes)

    def state_dict(self):
        saved_state_dict = OrderedDict()
        for name, dataset in self._datasets.items():
            saved_state_dict[name] = {}
            saved_state_dict[name]['past_epoch'] = self.past_epoch_dict[name]
            saved_state_dict[name]['c_sample'] = self.consume_sample_dict[name]
            saved_state_dict[name]['c_token'] = self.consume_token_dict[name]
            saved_state_dict[name]['num_tokens'] = dataset.size
            saved_state_dict['np_rng_state'] = np.random.get_state()
        return saved_state_dict

    def load_state_dict(self, state_dict):
        state = state_dict['np_rng_state']
        np.random.set_state(state)
        for name, dataset in self._datasets.items():
            if name in state_dict:
                self.past_epoch_dict[name] = state_dict[name]['past_epoch']
                self.consume_sample_dict[name] = state_dict[name]['c_sample']
                self.consume_token_dict[name] = state_dict[name]['c_token']
            else:
                print_rank_0(f"Appending new Dataset {name} with {dataset.size} tokens")

    def _sampling_index(self, local_dicts,worker_id):
        
        dataset_name = np.random.choice(list(self._datasets.keys()), size=self.global_bs, p=self.weights)
        sample_index = [0] * self.global_bs
        local_consume_sample_dict = local_dicts[worker_id]['consume_sample_dict']
        local_consume_token_dict = local_dicts[worker_id]['consume_token_dict']
        local_past_epoch_dict = local_dicts[worker_id]['past_epoch_dict']
        
        for i, d_name in enumerate(dataset_name):
            if local_consume_sample_dict[d_name] == self._datasets[d_name].num_sample:
                sample_index[i] = 0
                local_consume_sample_dict[d_name] = 0
                local_past_epoch_dict[d_name] += 1
            else:
                sample_index[i] = local_consume_sample_dict[d_name]
                local_consume_sample_dict[d_name] += 1
            local_consume_token_dict[d_name] += self._datasets[d_name]._chunk_size

        if worker_id == (self._num_workers - 1):
            self.consume_sample_dict.update(local_consume_sample_dict)
            self.consume_token_dict.update(local_consume_token_dict)
            self.past_epoch_dict.update(local_past_epoch_dict)
        
        return dataset_name, sample_index

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        dp = mpu.get_data_parallel_rank()
        is_last_worker = worker_id == (self._num_workers - 1)
        local_dicts = {}
        local_dicts[worker_id] = {
            'consume_sample_dict': dict(self.consume_sample_dict),
            'consume_token_dict': dict(self.consume_token_dict),
            'past_epoch_dict': dict(self.past_epoch_dict)
        }

        for _ in itertools.count():
            dataset_name, sample_index = self._sampling_index(local_dicts,worker_id)
            dp_offset = self._dp_rank
            dw_offset = dp_offset + worker_id * self._dp_world_size
            while True:
                try:
                    name, s_idx = dataset_name[dw_offset], sample_index[dw_offset]
                    yield self._datasets[name][s_idx]
                    dw_offset += self._num_workers * self._dp_world_size
                except:
                    break
