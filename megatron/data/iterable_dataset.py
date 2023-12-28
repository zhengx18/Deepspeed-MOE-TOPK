# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# copied from fairseq/fairseq/data/indexed_dataset.py
# Removed IndexedRawTextDataset since it relied on Fairseq dictionary
# other slight modifications to remove fairseq dependencies
# Added document index to index file and made it accessible.
#    An empty sentence no longer separates documents.

from functools import lru_cache
import os
import shutil
import struct
from itertools import accumulate

import numpy as np
import torch
from megatron import print_rank_0
from megatron import get_args
import multiprocessing
import itertools


def get_available_dataset_impl():
    return ['lazy', 'cached', 'mmap']


def make_dataset(path, impl, chunk_size, batch_size_per_iter, data_parallel_rank, data_parallel_size,
                 skip_warmup=False):
    if not MMapDataset.exists(path):
        print(f"Dataset does not exist: {path}")
        print("Path should be a basename that both .meta and .bin can be appended to get full filenames.")
        return None
    if impl == 'mmap':
        return MMapDataset(path, chunk_size, batch_size_per_iter, data_parallel_rank, data_parallel_size, skip_warmup)
    print(f"Unknown dataset implementation: {impl}")
    return None


def dataset_exists(path, impl):
    if impl == 'mmap':
        return MMapDataset.exists(path)


def data_file_path(prefix_path):
    return prefix_path + '.bin'


def loss_mask_file_path(prefix_path):
    return prefix_path + '_loss_mask.bin'


def meta_file_path(prefix_path):
    return prefix_path + '.bin.meta'


def _warmup_mmap_file(path):
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass


class MMapDataset(torch.utils.data.Dataset):

    def __init__(self, path, chunk_size, batch_size_per_iter,
                 data_parallel_rank, data_parallel_size, skip_warmup=False):
        super().__init__()
        args = get_args()

        # bin dataset attribute
        with open(meta_file_path(path)) as r:
            dtype = r.read()
        self._dtype = getattr(np, dtype)
        self._path = path
        self.name = self._path.split('/')[-1].rsplit('.', 1)[0]

        self._chunk_size = chunk_size
        self._v_chunk_size = chunk_size + 1

        self._itemsize = self._dtype().itemsize
        self._chunk_bytes = self._chunk_size * self._itemsize
        self._v_chunk_bytes = self._v_chunk_size * self._itemsize
        self._total_bytes = os.path.getsize(data_file_path(path))
        assert self._total_bytes % self._itemsize == 0

        # dataloader attribute
        self._prefetch_factor = args.prefetch_factor

        # init binmemory dataset
        self._bin_buffer = None
        self._do_init(skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state, skip_warmup=True)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap

    def _do_init(self, skip_warmup):
        if not skip_warmup:
            print_rank_0("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self._path))
        print_rank_0("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode='r', order='C')
        print_rank_0("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)
        if self.is_qa_dataset:
            self._loss_mask_bin_buffer_mmap = np.memmap(loss_mask_file_path(self._path), mode='r', order='C')
            self._loss_mask_bin_buffer = memoryview(self._loss_mask_bin_buffer_mmap)

    @property
    def size(self):
        return self._total_bytes // self._itemsize

    @property
    def num_sample(self):
        return self._total_bytes // self._chunk_bytes

    # @lru_cache(maxsize=8)
    def get(self, s_idx):
        if self.is_qa_dataset:
            np_array = np.frombuffer(self._bin_buffer, dtype=self._dtype,
                                     count=self._chunk_size, offset=s_idx * self._chunk_bytes)
            loss_mask = np.frombuffer(self._loss_mask_bin_buffer, dtype=self._dtype,
                                      count=self._chunk_size, offset=s_idx * self._chunk_bytes)
            return np_array, loss_mask
        else:
            np_array = np.frombuffer(self._bin_buffer, dtype=self._dtype,
                                     count=self._v_chunk_size, offset=s_idx * self._chunk_bytes)
            return np_array

    def __getitem__(self, s_idx):
        sample = self.get(s_idx)
        if self.is_qa_dataset:
            sample = np.concatenate(sample, axis=-1, dtype=np.int64)
            return {'text': sample}
        else:
            sample = np.array(sample, dtype=np.int64)
            return {'text': sample}

    @property
    def is_qa_dataset(self):
        return os.path.exists(loss_mask_file_path(self._path))

    @staticmethod
    def exists(path):
        return (
                os.path.exists(meta_file_path(path)) and os.path.exists(data_file_path(path))
        )
