## 20231228

# bug1: 训练3000+ iteration后,报错OutOfMemoryError:
    CUDA out of memory. Tried to allocate 1.54 GiB. GPU 4 has a total capacty of 79.10 GiB of which 1.43 GiB is free. Process 6123 has 77.66 GiB memory in use. Of the allocated memory 73.89 GiB is allocated by PyTorch, and 1.97 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF: return super().apply(*args, **kwargs)  # type: ignore[misc]
    尝试减小expert_num, 16-> 8

# bug2: fp16导致的OVERFLOW问题 
    [2023-12-28 07:44:59,843] [INFO] [loss_scaler.py:190:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 131072, but hysteresis is 2. Reducing hysteresis to 1
    修改为bf16继续


