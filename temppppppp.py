import os
import sys
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple

# from deepspeed.utils import groups
# from .mappings import drop_tokens, gather_tokens

uniform_map: Dict[torch.device, Callable] = {}
gumbel_map: Dict[torch.device, Callable] = {}
exp_selection_uniform_map: Dict[torch.device, Callable] = {}

# 用于在指定的设备上生成具有特定形状的 Gumbel 分布的样本,保存在字典里，避免重复生成
def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)

def test_input(input):
    model_dim, num_experts = 768, 16
    wg = torch.nn.Linear(model_dim, num_experts, bias=False).float() # torch.Size([16, 768])

    if wg.weight.dtype != torch.float32:
        wg = wg.float()
    return wg(input)

def _top_idx(source, k):
    return torch.topk(source, k=k, dim=0)[1]

# @torch.jit.script
def _capacity(gates: Tensor, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    return capacity

def _one_hot_to_float(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()

def top1gating(logits: Tensor,
               capacity_factor: float,
               min_capacity: int,
               used_token: Tensor = None,
               noisy_gate_policy: Optional[str] = None,
               drop_tokens: bool = True,
               use_rts: bool = True,
               use_tutel: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top1Gating on logits."""
    if noisy_gate_policy == 'RSample':
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1) # logits-> [num_tokens, num_experts], gates-> [num_tokens, num_experts]

    capacity = _capacity(gates, torch.tensor(capacity_factor), torch.tensor(min_capacity)) # int, max(min_capacity, num_tokens/num_experts*capacity_factor)

    # Create a mask for 1st's expert per token
    # noisy gating
    indices1_s = torch.argmax(logits_w_noise if noisy_gate_policy == 'RSample' else gates, dim=1) # [tokens_num], 元素是expert_index
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts) # [tokens_nums, experts_num]

    # mask only used tokens
    if used_token is not None:
        mask1 = einsum("s,se->se", used_token, mask1)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu') # [num_experts], count each expert tokens, sum(exp_counts) == gates.shape[0] == total tokens

    # if we don't want to drop any tokens
    if not drop_tokens:
        new_capacity = torch.max(exp_counts).to(logits.device)
        dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=dist.get_world_group())
        capacity = new_capacity

    # Compute l_aux
    me = torch.mean(gates, dim=0)  # torch.Size([num_experts]), 
    ce = torch.mean(mask1.float(), dim=0) # torch.Size([num_experts])
    l_aux = torch.sum(me * ce) * num_experts

    # Random Token Selection
    if use_rts:
        uniform = exp_selection_uniform_map.get(logits.device)
        if uniform is None:
            uniform = torch.distributions.uniform.Uniform(low=torch.tensor(0.0, device=logits.device),
                                                          high=torch.tensor(1.0, device=logits.device)).rsample
            exp_selection_uniform_map[logits.device] = uniform

        mask1_rand = mask1 * uniform(mask1.shape)
    else:
        mask1_rand = mask1

    assert logits.shape[
        0] >= min_capacity, "No. of tokens (batch-size) should be greater than min_capacity. Either set min_capacity to 0 or increase your batch size."

    top_idx = _top_idx(mask1_rand, capacity) # shape: [capacity, num_experts]

    new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1) # num_tokens, num_experts
    mask1 = new_mask1

    if use_tutel:
        # Tutel doesn't support index values masked with zero
        # so we need to replace masked indices with -1
        indices_mask = mask1.sum(dim=1) * num_experts - 1
        indices1_s = torch.min(indices1_s, indices_mask)

    # Compute locations in capacity buffer
    if use_tutel:
        locations1 = tutel_moe.fast_cumsum_sub_one(mask1)
    else:
        locations1 = torch.cumsum(mask1, dim=0) - 1 # [token_num, experts_num] 统计列中累计和，即1的所有元素数量

    if use_tutel:
        gates1_s = (gates * mask1).sum(dim=1)
        locations1_s = torch.sum(locations1 * mask1, dim=1)
        return l_aux, capacity, num_experts, [
            indices1_s,
        ], [
            locations1_s,
        ], [
            gates1_s,
        ], exp_counts

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1) # [experts_num]

    # Normalize gate probabilities
    mask1_float = mask1.float()
    gates = gates * mask1_float # [token_nums, experts_num]

    locations1_sc = _one_hot_to_float(locations1_s, capacity) # [token_nums, capacity]
    combine_weights = einsum("se,sc->sec", gates, locations1_sc) # gates [token_nums, experts_num], locations1_sc [token_nums, capacity]

    dispatch_mask = combine_weights.bool() # [token_nums, experts_num, capacity]
    # l_aux-tensor, combine_weights-[token_nums, experts_num, capacity], exp_counts-[experts_num]
    return l_aux, combine_weights, dispatch_mask, exp_counts # 1.1508

USE_EINSUM = True

def einsum(rule, a, b):
    if USE_EINSUM:
        return torch.einsum(rule, a, b)
    elif rule == 's,se->se':
        return a.reshape(a.shape[0], -1) * b
    elif rule == 'se,sc->sec':
        return a.unsqueeze(2) * b.unsqueeze(1)
    elif rule == 'se,se->s':
        return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).reshape(-1)
    elif rule == 'sec,sm->ecm':
        s = a.shape[0]
        e = a.shape[1]
        c = a.shape[2]
        m = b.shape[1]
        return torch.matmul(a.reshape(s, -1).t(), b).reshape(e, c, m)
    elif rule == 'sec,ecm->sm':
        return torch.matmul(a.reshape(a.shape[0], -1), b.reshape(-1, b.shape[-1]))
    elif rule == 'ks,ksm->sm':
        k = b.shape[0]
        s = b.shape[1]
        m = b.shape[2]
        # [k, s] -> [s, k] -> [s, 1, k]
        a = a.t().unsqueeze(1)
        # [k,s,m] -> [k, sm] -> [sm, k] -> [s, m, k]
        b = b.reshape(k, -1).t().reshape(s, m, k)
        # bmm([s, 1, k], [s, m, k]^t) -> [s, m, 1]
        return torch.bmm(a, b.transpose(1, 2)).squeeze(2)
    else:
        return torch.einsum(rule, a, b)


def top2gating(logits: Tensor, capacity_factor: float, min_capacity: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    capacity = _capacity(gates, torch.tensor(capacity_factor * 2), torch.tensor(min_capacity)) # max(min_capacity, num_tokens/num_experts*capacity_factor*topk)

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)  # torch.Size([num_tokens]), each element represents the index of expert
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts) # [num_tokens, num_experts]

    # Create a mask for 2nd's expert per token using Gumbel-max trick
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # Replace top-expert with min value
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf")) # [num_tokens, num_experts], ture的位置-inf
    indices2_s = torch.argmax(logits_except1, dim=1) # [num_tokens], 将概率最大的值置-inf,通过torch.argmax重新寻找概率最大的,即第二大
    mask2 = F.one_hot(indices2_s, num_classes=num_experts) # [num_tokens, num_experts]

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu') # [num_experts], each element -> num of tokens

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts

    # Remove locations outside capacity from mask
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    mask2_float = mask2.float()
    gates1_s = einsum("se,se->s", gates, mask1_float)
    gates2_s = einsum("se,se->s", gates, mask2_float)
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s

    # Calculate combine_weights and dispatch_mask
    gates1 = einsum("s,se->se", gates1_s, mask1_float)
    gates2 = einsum("s,se->se", gates2_s, mask2_float)
    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    locations2_sc = _one_hot_to_float(locations2_s, capacity)
    combine1_sec = einsum("se,sc->sec", gates1, locations1_sc)
    combine2_sec = einsum("se,sc->sec", gates2, locations2_sc)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts


class TopKGate(Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(self,
                 model_dim: int,
                 num_experts: int,
                 k: int = 1,
                 capacity_factor: float = 1.0,
                 eval_capacity_factor: float = 1.0,
                 min_capacity: int = 8,
                 noisy_gate_policy: Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts: bool = True) -> None:
        super().__init__()

        # Only top-1 and top-2 are supported at the moment.
        if k != 1 and k != 2:
            raise ValueError('Only top-1 and top-2 gatings are supported.')
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False).float()
        self.k = k
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.noisy_gate_policy = noisy_gate_policy
        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False
        self.gate_time = 0.0
        self.drop_tokens = drop_tokens
        self.use_rts = use_rts

    def forward(self,
                input: torch.Tensor,
                used_token: torch.Tensor = None,
                use_tutel: bool = False) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore

        if self.wall_clock_breakdown:
            self.timers('TopKGate').start()

        if self.wg.weight.dtype != torch.float32:
            self.wg = self.wg.float()
        input_fp32 = input.float() # change inot float32
        # input jittering
        if self.noisy_gate_policy == 'Jitter' and self.training:
            input_fp32 = multiplicative_jitter(input_fp32, device=input.device)
        logits = self.wg(input_fp32) # bz, seq_len, model_dim

        if self.k == 1:
            gate_output = top1gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                     self.min_capacity, used_token, self.noisy_gate_policy if self.training else None,
                                     self.drop_tokens, self.use_rts, use_tutel)

        else:
            gate_output = top2gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                     self.min_capacity)

        if self.wall_clock_breakdown:
            self.timers('TopKGate').stop()
            self.gate_time = self.timers('TopKGate').elapsed(reset=False)

        return gate_output


if __name__ == '__main__':

    # Define batch size
    batch_size = 10
    seq_len = 20
    model_dim = 768
    # Create a random input tensor of shape [batch_size, seq_len, model_dim]
    input = torch.randn(seq_len, model_dim).float()
    input_fp32 = input.float() # change inot float32
    logits = test_input(input_fp32) # torch.Size([10, 12, 16])

    # gate_output = top1gating(logits, 
    #                     capacity_factor=1.0, 
    #                     min_capacity=4, 
    #                     noisy_gate_policy='RSample',
    #                     drop_tokens=True, 
    #                     use_rts=True, 
    #                     use_tutel=False)

    gate_output = top2gating(logits, 
                        capacity_factor=1.0,
                        min_capacity=4,
                        )

    # import torch
    # # import torch 包

    # index = torch.LongTensor([[0, 1, 1, 0, 0],[2, 0, 0, 1, 1]])
    # # 初始化 b 为size 为 (3, 5) 的向量，二维向量，3行5列，每个元素被初始化为 0
    # b = torch.zeros(3, 5)
    
    # # torch.zeros_like(mask1).scatter_(0, top_idx, 1)
    # c = b.scatter_(0, index, 1)

    print('99999')


