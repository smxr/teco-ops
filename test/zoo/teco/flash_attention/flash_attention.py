# BSD 3-Clause License
#
# Copyright (c) 2024, Tecorigin Co., Ltd.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# encoding:utf-8
'''
FlashAttention CPU reference implementation.

Computes standard attention: O = softmax(Q @ K^T / sqrt(d)) @ V

Inputs (from prototxt):
    - blockTable [batch_size, block_table_dim] int32
    - qData [total_q_tokens, local_head_num, head_size] half
    - kCache [max_block_num, local_kv_head_num, block_size, head_size] half
    - vCache [max_block_num, local_kv_head_num, block_size, head_size] half

Proto params:
    - max_prefill_len, max_decode_len
    - q_seq_lens (repeated int32, length=batch_size)
    - kv_seq_lens (repeated int32, length=batch_size)

Output:
    - oData [total_q_tokens, local_head_num, head_size] half
'''

import os
import sys
import json
import math
import torch
import numpy as np
sys.path.append("../zoo/teco/")
sys.path.append("../")
from executor import *


def check_inputs(param_path, input_lists, reuse_lists, output_lists):
    if param_path == "":
        print("The path of prototxt file is empty.")
        return False
    if len(input_lists) != 4:
        print("The number of input data is wrong (expected 4: blockTable, qData, kCache, vCache).")
        return False
    if len(reuse_lists) != 0:
        print("The number of reuse data is wrong.")
        return False
    if len(output_lists) != 1:
        print("The number of output data is wrong (expected 1: oData).")
        return False
    return True


def test_flash_attention(param_path, input_lists, reuse_lists, output_lists, device):
    if not check_inputs(param_path, input_lists, reuse_lists, output_lists):
        return

    if device == "cuda":
        is_avail, used_device = is_device_available(device)
        if not is_avail:
            return

    params = read_prototxt(param_path)
    input_params = params["input"]
    output_params = params["output"]
    fa_params = params["tecokernel_param"]["flash_attention_param"]

    # Read proto params
    q_seq_lens = [int(x) for x in fa_params["q_seq_lens"]]
    kv_seq_lens = [int(x) for x in fa_params["kv_seq_lens"]]
    batch_size = len(q_seq_lens)

    # Read input tensors
    block_table = to_tensor(input_lists[0], input_params[0], device=device)
    q_data = to_tensor(input_lists[1], input_params[1], device=device)
    k_cache = to_tensor(input_lists[2], input_params[2], device=device)
    v_cache = to_tensor(input_lists[3], input_params[3], device=device)

    # Infer shapes
    # block_table: [batch_size, block_table_dim]
    # q_data: [total_q_tokens, num_heads, head_size]
    # k_cache: [max_block_num, num_kv_heads, block_size, head_size]
    num_heads = q_data.shape[1]
    num_kv_heads = k_cache.shape[1]
    head_size = q_data.shape[2]

    # Convert to float64 for CPU precision
    q_data = q_data.to(torch.float64)
    k_cache = k_cache.to(torch.float64)
    v_cache = v_cache.to(torch.float64)

    outputs = []
    q_offset = 0
    for b in range(batch_size):
        q_len = int(q_seq_lens[b])
        kv_len = int(kv_seq_lens[b])

        # Slice Q for this batch item
        Q_b = q_data[q_offset:q_offset + q_len]  # [q_len, num_heads, head_size]

        # Look up block from block_table
        block_id = int(block_table[b, 0].item()) if block_table.dim() > 1 else int(block_table[0].item())

        # Extract K, V from cache
        # k_cache: [max_block_num, num_kv_heads, block_size, head_size]
        # Take first kv_len tokens from the block
        K_b = k_cache[block_id, :, :kv_len, :]    # [num_kv_heads, kv_len, head_size]
        V_b = v_cache[block_id, :, :kv_len, :]    # [num_kv_heads, kv_len, head_size]

        # GQA: repeat kv heads to match num_heads
        if num_heads != num_kv_heads:
            group_size = num_heads // num_kv_heads
            K_b = K_b.repeat_interleave(group_size, dim=0)  # [num_heads, kv_len, head_size]
            V_b = V_b.repeat_interleave(group_size, dim=0)  # [num_heads, kv_len, head_size]

        # Standard attention
        scale = 1.0 / math.sqrt(float(head_size))
        # score: [q_len, num_heads, kv_len]
        score = torch.matmul(Q_b, K_b.transpose(-2, -1)) * scale
        prob = torch.softmax(score, dim=-1)  # [q_len, num_heads, kv_len]
        O_b = torch.matmul(prob, V_b)  # [q_len, num_heads, head_size]

        outputs.append(O_b)
        q_offset += q_len

    # Concatenate all batch outputs
    output = torch.cat(outputs, dim=0)  # [total_q_tokens, num_heads, head_size]

    # Save output
    output_dtype = output_params["dtype"]
    with open(output_lists[0], "wb") as f:
        save_tensor(f, output, output_dtype)


def parse_params(filename):
    with open(filename, "r") as f:
        params = json.load(f)
    return params


if __name__ == "__main__":
    params = parse_params(sys.argv[1])
    device = sys.argv[2]
    test_flash_attention(params["param_path"], params["input_lists"],
                         params["reuse_lists"], params["output_lists"], device)
