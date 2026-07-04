// BSD 3-Clause License
//
// Copyright (c) 2024, Tecorigin Co., Ltd.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <stdio.h>
#include <iostream>
#include <string>
#include <numeric>
#include "zoo/teco/convert.h"
#include "common/time.hpp"
#include "zoo/teco/flash_attention/flash_attention.h"


namespace optest {

void FlashAttentionExecutor::paramCheck() {
    if (parser_->inputs().size() != 4) {
        ALLOG(ERROR) << "flash_attention expects 4 inputs (blockTable, qData, kCache, vCache).";
        throw std::invalid_argument(std::string(__FILE__) + ":" + std::to_string(__LINE__));  // NOLINT
    }

    if (parser_->outputs().size() != 1) {
        ALLOG(ERROR) << "flash_attention expects 1 output (oData).";
        throw std::invalid_argument(std::string(__FILE__) + ":" + std::to_string(__LINE__));  // NOLINT
    }

    auto mt_k = parser_->input(2);
    auto mt_v = parser_->input(3);
    if (mt_k->shape != mt_v->shape) {
        ALLOG(ERROR) << "kCache and vCache must have the same shape.";
        throw std::invalid_argument(std::string(__FILE__) + ":" + std::to_string(__LINE__));  // NOLINT
    }
}

void FlashAttentionExecutor::paramParse() {
    // Read proto params
    auto fa_param = parser_->getProtoNode()->tecokernel_param().flash_attention_param();
    max_prefill_len_ = fa_param.max_prefill_len();
    max_decode_len_ = fa_param.max_decode_len();

    // Read q_seq_lens and kv_seq_lens from proto repeated fields
    q_seq_lens_.clear();
    for (int i = 0; i < fa_param.q_seq_lens_size(); ++i) {
        q_seq_lens_.push_back(fa_param.q_seq_lens(i));
    }
    kv_seq_lens_.clear();
    for (int i = 0; i < fa_param.kv_seq_lens_size(); ++i) {
        kv_seq_lens_.push_back(fa_param.kv_seq_lens(i));
    }

    batch_size_ = q_seq_lens_.size();

    // Infer shapes from tensor descriptors in the prototxt
    // Input 0: blockTable [batch_size, block_table_dim]
    block_table_dim_ = parser_->input(0)->shape[1];

    // Input 1: qData [total_q_tokens, local_head_num, size_per_head]
    total_q_tokens_ = parser_->input(1)->shape[0];
    local_head_num_ = parser_->input(1)->shape[1];
    size_per_head_ = parser_->input(1)->shape[2];

    // Input 2: kCache [max_block_num, local_kv_head_num, block_size, size_per_head]
    max_block_num_ = parser_->input(2)->shape[0];
    local_kv_head_num_ = parser_->input(2)->shape[1];
    block_size_ = parser_->input(2)->shape[2];

    // Verify total_q_tokens matches sum(q_seq_lens)
    int inferred_total = std::accumulate(q_seq_lens_.begin(), q_seq_lens_.end(), 0);
    if (total_q_tokens_ != inferred_total) {
        ALLOG(WARNING) << "qData.shape[0]=" << total_q_tokens_
                       << " != sum(q_seq_lens)=" << inferred_total
                       << ". Using qData.shape[0]=" << total_q_tokens_;
    }
}

void FlashAttentionExecutor::paramGeneration() {
    blockTableDesc_ = getInputDesc<tecoopsTensorDescriptor_t>(0);
    blockTable_ = dev_input[0];
    qDataDesc_ = getInputDesc<tecoopsTensorDescriptor_t>(1);
    qData_ = dev_input[1];
    kCacheDesc_ = getInputDesc<tecoopsTensorDescriptor_t>(2);
    kCache_ = dev_input[2];
    vCacheDesc_ = getInputDesc<tecoopsTensorDescriptor_t>(3);
    vCache_ = dev_input[3];

    oDataDesc_ = getOutputDesc<tecoopsTensorDescriptor_t>(0);
    oData_ = dev_output[0];
}

void FlashAttentionExecutor::compute() {
#ifdef USE_TECO
    checkTECOOPS(tecoopsFlashAttention(handle_,
        max_prefill_len_, max_decode_len_, max_block_num_,
        q_seq_lens_.data(), kv_seq_lens_.data(),
        blockTableDesc_, blockTable_,
        qDataDesc_, qData_,
        kCacheDesc_, kCache_,
        vCacheDesc_, vCache_,
        oDataDesc_, oData_,
        /*workspace=*/nullptr));
#endif
}

int64_t FlashAttentionExecutor::getTheoryOps() {
    // For each query token:
    //   score = Q @ K^T: 2 * total_q * total_kv * head_size  (MAD ops)
    //   softmax: ~3 * total_q * total_kv
    //   output = score @ V: 2 * total_q * total_kv * head_size (MAD ops)
    // Approx: 4 * total_q_tokens * total_kv_tokens * size_per_head * local_head_num
    int total_kv_tokens = std::accumulate(kv_seq_lens_.begin(), kv_seq_lens_.end(), 0);
    int64_t theory_ops = 4LL * total_q_tokens_ * total_kv_tokens * size_per_head_ * local_head_num_;
    return theory_ops;
}

int64_t FlashAttentionExecutor::getTheoryIoSize() {
    return getIoSize();
}

void FlashAttentionExecutor::cpuCompute() {
    pythonComputeCPU("cpu");
}

void FlashAttentionExecutor::gpuCompute() {
    // No CUDA reference for flash_attention
}

void FlashAttentionExecutor::destroy() {
    // q_seq_lens_ and kv_seq_lens_ are std::vector, auto cleanup
}

}  // namespace optest
