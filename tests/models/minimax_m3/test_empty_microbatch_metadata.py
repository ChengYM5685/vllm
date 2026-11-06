# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 metadata builders on a microbatch that holds no query tokens.

With DBO, the V2 runner splits every DP rank's batch at the same token offset,
so a rank with less than half the largest rank's tokens gets a trailing
microbatch that lies wholly in the DP padding. Its request slice is clamped
onto the last request, which is left with zero query tokens.
"""

import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata

# Spec decode with three draft tokens.
DECODE_THRESHOLD = 4


def _metadata(query_lens: list[int], seq_lens: list[int]) -> CommonAttentionMetadata:
    num_reqs = len(query_lens)
    qsl = torch.zeros(num_reqs + 1, dtype=torch.int32)
    qsl[1:] = torch.cumsum(torch.tensor(query_lens, dtype=torch.int32), 0)
    num_tokens = int(qsl[-1])
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl.clone(),
        seq_lens=seq_lens_t,
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=torch.zeros(num_reqs, 8, dtype=torch.int32),
        slot_mapping=torch.full((num_tokens,), -1, dtype=torch.int64),
        positions=torch.zeros(num_tokens, dtype=torch.int64),
        seq_lens_cpu_upper_bound=seq_lens_t.clone(),
    )


def _builder(cls):
    # Skip __init__: it needs a full VllmConfig, and the zero-token and
    # decode-only paths only read these attributes.
    builder = object.__new__(cls)
    builder.reorder_batch_threshold = DECODE_THRESHOLD
    builder.max_decode_query_len = DECODE_THRESHOLD
    builder.context_len_buffer = torch.zeros(64, dtype=torch.int32)
    builder.page16_slot_mapping_buffer = None
    return builder


def _builder_classes():
    from vllm.models.minimax_m3.amd.indexer_msa import (
        MiniMaxM3IndexerMSAMetadataBuilder,
    )
    from vllm.models.minimax_m3.common.indexer import (
        MiniMaxM3IndexerTritonMetadataBuilder,
    )
    from vllm.models.minimax_m3.common.sparse_attention import (
        MiniMaxM3SparseMetadataBuilder,
    )

    return [
        MiniMaxM3IndexerMSAMetadataBuilder,
        MiniMaxM3IndexerTritonMetadataBuilder,
        MiniMaxM3SparseMetadataBuilder,
    ]


@pytest.mark.parametrize("cls", _builder_classes(), ids=lambda c: c.__name__)
def test_padding_only_microbatch_has_no_work(cls):
    # The clamped last request: 1000 tokens of context, none scheduled here.
    md = _builder(cls).build(0, _metadata([0], [1000]))
    assert md.num_actual_tokens == 0
    assert md.num_decodes == md.num_prefills == 0
    assert md.num_decode_tokens == md.num_prefill_tokens == 0
    assert md.decode is None and md.prefill is None


@pytest.mark.parametrize("cls", _builder_classes(), ids=lambda c: c.__name__)
def test_uniform_decode_unchanged(cls):
    md = _builder(cls).build(0, _metadata([4, 4, 4], [100, 200, 300]))
    assert (md.num_decodes, md.num_prefills) == (3, 0)
    assert md.num_decode_tokens == 12
    assert md.decode is not None and md.decode.decode_query_len == 4
