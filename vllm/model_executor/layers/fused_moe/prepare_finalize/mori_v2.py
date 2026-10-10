# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.prepare_finalize.mori import (
    MoriPrepareAndFinalize,
)


class MoriV2PrepareAndFinalize(MoriPrepareAndFinalize):
    """Prepare/Finalize using MoRI EPv2 (`mori.ops.dispatch_combine_v2`)."""

    def __init__(
        self,
        mori_op: Any,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        rank_expert_offset: int,
        num_local_experts: int,
        use_fp8_dispatch: bool = False,
        scale_dim: int = 0,
    ):
        super().__init__(
            mori_op, max_tokens_per_rank, num_dispatchers, use_fp8_dispatch
        )
        self.scale_dim = scale_dim
        self.rank_expert_offset = rank_expert_offset
        self.num_local_experts = num_local_experts
        self._padding_expert_ids: torch.Tensor | None = None
        self._routing: Any = None

    def _route_padding_locally(self, topk_ids: torch.Tensor) -> torch.Tensor:
        # Padding tokens carry expert id -1 (and zero weight). MoRI EPv2 has no
        # invalid-expert slot, so send them to this rank's own experts instead.
        if self._padding_expert_ids is None:
            slots = torch.arange(topk_ids.shape[1], device=topk_ids.device)
            self._padding_expert_ids = (
                self.rank_expert_offset + slots % self.num_local_experts
            ).to(topk_ids.dtype)
        return torch.where(topk_ids < 0, self._padding_expert_ids, topk_ids)

    @staticmethod
    def _max_recv_rows(capacity: int) -> int:
        if not is_forward_context_available():
            return capacity
        dp_metadata = get_forward_context().dp_metadata
        if dp_metadata is None:
            return capacity
        return min(capacity, int(dp_metadata.num_tokens_across_dp_cpu.sum()))

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        assert not apply_router_weight_on_input, (
            "mori does not support apply_router_weight_on_input=True now."
        )
        a1, scale = self._quantize_dispatch_input(a1, quant_config, defer_input_quant)
        topk_ids = self._route_padding_locally(topk_ids)

        # combine() takes its token mapping from the routing handle that
        # dispatch() returns, not from topk ids.
        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
            self._routing,
        ) = self.mori_op.dispatch(
            a1, topk_weights.to(torch.float32), scale, topk_ids, return_routing=True
        )
        # The receive buffers span MoRI's worst-case capacity, but this rank
        # receives at most every DP rank's tokens, packed at the front. Trim
        # them so the experts and combine's copy-in skip the unused rows.
        num_rows = self._max_recv_rows(dispatch_a1.shape[0])
        dispatch_a1 = dispatch_a1[:num_rows]
        dispatch_weights = dispatch_weights[:num_rows]
        dispatch_ids = dispatch_ids[:num_rows]
        if dispatch_scale is not None:
            # MoRI forwards scales as opaque int32 rows, possibly padded.
            dispatch_scale = dispatch_scale[:num_rows].view(torch.float32)
            dispatch_scale = dispatch_scale[:, : self.scale_dim]

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=dispatch_recv_token_num, expert_num_tokens_cpu=None
        )

        return (
            dispatch_a1,
            dispatch_scale,
            expert_tokens_meta,
            dispatch_ids,
            dispatch_weights,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        assert self._routing is not None, (
            "finalize() called before prepare(); MoRI EPv2 combine() needs the "
            "routing handle that dispatch() returned"
        )
        result = self.mori_op.combine(
            fused_expert_output, None, None, routing=self._routing
        )[0]
        self._routing = None
        output.copy_(result[: output.shape[0]])
