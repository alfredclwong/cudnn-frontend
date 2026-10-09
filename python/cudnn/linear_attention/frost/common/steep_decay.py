# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This kernel is derived from cuDNN, NVIDIA Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exact intra-tile decay for steep key channels in the GDN-2 BT = 16 kernels.

The chunk kernels split the intra-tile decay of key channel d between tokens t >= s,
``2^(g[t, d] - g[s, d])`` with ``g`` the inclusive log2 gate prefix inside the 16-token
tile, across the two operands of one register MMA: K decay / Q decay carry
``exp2(+g[t, d])`` and K inv carries ``rcp(exp2(g[s, d]))``.  Once a prefix falls below
-126, ``exp2(+g)`` leaves the fp32 normal range, the approximate reciprocal returns inf,
and ``0 * inf`` turns KK, A and the state update NaN for the rest of the sequence.

A channel is *steep* in a tile when its prefix falls below ``STEEP_LOG2_PREFIX`` there (``-120``;
``-14`` with fp16 operands, where ``k * 2^-g`` overflows the fp16 range near a prefix of ``-16``).

Dispatch.  The steep path costs registers and code even where it never runs, so each kernel's ``host`` launches it
twice (``cfg.steep_mode``):

- ``STEEP_FAST``: the original kernel plus one ballot per scan warp; a warp that finds a steep channel stores 1 into
  the second int32 of the kernel's scheduler slot, which the prologue zeroes every call.  A steep tile's outputs are
  garbage at this point.
- ``STEEP_EXACT``: launched right after, from the same host, with the steep path compiled in.  Every thread exits on
  entry while the flag is 0, so a call without a steep tile pays one empty launch; otherwise the kernel redoes the whole
  launch exactly (pulling tiles off the flag int, past its 1), overwriting every output before the next kernel reads it.
  No host synchronisation is involved, so the pair is safe under CUDA graphs and ``torch.compile``.
- ``STEEP_INLINE``: one launch with the steep path compiled in, for launches whose output aliases an input they read
  (``overwrite_initial_state``, ``state_indices``), where the exact launch would read the fast launch's garbage.

The steep path (``STEEP_EXACT`` / ``STEEP_INLINE``):

- the gate scan stages a prefix below the threshold on gate-exchange rows ``0 .. BT - 2`` as
  the prefix itself instead of ``exp2(prefix)`` (``exp2`` values are positive, staged prefixes
  negative, so :func:`steep_log_prefix` tells them apart); the last row keeps ``exp2(prefix)``,
  which the state decay reads, and a steep channel's last-row prefix goes to a per-channel array;
  each scan warp publishes one ballot word;
- the row stage runs unchanged, so a steep channel's operand entries come out of the exchange as
  finite garbage; one block per row stage, taken only in a steep tile, overwrites them
  (``steep_store_operands``): K decay / Q decay from ``exp2(g[t])``, which may underflow to zero harmlessly, and
  K inv and K restore the same value, ``k[t] * exp2(g[BT - 1] - g[t])``, whose exponent is never positive;
- ``steep_row_corrections`` adds the steep channels' exact KK and A entries, every
  exponent ``g[t] - g[s] <= 0``, minus the products the register MMAs formed with
  their K inv column; the register-MMA warps add the corrections to their
  accumulators;
- the backward rebuilds a steep channel's dQ / dK / dgate / dbeta lanes from the accumulators and adds the exact
  intra-tile terms (``steep_grad_terms``).
"""

import dataclasses

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm

from cudnn.frost.tile_dsl.barrier import launch_dependent_grids
from cudnn.frost.tile_dsl.pointwise import sigmoid
from cudnn.frost.tile_dsl.swizzle import swizzle_xor_128b

STEEP_LOG2_PREFIX: float = -120.0
"""bf16 operands: a key channel whose inclusive log2 gate prefix inside a tile falls below this is steep for the tile.
``exp2`` of the prefix reaches the fp32 (and bf16) normal floor, ``2^-126``, six binades further down."""

STEEP_LOG2_PREFIX_FP16: float = -14.0
"""fp16 operands: K inv = k * 2^-g reaches the fp16 maximum, ``65504 ~ 2^16``, two binades further down."""


def steep_log2_prefix(io_dtype) -> float:
    """The steep threshold for the operand dtype (a trace-time constant)."""
    return STEEP_LOG2_PREFIX_FP16 if io_dtype == cutlass.Float16 else STEEP_LOG2_PREFIX


STEEP_FAST: int = 0
"""Kernel ``cfg.steep_mode``: the original arithmetic plus the steep vote; a steep tile raises the call's steep flag.
``host`` follows the launch with a ``STEEP_EXACT`` launch of the same kernel."""

STEEP_EXACT: int = 1
"""The exact kernel of a ``STEEP_FAST`` launch: every thread exits on entry unless the flag is raised, else it redoes
the whole launch with the steep path, pulling tiles from its own ticket."""

STEEP_INLINE: int = 2
"""One launch with the steep path compiled in, for plans whose outputs alias their inputs (a final state written over
the initial state), where a second launch would read its own output."""

STEEP_FLAG_SLOT: int = 1
"""The steep flag and the exact launch's ticket share the second int32 of the kernel's scheduler slot, which the
prologue zeroes with the first: 0 means no steep tile, and the exact launch's tickets start at the 1 the fast launch
stored."""


def steep_exact_cfg(cfg):
    """The ``STEEP_EXACT`` twin of a ``STEEP_FAST`` kernel config (a trace-time copy)."""
    return dataclasses.replace(cfg, steep_mode=STEEP_EXACT)


@cute.jit
def steep_exit_unless_flagged(cfg: cutlass.Constexpr, flag_slot):
    """``STEEP_EXACT`` kernel entry, after the grid-dependency wait: release the next launch (which waits for this grid
    anyway) and leave unless the fast launch raised the flag."""
    if cutlass.const_expr(cfg.steep_mode == STEEP_EXACT):
        launch_dependent_grids()
        if flag_slot[STEEP_FLAG_SLOT] == cutlass.Int32(0):
            nvvm.exit()


@cute.jit
def steep_raise_flag(cfg: cutlass.Constexpr, flag_slot, steep_word, lane_idx):
    """``STEEP_FAST`` scan: a warp whose ballot found a steep channel raises the flag (idempotent, never read back)."""
    if cutlass.const_expr(cfg.steep_mode == STEEP_FAST):
        if steep_word != 0:
            if lane_idx == 0:
                flag_slot[STEEP_FLAG_SLOT] = cutlass.Int32(1)


@cute.jit
def steep_stage(cfg: cutlass.Constexpr, prefix, exp_prefix, threshold):
    """The gate-exchange entry of a row below the last: ``exp2(prefix)``, or with the steep path compiled in the prefix
    itself when it is below ``threshold`` (:func:`steep_decode`)."""
    value = exp_prefix
    if cutlass.const_expr(cfg.steep_mode != STEEP_FAST):
        value = prefix if prefix < threshold else exp_prefix
    return value


@cute.jit
def steep_ticket(cfg: cutlass.Constexpr, mScheduler):
    """The next ticket off the scheduler slot: the first int32, or under ``STEEP_EXACT`` the flag slot past its 1."""
    fetched = cutlass.Int32(0)
    if cutlass.const_expr(cfg.steep_mode == STEEP_EXACT):
        fetched = cutlass.Int32(nvvm.atomicrmw("add", mScheduler.iterator + STEEP_FLAG_SLOT, cutlass.Int32(1), mem_order="relaxed", syncscope="gpu")) - 1
    else:
        fetched = cutlass.Int32(nvvm.atomicrmw("add", mScheduler.iterator, cutlass.Int32(1), mem_order="relaxed", syncscope="gpu"))
    return fetched


@cute.jit
def exchange_offset(cfg: cutlass.Constexpr, row, channel):
    """Element offset of ``(row, channel)`` in one ``[BT, DK]`` fp32 gate-exchange stage."""
    seg = channel // 32
    return seg * (cfg.b_t * 32) + row * 32 + swizzle_xor_128b(row ^ seg, channel - seg * 32, elem_bytes=4)


@cute.jit
def raw_offset(cfg: cutlass.Constexpr, row, channel):
    """Element offset of ``(row, channel)`` in one ``[BT, DK]`` 16-bit row-major operand tile."""
    seg = channel // 64
    return seg * (cfg.b_t * 64) + row * 64 + swizzle_xor_128b(row, channel - seg * 64, elem_bytes=2)


@cute.jit
def lane_steep_bits(cfg: cutlass.Constexpr, words_ptr, lane_in_row_group, channel_rows: cutlass.Constexpr[int]):
    """The steep channels a row-stage lane owns: bit ``h * 8 + j`` is channel ``h * 64 + lane_in_row_group * 8 + j``.

    ``words_ptr`` holds one ballot word per scan warp; channel ``c`` is bit ``c % channel_rows`` of word
    ``c // channel_rows``."""
    bits = cutlass.Int32(0)
    for dim_half in cutlass.range_constexpr(cfg.d_k // 64):
        first = dim_half * 64 + lane_in_row_group * 8
        word = (words_ptr + first // channel_rows).load()
        bits = bits | (((word >> (first % channel_rows)) & 0xFF) << (dim_half * 8))
    return bits


@cute.jit
def steep_beta(cfg: cutlass.Constexpr, beta_ptr, row, channel):
    """The erase gate the operands used, from the chunk's raw beta tile (beta_eff under the guard, written back)."""
    beta = (beta_ptr + raw_offset(cfg, row, channel)).load().to(cutlass.Float32)
    if cutlass.const_expr(cfg.beta_sigmoid and not cfg.beta_guard):
        beta = (sigmoid(beta) * (2.0 if cfg.allow_neg_eigval else 1.0)).to(cfg.io_dtype).to(cutlass.Float32)
    return beta


@cute.jit
def steep_decode(value):
    """A staged gate-exchange entry of rows ``0 .. BT - 2`` as ``(log2 prefix, exp2(prefix))``: a negative entry is
    the prefix itself, a positive one ``exp2`` of a prefix above the steep threshold."""
    log_g = value
    exp_g = value
    if value < cutlass.Float32(0.0):
        exp_g = cute.math.exp2(value, fastmath=True)
    else:
        log_g = cute.math.log2(value)
    return log_g, exp_g


@cute.jit
def steep_log_prefix(cfg: cutlass.Constexpr, exchange_ptr, log_last_ptr, row, channel):
    """A steep channel's log2 prefix on ``row``: decoded from the gate exchange on rows ``0 .. BT - 2``, the last-row
    array on the last."""
    log_g = (log_last_ptr + channel).load()
    if row < cfg.b_t - 1:
        log_g, _ = steep_decode((exchange_ptr + exchange_offset(cfg, row, channel)).load())
    return log_g


@cute.jit
def steep_operand_scales(cfg: cutlass.Constexpr, row, value, log_last):
    """The row stage's (exp2(g[t]), K inv / K restore scale) for a steep channel.

    ``value`` is the channel's gate-exchange entry on ``row`` (:func:`steep_decode` on rows ``0 .. BT - 2``,
    ``exp2(prefix)`` on the last row).  Both results lie in ``[0, 1]`` for a non-increasing prefix."""
    exp_g = value
    log_g = log_last
    if row < cfg.b_t - 1:
        log_g, exp_g = steep_decode(value)
    return exp_g, cute.math.exp2(log_last - log_g, fastmath=True)


@cute.jit
def steep_store_operands(
    cfg: cutlass.Constexpr,
    bits,
    row,
    lane_in_row_group,
    k_norm,
    q_scale,
    exchange_ptr,
    log_last_ptr,
    raw_k_ptr,
    raw_q_ptr,
    beta_ptr,
    k_inv_ptr,
    k_decay_ptr,
    q_decay_ptr,
    k_restore_ptr,
):
    """Overwrite the steep channels' entries of tile row ``row`` in the operand tiles the row stage stored.

    The row stage builds every channel from the gate exchange, which holds a steep channel's log2 prefix, so its
    entries are finite garbage; this lane rewrites its own (``bits``, as :func:`lane_steep_bits`) as

        K decay = beta k_n 2^g[row],   Q decay = q_n 2^g[row],   K inv = K restore = k_n 2^(g[BT - 1] - g[row])

    rounded to the io dtype, with the row stage's operation order.  ``k_norm`` / ``q_scale`` are the row's K inverse
    norm and Q factor; a ``None`` operand pointer skips that tile (``raw_q_ptr`` / ``beta_ptr`` may be ``None`` when no
    tile needs them).  The tiles share the raw tiles' ``[DK / 64, BT, 64]`` 128B-swizzled layout."""
    for j in cutlass.range(cfg.d_k // 64 * 8, unroll=1):
        if ((bits >> j) & 1) != 0:
            channel = (j // 8) * 64 + lane_in_row_group * 8 + (j % 8)
            offset = raw_offset(cfg, row, channel)
            exp_t, restore = steep_operand_scales(cfg, row, (exchange_ptr + exchange_offset(cfg, row, channel)).load(), (log_last_ptr + channel).load())
            if cutlass.const_expr(k_inv_ptr is not None or k_decay_ptr is not None or k_restore_ptr is not None):
                k_n = (raw_k_ptr + offset).load().to(cutlass.Float32) * k_norm
                k_inv = (k_n * restore).to(cfg.io_dtype)
                if cutlass.const_expr(k_inv_ptr is not None):
                    (k_inv_ptr + offset).store(k_inv)
                if cutlass.const_expr(k_restore_ptr is not None):
                    (k_restore_ptr + offset).store(k_inv)
                if cutlass.const_expr(k_decay_ptr is not None):
                    (k_decay_ptr + offset).store(((k_n * steep_beta(cfg, beta_ptr, row, channel)) * exp_t).to(cfg.io_dtype))
            if cutlass.const_expr(q_decay_ptr is not None):
                (q_decay_ptr + offset).store((((raw_q_ptr + offset).load().to(cutlass.Float32) * q_scale) * exp_t).to(cfg.io_dtype))


@cute.jit
def steep_row_corrections(
    cfg: cutlass.Constexpr,
    bits,
    row,
    lane_in_row_group,
    k_norm,
    q_scale,
    exchange_ptr,
    log_last_ptr,
    raw_k_ptr,
    raw_q_ptr,
    beta_ptr,
    k_norm_ptr,
    corr_ptr,
):
    """Add the steep channels' KK and A entries of tile row ``row`` to ``corr_ptr``.

    Called by all lanes of the row stage after its K inv / K decay stores.  ``bits`` is the lane's
    :func:`lane_steep_bits`; ``k_norm`` / ``q_scale`` the row's K inverse norm and Q factor (the inverse norm, times
    the attention scale where the A tile carries it); ``raw_q_ptr`` is ``None`` without an A tile.  Every operand is
    reloaded from the chunk's raw tiles, so the slow path holds no registers across the fast one.  ``corr_ptr`` is
    ``[2, BT, BT]`` fp32, the KK rows then the A rows; this row group zeroes its rows and the lanes add, for
    ``s <= row`` and each steep channel,

        A[row, s]  += q_n[row] k_n[s] 2^(g[row] - g[s]) - Q decay[row] K inv[s]
        KK[row, s] += beta[row] k_n[row] k_n[s] 2^(g[row] - g[s]) - K decay[row] K inv[s]   (s < row)

    with K decay / Q decay / K inv rounded to the io dtype exactly as the row stage stored them, so the second
    terms cancel the register MMAs' products with the steep columns."""
    with_a = cutlass.const_expr(raw_q_ptr is not None)
    entries = cutlass.const_expr(cfg.b_t // 8)
    kk_row_ptr = corr_ptr + row * cfg.b_t
    a_row_ptr = corr_ptr + cfg.b_t * cfg.b_t + row * cfg.b_t
    for e in cutlass.range_constexpr(entries):
        (kk_row_ptr + lane_in_row_group * entries + e).store(cutlass.Float32(0.0))
        if cutlass.const_expr(with_a):
            (a_row_ptr + lane_in_row_group * entries + e).store(cutlass.Float32(0.0))
    nvvm.bar_warp_sync(0xFFFFFFFF)
    for j in cutlass.range(cfg.d_k // 64 * 8, unroll=1):
        if ((bits >> j) & 1) != 0:
            channel = (j // 8) * 64 + lane_in_row_group * 8 + (j % 8)
            log_last = (log_last_ptr + channel).load()
            log_t = steep_log_prefix(cfg, exchange_ptr, log_last_ptr, row, channel)
            exp_t, _ = steep_operand_scales(cfg, row, (exchange_ptr + exchange_offset(cfg, row, channel)).load(), log_last)
            x = ((raw_k_ptr + raw_offset(cfg, row, channel)).load().to(cutlass.Float32) * k_norm) * steep_beta(cfg, beta_ptr, row, channel)
            x_decay = (x * exp_t).to(cfg.io_dtype).to(cutlass.Float32)
            if cutlass.const_expr(with_a):
                z = (raw_q_ptr + raw_offset(cfg, row, channel)).load().to(cutlass.Float32) * q_scale
                z_decay = (z * exp_t).to(cfg.io_dtype).to(cutlass.Float32)
            for s in cutlass.range(row + 1, unroll=1):
                log_s = steep_log_prefix(cfg, exchange_ptr, log_last_ptr, s, channel)
                k_s = (raw_k_ptr + raw_offset(cfg, s, channel)).load().to(cutlass.Float32)
                if cutlass.const_expr(k_norm_ptr is not None):
                    k_s = k_s * (k_norm_ptr + s).load()
                k_inv_s = (k_s * cute.math.exp2(log_last - log_s, fastmath=True)).to(cfg.io_dtype).to(cutlass.Float32)
                k_decayed = k_s * cute.math.exp2(log_t - log_s, fastmath=True)
                if s < row:
                    nvvm.atomicrmw("add", kk_row_ptr + s, x * k_decayed - x_decay * k_inv_s, space=nvvm.SharedSpace.shared_cta)
                if cutlass.const_expr(with_a):
                    nvvm.atomicrmw("add", a_row_ptr + s, z * k_decayed - z_decay * k_inv_s, space=nvvm.SharedSpace.shared_cta)


@cute.jit
def add_steep_corrections(cfg: cutlass.Constexpr, acc, corr_ptr, lane_idx):
    """Add a ``[BT, BT]`` fp32 correction tile to an m16n16 register-MMA accumulator fragment (8 values per lane)."""
    for accum_idx in cutlass.range_constexpr(8):
        row = lane_idx // 4
        if cutlass.const_expr(accum_idx % 4 >= 2):
            row = row + cutlass.Int32(8)
        col = (accum_idx // 4) * 8 + 2 * (lane_idx % 4)
        if cutlass.const_expr(accum_idx % 2 == 1):
            col = col + cutlass.Int32(1)
        acc[accum_idx] = acc[accum_idx] + (corr_ptr + row * cfg.b_t + col).load()


@cute.jit
def steep_grad_terms(cfg: cutlass.Constexpr, channel, t, scale, exchange_ptr, log_last_ptr, raw_q_ptr, raw_k_ptr, beta_ptr, q_norm_ptr, k_norm_ptr, dadm_ptr):
    """The backward's intra-tile terms of steep channel ``channel`` at token ``t`` (bprop compute group 2).

    ``dadm_ptr`` holds the chunk's strict-lower dA and dM tiles, ``[2, BT, BT]`` in the io dtype, exactly as the
    tcgen05 GEMMs read them.  With ``F(t, s) = 2^(g[t] - g[s])`` and ``K inv[s] = k_n[s] * 2^(g[BT - 1] - g[s])``
    rounded to the io dtype (the steep column of the K inv operand), returns

        dq_attn  = sum_{s<t} dA[t, s] k_n[s] F(t, s)          dq_mma  = sum_{s<t} dA[t, s] K inv[s]
        dkd_attn = sum_{s<t} dM[t, s] k_n[s] F(t, s)          dkd_mma = sum_{s<t} dM[t, s] K inv[s]
        dk_inv   = sum_{s>t} (scale q_n[s] dA[s, t] - beta[s] k_n[s] dM[s, t]) F(s, t)

    every exponent of F at most 0.  ``*_mma`` are the steep column's products that the dQ attn and dK decay GEMMs
    already accumulated."""
    log_last = (log_last_ptr + channel).load()
    log_t = steep_log_prefix(cfg, exchange_ptr, log_last_ptr, t, channel)
    dq_attn = cutlass.Float32(0.0)
    dq_mma = cutlass.Float32(0.0)
    dkd_attn = cutlass.Float32(0.0)
    dkd_mma = cutlass.Float32(0.0)
    dk_inv = cutlass.Float32(0.0)
    tile = cutlass.const_expr(cfg.b_t * cfg.b_t)
    for s in cutlass.range(cfg.b_t, unroll=1):
        if s != t:
            log_s = steep_log_prefix(cfg, exchange_ptr, log_last_ptr, s, channel)
            k_s = (raw_k_ptr + raw_offset(cfg, s, channel)).load().to(cutlass.Float32)
            if cutlass.const_expr(cfg.l2norm):
                k_s = k_s * (k_norm_ptr + s).load()
            if s < t:
                k_inv_s = (k_s * cute.math.exp2(log_last - log_s, fastmath=True)).to(cfg.io_dtype).to(cutlass.Float32)
                decay = cute.math.exp2(log_t - log_s, fastmath=True)
                da = (dadm_ptr + t * cfg.b_t + s).load().to(cutlass.Float32)
                dm = (dadm_ptr + tile + t * cfg.b_t + s).load().to(cutlass.Float32)
                dq_attn = dq_attn + da * (k_s * decay)
                dq_mma = dq_mma + da * k_inv_s
                dkd_attn = dkd_attn + dm * (k_s * decay)
                dkd_mma = dkd_mma + dm * k_inv_s
            else:
                q_s = (raw_q_ptr + raw_offset(cfg, s, channel)).load().to(cutlass.Float32)
                if cutlass.const_expr(cfg.l2norm):
                    q_s = q_s * (q_norm_ptr + s).load()
                decay = cute.math.exp2(log_s - log_t, fastmath=True)
                da = (dadm_ptr + s * cfg.b_t + t).load().to(cutlass.Float32)
                dm = (dadm_ptr + tile + s * cfg.b_t + t).load().to(cutlass.Float32)
                dk_inv = dk_inv + ((q_s * scale) * da - (k_s * steep_beta(cfg, beta_ptr, s, channel)) * dm) * decay
    return dq_attn, dq_mma, dk_inv, dkd_attn, dkd_mma


@cute.jit
def steep_restore_scale(cfg: cutlass.Constexpr, exchange_ptr, log_last_ptr, row, channel):
    """A steep channel's K inv / K restore scale on ``row``, ``exp2(g[BT - 1] - g[row])``, at most 1."""
    return cute.math.exp2((log_last_ptr + channel).load() - steep_log_prefix(cfg, exchange_ptr, log_last_ptr, row, channel), fastmath=True)


@cute.jit
def store_steep_beta(cfg: cutlass.Constexpr, bits, beta_regs, beta_ptr, row, lane_in_row_group):
    """Write a row's guarded beta_eff back over the raw beta tile on its steep channels, for steep_beta to read."""
    for reg_idx in cutlass.range_constexpr(cfg.d_k // 64 * 8):
        if ((bits >> reg_idx) & 1) != 0:
            channel = (reg_idx // 8) * 64 + lane_in_row_group * 8 + reg_idx % 8
            (beta_ptr + raw_offset(cfg, row, channel)).store(beta_regs[reg_idx].to(cfg.io_dtype))
