"""
reference: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens


@dataclass
class MDLMSamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 128
    max_length: int = (
        None  # There's no explicit length_limit except for the tokenizer/model context
    )
    block_size: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] | None = None
    suppress_tokens: list[int] | None = None
    begin_suppress_tokens: list[int] | None = None
    right_shift_logits: bool = False


@dataclass
class MDLMSampler(BaseSampler):
    @torch.no_grad()
    # def sample(
    def sample_original(
        self,
        inputs: list[torch.Tensor | list],
        config: MDLMSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Generate text using masked diffusion language modeling.

        Iteratively unmasks tokens over multiple diffusion steps, starting from
        fully masked sequences appended to the input prompts.

        Args:
            inputs: List of input prompts (token tensors or lists of token IDs).
            config: Sampler configuration, or None to use defaults.
            **kwargs: Override specific config parameters.

        Returns:
            BaseSamplerOutput with generated sequences, or raw tensor if return_dict=False.
        """
        if config is None:
            config = MDLMSamplerConfig()

        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )

        assert 1 <= block_size
        assert 1 <= steps
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Shape bookkeeping: per-sample prompt lengths and final canvas width -----
        # If right_shift_logits is true and a sequence has length 0, replace that sequence with [bos].
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length

        # ----- Initialize canvas with EOS, copy inputs, and append mask tail -----
        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p  # keep original prompt tokens
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = (
                mask_id  # append `max_new_tokens` masks to be generated
            )
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, pl in enumerate(prompt_lens):
            valid_end = min(pl + max_new_tokens, T)
            attention_mask[i, :valid_end] = 1

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Block scheduling over the appended mask tail -----
        num_blocks = math.ceil(max_new_tokens / block_size)
        steps = math.ceil(steps / num_blocks)  # per-block step budget
        histories = [x.clone()] if return_dict else None

        for b in range(num_blocks):
            # Build a per-sample mask *within this block* (aligned to each prompt's tail)
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=x.device
            )

            for j in range(B):
                start = prompt_lens[j] + b * block_size
                end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    width = end - start
                    block_mask_index[j, :width] = (
                        x[j, start:end] == mask_id
                    )  # which positions in this block are still masked

            # Decide how many tokens to reveal per step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            # Some steps may be skipped if there are no transfers
            effective_steps = num_transfer_tokens.size(1)

            # ----- Iterative reveal inside the current block -----
            for i in range(effective_steps):
                mask_index = x == mask_id  # current global mask map

                # Optional CFG: second forward where original prompt tokens are masked out
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask.repeat(2, 1)
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(
                        x, attention_mask=attention_mask
                    ).logits  # Use attention mask here

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # Argmax decoding with optional Gumbel-Max noise for exploration
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(
                    logits_with_noise, dim=-1
                )  # [B, T] predicted token ids

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                # Per-position confidence used to pick which masks to commit this step
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )  # [B, T] confidence of predicted token
                elif remasking == "random":
                    x0_p = torch.rand(
                        (x0.shape[0], x0.shape[1]), device=x0.device
                    )  # random scores
                else:
                    raise NotImplementedError(remasking)

                # Restrict selection window to the *current block's* tail region
                for j in range(B):
                    x0_p[j, prompt_lens[j] + (b + 1) * block_size :] = -np.inf

                # Only allow updates at currently masked positions; keep others fixed
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(
                    mask_index, x0_p, -np.inf
                )  # consider masked positions only

                # Pick exactly `num_transfer_tokens[j, i]` highest-confidence positions per sample
                transfer_index = torch.zeros_like(
                    x0, dtype=torch.bool, device=x0.device
                )
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens[j, i]
                    )
                    transfer_index[j, select_index] = True

                # Commit chosen predictions into the canvas
                x[transfer_index] = x0[transfer_index]
                if histories is not None:
                    histories.append(x.clone())

        # ----- Output format -----
        if not return_dict:
            return x
        else:
            return BaseSamplerOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(
        self, inputs: list[torch.Tensor | list], config, **kwargs
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Fill in-place the <|mdm_mask|> tokens contained in `inputs`.
        The whole (padded) sequence is split into block windows of length
        `block_size`; within each window we progressively "unmask" positions
        according to the scheduler and chosen remasking strategy.

        Notes:
        - Right padding uses EOS.
        - CFG masks out *originally known* (non-mask, non-EOS) tokens in the
        unconditional branch, identical to `generate`.
        - Only masked positions are ever updated; non-mask tokens are left intact.
        """
        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )

        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Build canvas: right-pad with EOS to the max length in the batch -----
        # If right_shift_logits is true and a sequence has length 0, replace that sequence with [bos].
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        B = len(inputs)
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)

        # Default to a single block spanning the whole sequence
        if block_size is None:
            block_size = T

        assert 1 <= block_size
        assert 1 <= steps

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, t in enumerate(inputs):
            x[i, : seq_lens[i]] = t

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, L in enumerate(seq_lens):
            if L > 0:
                attention_mask[i, :L] = 1

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Blockwise schedule over the *entire* (padded) sequence -----
        num_blocks = math.ceil(T / block_size)
        steps_per_block = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None

        for b in range(num_blocks):
            start = b * block_size
            stop = min(start + block_size, T)

            # Per-sample view of which positions in this block are masks
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=self.model.device
            )
            widths = []
            for j in range(B):
                # Width limited by sample's true length and sequence end
                width = max(0, min(seq_lens[j], stop) - start)
                widths.append(width)
                if width > 0:
                    block_mask_index[j, :width] = x[j, start : start + width] == mask_id

            # Decide how many tokens to reveal at each step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            # Some blocks may have no masks => effective_steps == 0
            effective_steps = num_transfer_tokens.size(1)

            for s in range(effective_steps):
                mask_index_full = x == mask_id

                # ----- Forward pass (+ optional CFG) -----
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask.repeat(2, 1)
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(
                        x, attention_mask=attention_mask
                    ).logits  # Use attention mask here

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # Greedy with optional Gumbel-Max noise
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, T]

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                # Confidence used for choosing which masks to commit this step
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(
                        -1
                    )  # [B, T]
                elif remasking == "random":
                    x0_p = torch.rand((B, T), device=self.model.device)
                else:
                    raise NotImplementedError(remasking)

                # Restrict selection to the *current* block only
                for j in range(B):
                    end_j = start + widths[j]
                    # Outside current block => impossible to select
                    x0_p[j, :start] = -np.inf
                    x0_p[j, end_j:] = -np.inf

                # Only consider currently-masked positions as candidates
                x0 = torch.where(mask_index_full, x0, x)
                confidence = torch.where(mask_index_full, x0_p, -np.inf)

                # Pick exactly num_transfer_tokens[j, s] positions per sample
                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                for j in range(B):
                    k = int(num_transfer_tokens[j, s].item())
                    if k > 0:
                        _, select_idx = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_idx] = True

                # Commit selected predictions into the canvas
                x[transfer_index] = x0[transfer_index]
                if histories is not None:
                    histories.append(x.clone())

        # ----- Output format -----
        if not return_dict:
            return x
        else:
            return BaseSamplerOutput(sequences=x, histories=histories)



    @torch.no_grad()
    def infill_remask_independent(
        self, inputs: list[torch.Tensor | list], config, **kwargs
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Fill in-place the <|mdm_mask|> tokens contained in `inputs`.
        The whole (padded) sequence is split into block windows of length
        `block_size`; within each window we progressively "unmask" positions
        according to the scheduler and chosen remasking strategy.

        Notes:
        - Right padding uses EOS.
        - CFG masks out *originally known* (non-mask, non-EOS) tokens in the
        unconditional branch, identical to `generate`.
        - Only masked positions are ever updated; non-mask tokens are left intact.
        """
        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )
        print(f"----------------------infill config-------------------")
        print(f"steps: {steps}, block_size: {block_size}, temperature: {temperature}, cfg_scale: {cfg_scale}, remasking: {remasking}") # 128
        mask_id = self.tokenizer.mask_token_id
        # print(mask_id) # 126336
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Build canvas: right-pad with EOS to the max length in the batch -----
        # If right_shift_logits is true and a sequence has length 0, replace that sequence with [eos].
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        B = len(inputs)
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)
        print("maximum sequence length in batch:", T)
        # Default to a single block spanning the whole sequence
        if block_size is None:
            block_size = T

        assert 1 <= block_size
        assert 1 <= steps

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, t in enumerate(inputs):
            x[i, : seq_lens[i]] = t

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, L in enumerate(seq_lens):
            if L > 0:
                attention_mask[i, :L] = 1

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Blockwise schedule over the *entire* (padded) sequence -----
        num_blocks = math.ceil(T / block_size)
        steps_per_block = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None
        print(f"number of blocks: {num_blocks}, steps per block: {steps_per_block}")
        for b in range(num_blocks):
            start = b * block_size
            stop = min(start + block_size, T)

            # Per-sample view of which positions in this block are masks
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=self.model.device
            )
            widths = []
            for j in range(B):
                # Width limited by sample's true length and sequence end
                width = max(0, min(seq_lens[j], stop) - start)
                widths.append(width)
                if width > 0:
                    block_mask_index[j, :width] = x[j, start : start + width] == mask_id

            # Decide how many tokens to reveal at each step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )
            # print("num_transfer_tokens", num_transfer_tokens) # num_transfer_tokens: tensor([[1, 1]], device='cuda:0')
            # Some blocks may have no masks => effective_steps == 0
            effective_steps = num_transfer_tokens.size(1)

            block_starts = [0 for _ in range(B)]
            block_ends = [T for j in range(B)]

            for s in range(effective_steps):
                mask_index_full = x == mask_id

                # ----- Forward pass (+ optional CFG) -----
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask
                    ).logits  # Use attention mask here
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    # print("x.shape:", x.shape) # x.shape: torch.Size([1, 82])
                    logits = self.model(
                        x, attention_mask=attention_mask
                    ).logits  # Use attention mask here

                # print("logits shape:", logits.shape) # logits shape: torch.Size([1, 55, 126464])
                # print("suppress_tokens:", suppress_tokens) # suppress_tokens: None
                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf
                # print("right_shift_logits:", right_shift_logits) # right_shift_logits: False
                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # Greedy with optional Gumbel-Max noise
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, T]
                x0_raw = torch.argmax(logits_with_noise, dim=-1)  # save raw predictions BEFORE the where-clamp

                # print("x0 shape:", x0.shape) # x0 shape: torch.Size([1, 55])
                # print("begin_suppress_tokens:", begin_suppress_tokens) # begin_suppress_tokens: None
                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                # Confidence used for choosing which masks to commit this step
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(
                        -1
                    )  # [B, T]
                elif remasking == "random":
                    x0_p = torch.rand((B, T), device=self.model.device)
                else:
                    raise NotImplementedError(remasking)

                # Restrict selection to the *current* block only
                for j in range(B):
                    end_j = start + widths[j]
                    # Outside current block => impossible to select
                    x0_p[j, :start] = -np.inf
                    x0_p[j, end_j:] = -np.inf

                
                # Only consider currently-masked positions as candidates
                x0 = torch.where(mask_index_full, x0, x)
                confidence = torch.where(mask_index_full, x0_p, -np.inf)

                # Pick exactly num_transfer_tokens[j, s] positions per sample
                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                for j in range(B):
                    k = int(num_transfer_tokens[j, s].item())
                    if k > 0:
                        _, select_idx = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_idx] = True
                        
                x[transfer_index] = x0[transfer_index]

                # x = x0_raw.clone()  # <-- REMASK ALL OTHER POSITIONS INDEPENDENTLY OF THE BLOCK SCHEDULE

                if histories is not None:
                    histories.append(x.clone())
            # ===== PHASE 2: Multi-pass revision after block is fully unmasked =====
            MAX_REVISIONS = 8
            CONFIDENCE_THRESHOLD = 0.0
            MAX_REVISION_PASSES = 20
            revised_and_refilled = torch.zeros_like(x, dtype=torch.bool)  # ← uncomment
    
            # precompute per-sample block boundaries
            block_starts = [0 for _ in range(B)]
            block_ends = [T for j in range(B)]
            forbidden_source_ids = [198, 13444, 14975, 126081, 91, 126080, 20679, 7351, 486, 27, 2983, 95591, 114654, 3583, 797, 3840, 68, 335, 598]
            forbidden_target_ids = [198, 13444, 14975, 126081, 91, 126080, 20679, 7351, 486, 27, 2983, 95591, 114654, 3583, 797, 3840, 68, 335, 598]

            for rev_pass in range(MAX_REVISION_PASSES):
                pre_remask_tokens = x.clone()  # ← ADD THIS LINE
                logits = self.model(x, attention_mask=attention_mask, revise_step=True, block_starts=block_starts, block_ends=block_ends).logits
                p = F.softmax(logits, dim=-1)
                x0_rev = torch.argmax(logits, dim=-1)
                x0_p_rev = torch.gather(p, dim=-1, index=x0_rev.unsqueeze(-1)).squeeze(-1)

                revision_index = torch.zeros_like(x, dtype=torch.bool)

                for j in range(B):
                    block_start = block_starts[j]  # ← correct
                    block_end = block_ends[j]      # ← correct

                    candidate_mask = torch.zeros(T, dtype=torch.bool, device=x.device)
                    candidate_mask[block_start:block_end] = True
                    candidate_mask = candidate_mask & ~revised_and_refilled[j]

                    if remasking == "low_confidence":
                        disagree = candidate_mask & (x0_rev[j] != x[j]) & (x0_p_rev[j] > CONFIDENCE_THRESHOLD)
                    else:
                        disagree = candidate_mask & (x0_rev[j] != x[j])

                    candidate_positions = disagree.nonzero(as_tuple=True)[0]
                    if len(candidate_positions) == 0:
                        continue

                    keep = torch.ones(len(candidate_positions), dtype=torch.bool)
                    for idx, pos in enumerate(candidate_positions):
                        src = x[j, pos].item()
                        tgt = x0_rev[j, pos].item()
                        if src in forbidden_source_ids or tgt in forbidden_target_ids:
                            keep[idx] = False
                    candidate_positions = candidate_positions[keep]
                    if len(candidate_positions) == 0:
                        continue

                    candidate_confidences = x0_p_rev[j, candidate_positions]

                    rows = []
                    for pos, new_conf in zip(candidate_positions, candidate_confidences):
                        src_tid = x[j, pos].item()
                        tgt_tid = x0_rev[j, pos].item()

                        src_tok = self.tokenizer.decode([src_tid])
                        tgt_tok = self.tokenizer.decode([tgt_tid])

                        current_conf = p[j, pos, src_tid].item()

                        rows.append((
                            pos.item(),
                            src_tok, src_tid,
                            current_conf,
                            tgt_tok, tgt_tid,
                            new_conf.item()
                        ))

                    # Sort EXACTLY like selection logic
                    rows.sort(key=lambda r: (-r[3] + r[6] * 1e-6), reverse=True)

                    print(f"[Block {b} Revision pass {rev_pass}]")
                    for pos, src_tok, src_tid, current_conf, tgt_tok, tgt_tid, new_conf in rows:
                        print(
                            f"  pos {pos}: '{src_tok}({src_tid})' p={current_conf:.3f} "
                            f"→ '{tgt_tok}({tgt_tid})' p={new_conf:.3f}"
                        )
                    '''
                    k = min(MAX_REVISIONS, len(candidate_positions))
                    _, top_k = torch.topk(candidate_confidences, k=k)
                    revision_index[j, candidate_positions[top_k]] = True
                    '''
                    # Sort by: (1) lowest current token confidence, (2) highest new token confidence as tiebreak
                    current_confidences = torch.tensor(
                        [p[j, pos, x[j, pos]].item() for pos in candidate_positions],
                        device=x.device
                    )
                    new_confidences = x0_p_rev[j, candidate_positions]

                    # Combined score: lowest current conf first (-current_conf), tiebreak by highest new conf
                    combined_scores = -current_confidences + new_confidences * 1e-6

                    k = min(MAX_REVISIONS, len(candidate_positions))
                    _, top_k = torch.topk(combined_scores, k=k)
                    revision_index[j, candidate_positions[top_k]] = True

                if not revision_index.any():
                    print(f"[Block {b}] Revision converged after {rev_pass + 1} pass(es)")
                    break

                x[revision_index] = mask_id
                if histories is not None:
                    histories.append(x.clone())

                # Re-fill only within current block
                refill_count = 0
                refill_logits = self.model(x, attention_mask=attention_mask).logits
                x0_refill = torch.argmax(refill_logits, dim=-1)
                for j in range(B):
                    block_start = block_starts[j]  # ← correct
                    block_end = block_ends[j]      # ← correct
                    for pos in range(block_start, block_end):
                        if x[j, pos] == mask_id:
                            # REPLACE WITH THIS:
                            new_token = x0_refill[j, pos].item()
                            original_token = pre_remask_tokens[j, pos].item()
                            x[j, pos] = new_token
                            if new_token == original_token:
                                revised_and_refilled[j, pos] = True
                            refill_count += 1
                            print("Change: ", self.tokenizer.decode([original_token]), " -->  ", self.tokenizer.decode([new_token]))
                if histories is not None:
                    histories.append(x.clone())
                print(f"[Block {b}] Refilled {refill_count} re-masked tokens")  # ← correct count
                
      
        # ----- Output format -----
        if not return_dict:
            return x
        else:
            return BaseSamplerOutput(sequences=x, histories=histories)


    @torch.no_grad()
    # def sample(
    def sampling_revising_by_remasking(
        self,
        inputs: list[torch.Tensor | list],
        config: MDLMSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Generate text using masked diffusion language modeling.

        Iteratively unmasks tokens over multiple diffusion steps, starting from
        fully masked sequences appended to the input prompts.

        Args:
            inputs: List of input prompts (token tensors or lists of token IDs).
            config: Sampler configuration, or None to use defaults.
            **kwargs: Override specific config parameters.

        Returns:
            SamplerOutput with generated sequences, or raw tensor if return_dict=False.
        """
        if config is None:
            config = MDLMSamplerConfig()

        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )

        assert 1 <= block_size
        assert 1 <= steps
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Shape bookkeeping: per-sample prompt lengths and final canvas width -----
        # If right_shift_logits is true and a sequence has length 0, replace that sequence with [eos].
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length

        # print(f"----------------------T, max_length, max_new_tokens-------------------")
        # print(T, max_length, max_new_tokens) # 171 171 128
        # print(f"----------------------inputs-------------------")
        # print(len(inputs), inputs[0].shape, inputs) # 1 torch.Size([43]) [tensor([  1,  38,  39,  ...,  13,  11])]


        # ----- Initialize canvas with EOS, copy inputs, and append mask tail -----
        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p  # keep original prompt tokens
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = (
                mask_id  # append `max_new_tokens` masks to be generated
            )

        # print(f"----------------------x-------------------")
        # print(x.shape, x) # torch.Size([1, 171]) tensor([[  1,  38,  39,  ..., 50257, 50257, 50257]])

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, pl in enumerate(prompt_lens):
            valid_end = min(pl + max_new_tokens, T)
            attention_mask[i, :valid_end] = 1

        # print(f"----------------------attention_mask-------------------")
        # print(attention_mask.shape, attention_mask) # torch.Size([1, 171]) tensor([[1, 1, 1,  ..., 1, 1, 1]])

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # print(f"----------------------unmasked_index-------------------")
        # print(unmasked_index.shape, unmasked_index) # torch.Size([1, 171]) tensor([[ True,  True,  True,  ..., False, False, False]])

        # ----- Block scheduling over the appended mask tail -----
        num_blocks = math.ceil(max_new_tokens / block_size)
        steps = math.ceil(steps / num_blocks)  # per-block step budget
        histories = [x.clone()] if return_dict else None

        # print(f"----------------------block_size, num_blocks, steps-------------------")
        # print(block_size, num_blocks, steps) # 32 4 32

        for b in range(num_blocks):
            # Build a per-sample mask *within this block* (aligned to each prompt's tail)
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=x.device
            )
            widths = []
            block_starts = []
            block_ends = []
            for j in range(B):
                start = prompt_lens[j] + b * block_size
                end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
                width = max(0, end - start)
                widths.append(width)
                block_starts.append(start)
                block_ends.append(end)
                if width > 0:
                    block_mask_index[j, :width] = x[j, start:end] == mask_id

            # Decide how many tokens to reveal per step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )
            # print(f"----------------------num_transfer_tokens (block {b})-------------------")
            # print(num_transfer_tokens.shape, num_transfer_tokens) # torch.Size([1, 32]) tensor([[1,  ..., 1, 1, 1]])

            # Some steps may be skipped if there are no transfers
            effective_steps = num_transfer_tokens.size(1)
            # ----- Iterative reveal inside the current block -----
            for i in range(effective_steps):
                mask_index = x == mask_id  # current global mask map

                # print(f"----------------------cfg_scale-------------------")
                # print(cfg_scale) # 0.0
                # Optional CFG: second forward where original prompt tokens are masked out
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask
                    ).logits  # Use attention mask here
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(
                        x, attention_mask=attention_mask
                    ).logits  # Use attention mask here

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                # print("right_shift_logits", right_shift_logits) # False
                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # Argmax decoding with optional Gumbel-Max noise for exploration
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(
                    logits_with_noise, dim=-1
                )  # [B, T] predicted token ids
                x0_raw = torch.argmax(logits_with_noise, dim=-1)  # save raw predictions BEFORE the where-clamp

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                # Per-position confidence used to pick which masks to commit this step
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )  # [B, T] confidence of predicted token
                elif remasking == "random":
                    x0_p = torch.rand(
                        (x0.shape[0], x0.shape[1]), device=x0.device
                    )  # random scores
                else:
                    raise NotImplementedError(remasking)

                # Restrict selection window to the *current block's* tail region
                for j in range(B):
                    x0_p[j, prompt_lens[j] + (b + 1) * block_size :] = -np.inf

                # Only allow updates at currently masked positions; keep others fixed
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(
                    mask_index, x0_p, -np.inf
                )  # consider masked positions only

                # Pick exactly `num_transfer_tokens[j, i]` highest-confidence positions per sample
                transfer_index = torch.zeros_like(
                    x0, dtype=torch.bool, device=x0.device
                )
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens[j, i]
                    )
                    transfer_index[j, select_index] = True

                # Commit chosen predictions into the canvas
                x[transfer_index] = x0[transfer_index]
                if histories is not None:
                    histories.append(x.clone())


            # ===== PHASE 2: Multi-pass revision after block is fully unmasked =====
            MAX_REVISIONS = 8
            CONFIDENCE_THRESHOLD = 0.0
            MAX_REVISION_PASSES = 40
            revised_and_refilled = torch.zeros_like(x, dtype=torch.bool)  # ← uncomment
    
            # # precompute per-sample block boundaries
            # block_starts = [b * block_size for _ in range(B)]
            # block_ends = [min(b * block_size + widths[j], T) for j in range(B)]

            forbidden_source_ids = [198, 13444, 14975, 126081, 91, 126080, 20679, 7351, 486, 27, 2983, 95591, 114654, 3583, 797, 3840, 68, 335, 598]
            forbidden_target_ids = [198, 13444, 14975, 126081, 91, 126080, 20679, 7351, 486, 27, 2983, 95591, 114654, 3583, 797, 3840, 68, 335, 598]

            for rev_pass in range(MAX_REVISION_PASSES):
                pre_remask_tokens = x.clone()  # ← ADD THIS LINE
                logits = self.model(x, attention_mask=attention_mask, revise_step=True, block_starts=block_starts, block_ends=block_ends).logits
                p = F.softmax(logits, dim=-1)
                x0_rev = torch.argmax(logits, dim=-1)
                x0_p_rev = torch.gather(p, dim=-1, index=x0_rev.unsqueeze(-1)).squeeze(-1)

                revision_index = torch.zeros_like(x, dtype=torch.bool)

                for j in range(B):
                    block_start = block_starts[j]  # ← correct
                    block_end = block_ends[j]      # ← correct

                    candidate_mask = torch.zeros(T, dtype=torch.bool, device=x.device)
                    candidate_mask[block_start:block_end] = True
                    candidate_mask = candidate_mask & ~revised_and_refilled[j]

                    if remasking == "low_confidence":
                        disagree = candidate_mask & (x0_rev[j] != x[j]) & (x0_p_rev[j] > CONFIDENCE_THRESHOLD)
                    else:
                        disagree = candidate_mask & (x0_rev[j] != x[j])

                    candidate_positions = disagree.nonzero(as_tuple=True)[0]
                    if len(candidate_positions) == 0:
                        continue

                    keep = torch.ones(len(candidate_positions), dtype=torch.bool)
                    for idx, pos in enumerate(candidate_positions):
                        src = x[j, pos].item()
                        tgt = x0_rev[j, pos].item()
                        if src in forbidden_source_ids or tgt in forbidden_target_ids:
                            keep[idx] = False
                    candidate_positions = candidate_positions[keep]
                    if len(candidate_positions) == 0:
                        continue

                    candidate_confidences = x0_p_rev[j, candidate_positions]

                    rows = []
                    for pos, new_conf in zip(candidate_positions, candidate_confidences):
                        src_tid = x[j, pos].item()
                        tgt_tid = x0_rev[j, pos].item()

                        src_tok = self.tokenizer.decode([src_tid])
                        tgt_tok = self.tokenizer.decode([tgt_tid])

                        current_conf = p[j, pos, src_tid].item()

                        rows.append((
                            pos.item(),
                            src_tok, src_tid,
                            current_conf,
                            tgt_tok, tgt_tid,
                            new_conf.item()
                        ))

                    # Sort EXACTLY like selection logic
                    rows.sort(key=lambda r: (-r[3] + r[6] * 1e-6), reverse=True)

                    print(f"[Block {b} Revision pass {rev_pass}]")
                    for pos, src_tok, src_tid, current_conf, tgt_tok, tgt_tid, new_conf in rows:
                        print(
                            f"  pos {pos}: '{src_tok}({src_tid})' p={current_conf:.3f} "
                            f"→ '{tgt_tok}({tgt_tid})' p={new_conf:.3f}"
                        )
                    '''
                    k = min(MAX_REVISIONS, len(candidate_positions))
                    _, top_k = torch.topk(candidate_confidences, k=k)
                    revision_index[j, candidate_positions[top_k]] = True
                    '''
                    # Sort by: (1) lowest current token confidence, (2) highest new token confidence as tiebreak
                    current_confidences = torch.tensor(
                        [p[j, pos, x[j, pos]].item() for pos in candidate_positions],
                        device=x.device
                    )
                    new_confidences = x0_p_rev[j, candidate_positions]

                    # Combined score: lowest current conf first (-current_conf), tiebreak by highest new conf
                    combined_scores = -current_confidences + new_confidences * 1e-6

                    k = min(MAX_REVISIONS, len(candidate_positions))
                    _, top_k = torch.topk(combined_scores, k=k)
                    revision_index[j, candidate_positions[top_k]] = True

                if not revision_index.any():
                    print(f"[Block {b}] Revision converged after {rev_pass + 1} pass(es)")
                    break

                x[revision_index] = mask_id
                if histories is not None:
                    histories.append(x.clone())

                # Re-fill only within current block
                refill_count = 0
                refill_logits = self.model(x, attention_mask=attention_mask).logits
                x0_refill = torch.argmax(refill_logits, dim=-1)
                for j in range(B):
                    block_start = block_starts[j]  # ← correct
                    block_end = block_ends[j]      # ← correct
                    for pos in range(block_start, block_end):
                        if x[j, pos] == mask_id:
                            # REPLACE WITH THIS:
                            new_token = x0_refill[j, pos].item()
                            original_token = pre_remask_tokens[j, pos].item()
                            x[j, pos] = new_token
                            if new_token == original_token:
                                revised_and_refilled[j, pos] = True
                            refill_count += 1
                            print("Change: ", self.tokenizer.decode([original_token]), " -->  ", self.tokenizer.decode([new_token]))
                if histories is not None:
                    histories.append(x.clone())
                print(f"[Block {b}] Refilled {refill_count} re-masked tokens")  # ← correct count
                
      

        if not return_dict:
            return x
        else:
            return BaseSamplerOutput(sequences=x, histories=histories)





    @torch.no_grad()
    # def sampling_revising_by_gradualy_remasking(
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: MDLMSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:

        if config is None:
            config = MDLMSamplerConfig()

        # ============================================================
        # Config
        # ============================================================
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get("begin_suppress_tokens", config.begin_suppress_tokens)
       

        # ============================================================
        # Gradual remasking hyperparameters
        # ============================================================

        K_REVEAL = kwargs.get("k_reveal", 2)
        R_REMASK = kwargs.get("r_remask", 1)
        MAX_REMASK_PER_POS = kwargs.get("max_remask_per_pos", 3,)
        REVISION_THRESHOLD = kwargs.get("revision_threshold", 0.0,)
        assert K_REVEAL > R_REMASK, ("Need K_REVEAL > R_REMASK for guaranteed convergence")
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ============================================================
        # Input preprocessing
        # ============================================================

        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p
                for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [torch.as_tensor(p, dtype=torch.long, device=self.model.device,) for p in inputs]

        prompt_lens = [p.shape[0] for p in inputs]
        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)
        B = len(inputs)
        T = max_length

        # ============================================================
        # Initialize canvas
        # ============================================================
        x = torch.full((B, T), eos_id,dtype=torch.long, device=self.model.device,)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens,] = mask_id

        
        attention_mask = torch.zeros(
            (B, T),
            dtype=torch.long,
            device=self.model.device,
        )

        for i, pl in enumerate(prompt_lens):
            valid_end = min(pl + max_new_tokens, T)
            attention_mask[i, :valid_end] = 1

        unmasked_index = ((x != mask_id) & attention_mask.bool())
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x,
                torch.as_tensor(
                    cfg_keep_tokens,
                    device=self.model.device,
                ),
            )
            unmasked_index = (unmasked_index & ~keep_mask)

        histories = [x.clone()] if return_dict else None
        num_blocks = math.ceil(max_new_tokens / block_size)

        for b in range(num_blocks):
            print(f"\n\n############################################################")
            print(f"################## START BLOCK {b} #########################")
            print(f"############################################################\n")
            block_starts = []
            block_ends = []
            for j in range(B):
                start = prompt_lens[j] + b * block_size
                end = min( start + block_size, prompt_lens[j] + max_new_tokens, T,)
                block_starts.append(start)
                block_ends.append(end)

            remask_counter = torch.zeros_like(x, dtype=torch.long,)
            iteration = 0
            number_remask = 0
            while True:
                iteration += 1
                block_mask = torch.zeros_like(x, dtype=torch.bool,)
                for j in range(B):
                    block_mask[j, block_starts[j]:block_ends[j],] = True
                masked_positions = ((x == mask_id) & block_mask)

                if not masked_positions.any():
                    print(f"\n================ BLOCK {b} FINISHED ================")
                    for j in range(B):
                        decoded = self.tokenizer.decode(x[j, :block_ends[j]].tolist())
                        print(f"[Sample {j}] Final block text:")
                        print(decoded)
                        print()
                    break

                print(f"\n------------------------------------------------------------")
                print(f"BLOCK {b} | ITERATION {iteration}")
                print(f"Remaining masks: {masked_positions.sum().item()}")
                print(f"------------------------------------------------------------")

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(x_, attention_mask=attention_mask,).logits
                    logits, un_logits = torch.chunk( logits, 2, dim=0,)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x,attention_mask=attention_mask,).logits

               
                if suppress_tokens is not None:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf
                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1,)

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature,)
                pred_tokens = torch.argmax(logits_with_noise, dim=-1,)
                probs = F.softmax(logits, dim=-1,)
                pred_confidence = torch.gather(probs, dim=-1, index=pred_tokens.unsqueeze(-1),).squeeze(-1)
                reveal_scores = torch.where(masked_positions, pred_confidence, -torch.inf,)
                reveal_index = torch.zeros_like(x, dtype=torch.bool,)

                for j in range(B):
                    num_masks = (masked_positions[j].sum().item())
                    if num_masks == 0:
                        continue
                    k = min(K_REVEAL, num_masks)
                    _, idx = torch.topk(reveal_scores[j], k=k,)
                    reveal_index[j, idx] = True
                
                '''
                print(f"\n================ BLOCK {b} : REVEAL =================")
                for j in range(B):
                    selected_positions = (reveal_index[j].nonzero(as_tuple=True)[0])
                    if len(selected_positions) == 0:
                        continue
                    print(f"[Sample {j}]")
                    for pos in selected_positions:
                        pos = pos.item()
                        token_id = pred_tokens[j, pos].item()
                        token_str = self.tokenizer.decode([token_id])
                        conf = pred_confidence[j, pos].item()
                        print(f"  Reveal pos={pos:3d} "f"token='{token_str}' "f"id={token_id} "f"conf={conf:.4f}")
                '''
                
                x[reveal_index] = pred_tokens[reveal_index]
                if histories is not None:
                    histories.append(x.clone())

                # ====================================================
                # REVISE FORWARD
                # ====================================================
                filled_positions = ((x != mask_id) & block_mask)
                
                revise_logits = self.model(x, attention_mask=attention_mask, revise_step=True, block_starts=block_starts, block_ends=block_ends).logits
                revise_probs = F.softmax(revise_logits, dim=-1,)
                revise_tokens = torch.argmax(revise_logits, dim=-1,)
                revise_confidence = torch.gather(revise_probs, dim=-1, index=revise_tokens.unsqueeze(-1),).squeeze(-1)

                
                revision_candidates = (filled_positions & (revise_tokens != x))
                current_token_conf = torch.gather(revise_probs, dim=-1, index=x.unsqueeze(-1), ).squeeze(-1)
                revision_gain = (revise_confidence - current_token_conf)
                revision_candidates = (revision_candidates & (revision_gain > REVISION_THRESHOLD))
                revision_scores = torch.where(revision_candidates, revision_gain, -torch.inf,)
                remask_index = torch.zeros_like(x, dtype=torch.bool, )

                # ====================================================
                # REMASK SELECTION
                # ====================================================
                for j in range(B):
                    candidate_positions = (revision_candidates[j].nonzero(as_tuple=True)[0])
                    if len(candidate_positions) == 0:
                        continue
                    valid_mask = (remask_counter[j, candidate_positions,] < MAX_REMASK_PER_POS)
                    candidate_positions = candidate_positions[valid_mask]

                    if len(candidate_positions) == 0:
                        continue

                    current_masks = (masked_positions[j].sum().item())
                    r = min(R_REMASK, max(0, current_masks // 2),)
                    if r == 0:
                        continue
                    candidate_scores = revision_scores[j, candidate_positions,]
                    _, top_idx = torch.topk(candidate_scores, k=min(r, len(candidate_positions)),)
                    selected = candidate_positions[top_idx]
                    remask_index[j, selected] = True

                if remask_index.any():
                    print(f"\n================ BLOCK {b} : REMASK =================")
                else:
                    print("remask_index is empty", remask_index.sum())

                for j in range(B):
                    selected_positions = (remask_index[j].nonzero(as_tuple=True)[0])
                    if len(selected_positions) == 0:
                        continue
                        
                    print(f"[Sample {j}]")
                    for pos in selected_positions:
                        pos = pos.item()
                        old_token_id = x[j, pos].item()
                        old_token_str = self.tokenizer.decode([old_token_id])
                        new_token_id = revise_tokens[j, pos].item()
                        new_token_str = self.tokenizer.decode([new_token_id])
                        current_conf = (current_token_conf[j, pos].item())
                        revise_conf = (revise_confidence[j, pos].item())
                        gain = revision_gain[j, pos].item()
                        print(
                            f"  Remask pos={pos:3d} "
                            f"current='{old_token_str}'({old_token_id}) "
                            f"p={current_conf:.4f} "
                            f"--> revise='{new_token_str}'({new_token_id}) "
                            f"p={revise_conf:.4f} "
                            f"gain={gain:.4f}"
                        )
                number_remask += remask_index.sum()
                x[remask_index] = mask_id
                remask_counter[remask_index] += 1
                if histories is not None:
                    histories.append(x.clone())


                '''
                print(f"\n================ CURRENT TEXT =================")
                for j in range(B):
                    decoded = self.tokenizer.decode(x[j, :block_ends[j]].tolist())
                    print(f"[Sample {j}]")
                    print(decoded)
                    print()
                '''

        print("total_number_remask", number_remask)
                
        if not return_dict:
            return x

        return BaseSamplerOutput(
            sequences=x,
            histories=histories,
        )





    # @torch.no_grad()
    # def sampling_revising_by_gradualy_remasking(
    #     self,
    #     inputs: list[torch.Tensor | list],
    #     config: MDLMSamplerConfig | None = None,
    #     **kwargs,
    # ) -> BaseSamplerOutput | torch.Tensor:

    #     if config is None:
    #         config = MDLMSamplerConfig()

    #     # ============================================================
    #     # Config
    #     # ============================================================

    #     steps = kwargs.get("steps", config.steps)
    #     max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
    #     max_length = kwargs.get("max_length", config.max_length)
    #     block_size = kwargs.get("block_size", config.block_size)

    #     temperature = kwargs.get("temperature", config.temperature)

    #     cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
    #     cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)

    #     remasking = kwargs.get("remasking", config.remasking)

    #     suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)

    #     stochastic_transfer = kwargs.get(
    #         "stochastic_transfer",
    #         config.stochastic_transfer,
    #     )

    #     return_dict = kwargs.get("return_dict", config.return_dict)

    #     right_shift_logits = kwargs.get(
    #         "right_shift_logits",
    #         config.right_shift_logits,
    #     )

    #     begin_suppress_tokens = kwargs.get(
    #         "begin_suppress_tokens",
    #         config.begin_suppress_tokens,
    #     )

    #     # ============================================================
    #     # New gradual remasking hyperparameters
    #     # ============================================================

    #     K_REVEAL = kwargs.get("k_reveal", 4)
    #     R_REMASK = kwargs.get("r_remask", 1)

    #     MAX_REMASK_PER_POS = kwargs.get(
    #         "max_remask_per_pos",
    #         2,
    #     )

    #     REVISION_THRESHOLD = kwargs.get(
    #         "revision_threshold",
    #         0.0,
    #     )

    #     assert K_REVEAL > R_REMASK, (
    #         "Need K_REVEAL > R_REMASK for guaranteed convergence"
    #     )

    #     mask_id = self.tokenizer.mask_token_id
    #     bos_id = self.tokenizer.bos_token_id
    #     eos_id = self.tokenizer.eos_token_id

    #     # ============================================================
    #     # Input preprocessing
    #     # ============================================================

    #     if right_shift_logits:
    #         inputs = [
    #             [bos_id] if isinstance(p, list) and len(p) == 0 else p
    #             for p in inputs
    #         ]

    #     if isinstance(inputs[0], list):
    #         inputs = [
    #             torch.as_tensor(
    #                 p,
    #                 dtype=torch.long,
    #                 device=self.model.device,
    #             )
    #             for p in inputs
    #         ]

    #     prompt_lens = [p.shape[0] for p in inputs]

    #     if max_new_tokens:
    #         max_length = max_new_tokens + max(prompt_lens)
    #     else:
    #         max_new_tokens = max_length - max(prompt_lens)

    #     B = len(inputs)
    #     T = max_length

    #     # ============================================================
    #     # Initialize sequence canvas
    #     # ============================================================

    #     x = torch.full(
    #         (B, T),
    #         eos_id,
    #         dtype=torch.long,
    #         device=self.model.device,
    #     )

    #     for i, p in enumerate(inputs):

    #         x[i, : prompt_lens[i]] = p

    #         x[
    #             i,
    #             prompt_lens[i] : prompt_lens[i] + max_new_tokens,
    #         ] = mask_id

    #     # ============================================================
    #     # Attention mask
    #     # ============================================================

    #     attention_mask = torch.zeros(
    #         (B, T),
    #         dtype=torch.long,
    #         device=self.model.device,
    #     )

    #     for i, pl in enumerate(prompt_lens):

    #         valid_end = min(pl + max_new_tokens, T)

    #         attention_mask[i, :valid_end] = 1

    #     # ============================================================
    #     # CFG bookkeeping
    #     # ============================================================

    #     unmasked_index = (x != mask_id) & attention_mask.bool()

    #     if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):

    #         keep_mask = torch.isin(
    #             x,
    #             torch.as_tensor(
    #                 cfg_keep_tokens,
    #                 device=self.model.device,
    #             ),
    #         )

    #         unmasked_index = unmasked_index & ~keep_mask

    #     # ============================================================
    #     # Histories
    #     # ============================================================

    #     histories = [x.clone()] if return_dict else None

    #     # ============================================================
    #     # Block scheduling
    #     # ============================================================

    #     num_blocks = math.ceil(max_new_tokens / block_size)

    #     # ============================================================
    #     # Main block loop
    #     # ============================================================

    #     for b in range(num_blocks):

    #         # --------------------------------------------------------
    #         # Block boundaries
    #         # --------------------------------------------------------

    #         block_starts = []
    #         block_ends = []

    #         for j in range(B):

    #             start = prompt_lens[j] + b * block_size

    #             end = min(
    #                 start + block_size,
    #                 prompt_lens[j] + max_new_tokens,
    #                 T,
    #             )

    #             block_starts.append(start)
    #             block_ends.append(end)

    #         # --------------------------------------------------------
    #         # Remask counter to prevent oscillation
    #         # --------------------------------------------------------

    #         remask_counter = torch.zeros_like(
    #             x,
    #             dtype=torch.long,
    #         )

    #         # ========================================================
    #         # Unified reveal + revise loop
    #         # ========================================================

    #         while True:

    #             # ----------------------------------------------------
    #             # Current block mask
    #             # ----------------------------------------------------

    #             block_mask = torch.zeros_like(
    #                 x,
    #                 dtype=torch.bool,
    #             )

    #             for j in range(B):

    #                 block_mask[
    #                     j,
    #                     block_starts[j]:block_ends[j],
    #                 ] = True

    #             masked_positions = (
    #                 (x == mask_id)
    #                 & block_mask
    #             )

    #             # ----------------------------------------------------
    #             # Convergence
    #             # ----------------------------------------------------

    #             if not masked_positions.any():
    #                 break

    #             # ====================================================
    #             # FORWARD PREDICTION
    #             # ====================================================

    #             if cfg_scale > 0.0:

    #                 un_x = x.clone()

    #                 un_x[unmasked_index] = mask_id

    #                 x_ = torch.cat([x, un_x], dim=0)

    #                 logits = self.model(
    #                     x_,
    #                     attention_mask=attention_mask,
    #                 ).logits

    #                 logits, un_logits = torch.chunk(
    #                     logits,
    #                     2,
    #                     dim=0,
    #                 )

    #                 logits = un_logits + (
    #                     cfg_scale + 1
    #                 ) * (logits - un_logits)

    #             else:

    #                 logits = self.model(
    #                     x,
    #                     attention_mask=attention_mask,
    #                 ).logits

    #             # ----------------------------------------------------
    #             # Token suppression
    #             # ----------------------------------------------------

    #             if suppress_tokens is not None:

    #                 for token_id in suppress_tokens:
    #                     logits[:, :, token_id] = -torch.inf

    #             if right_shift_logits:

    #                 logits = torch.cat(
    #                     [logits[:, :1], logits[:, :-1]],
    #                     dim=1,
    #                 )

    #             logits_with_noise = add_gumbel_noise(
    #                 logits,
    #                 temperature=temperature,
    #             )

    #             pred_tokens = torch.argmax(
    #                 logits_with_noise,
    #                 dim=-1,
    #             )

    #             probs = F.softmax(logits, dim=-1)

    #             pred_confidence = torch.gather(
    #                 probs,
    #                 dim=-1,
    #                 index=pred_tokens.unsqueeze(-1),
    #             ).squeeze(-1)

    #             # ====================================================
    #             # REVEAL STEP
    #             # ====================================================

    #             reveal_scores = torch.where(
    #                 masked_positions,
    #                 pred_confidence,
    #                 -torch.inf,
    #             )

    #             reveal_index = torch.zeros_like(
    #                 x,
    #                 dtype=torch.bool,
    #             )

    #             for j in range(B):

    #                 num_masks = masked_positions[j].sum().item()

    #                 if num_masks == 0:
    #                     continue

    #                 k = min(K_REVEAL, num_masks)

    #                 _, idx = torch.topk(
    #                     reveal_scores[j],
    #                     k=k,
    #                 )

    #                 reveal_index[j, idx] = True

    #             x[reveal_index] = pred_tokens[reveal_index]

    #             if histories is not None:
    #                 histories.append(x.clone())

    #             # ====================================================
    #             # REVISE FORWARD
    #             # ====================================================

    #             revise_logits = self.model(
    #                 x,
    #                 attention_mask=attention_mask,
    #                 revise_step=True,
    #                 block_starts=block_starts,
    #                 block_ends=block_ends,
    #             ).logits

    #             revise_probs = F.softmax(
    #                 revise_logits,
    #                 dim=-1,
    #             )

    #             revise_tokens = torch.argmax(
    #                 revise_logits,
    #                 dim=-1,
    #             )

    #             revise_confidence = torch.gather(
    #                 revise_probs,
    #                 dim=-1,
    #                 index=revise_tokens.unsqueeze(-1),
    #             ).squeeze(-1)

    #             # ====================================================
    #             # SELECT REMASK CANDIDATES
    #             # ====================================================

    #             filled_positions = (
    #                 (x != mask_id)
    #                 & block_mask
    #             )

    #             revision_candidates = (
    #                 filled_positions
    #                 & (revise_tokens != x)
    #             )

    #             # ----------------------------------------------------
    #             # Current token confidence
    #             # ----------------------------------------------------

    #             current_token_conf = torch.gather(
    #                 revise_probs,
    #                 dim=-1,
    #                 index=x.unsqueeze(-1),
    #             ).squeeze(-1)

    #             # ----------------------------------------------------
    #             # Revision gain score
    #             # ----------------------------------------------------

    #             revision_gain = (
    #                 revise_confidence
    #                 - current_token_conf
    #             )

    #             revision_candidates = (
    #                 revision_candidates
    #                 & (revision_gain > REVISION_THRESHOLD)
    #             )

    #             revision_scores = torch.where(
    #                 revision_candidates,
    #                 revision_gain,
    #                 -torch.inf,
    #             )

    #             remask_index = torch.zeros_like(
    #                 x,
    #                 dtype=torch.bool,
    #             )

    #             # ====================================================
    #             # REMASK SELECTION
    #             # ====================================================

    #             for j in range(B):

    #                 candidate_positions = (
    #                     revision_candidates[j]
    #                     .nonzero(as_tuple=True)[0]
    #                 )

    #                 if len(candidate_positions) == 0:
    #                     continue

    #                 # ------------------------------------------------
    #                 # Prevent infinite oscillation
    #                 # ------------------------------------------------

    #                 valid_mask = (
    #                     remask_counter[
    #                         j,
    #                         candidate_positions,
    #                     ]
    #                     < MAX_REMASK_PER_POS
    #                 )

    #                 candidate_positions = candidate_positions[
    #                     valid_mask
    #                 ]

    #                 if len(candidate_positions) == 0:
    #                     continue

    #                 # ------------------------------------------------
    #                 # Adaptive remasking
    #                 # ------------------------------------------------

    #                 current_masks = (
    #                     masked_positions[j]
    #                     .sum()
    #                     .item()
    #                 )

    #                 r = min(
    #                     R_REMASK,
    #                     max(0, current_masks // 2),
    #                 )

    #                 if r == 0:
    #                     continue

    #                 candidate_scores = revision_scores[
    #                     j,
    #                     candidate_positions,
    #                 ]

    #                 _, top_idx = torch.topk(
    #                     candidate_scores,
    #                     k=min(r, len(candidate_positions)),
    #                 )

    #                 selected = candidate_positions[top_idx]

    #                 remask_index[j, selected] = True

    #             # ====================================================
    #             # APPLY REMASKING
    #             # ====================================================

    #             x[remask_index] = mask_id

    #             remask_counter[remask_index] += 1

    #             if histories is not None:
    #                 histories.append(x.clone())

    #     # ============================================================
    #     # Return
    #     # ============================================================

    #     if not return_dict:
    #         return x

    #     return BaseSamplerOutput(
    #         sequences=x,
    #         histories=histories,
    #     )






























    # @torch.no_grad()
    # def sampling_revising_by_gradualy_remasking(
    #     self,
    #     inputs: list[torch.Tensor | list],
    #     config: MDLMSamplerConfig | None = None,
    #     **kwargs,
    # ) -> BaseSamplerOutput | torch.Tensor:
    #     """
    #     Generate text using masked diffusion language modeling.

    #     Iteratively unmasks tokens over multiple diffusion steps, starting from
    #     fully masked sequences appended to the input prompts.

    #     Args:
    #         inputs: List of input prompts (token tensors or lists of token IDs).
    #         config: Sampler configuration, or None to use defaults.
    #         **kwargs: Override specific config parameters.

    #     Returns:
    #         SamplerOutput with generated sequences, or raw tensor if return_dict=False.
    #     """
    #     if config is None:
    #         config = MDLMSamplerConfig()

    #     # ----- pull args from config, allow kwargs to override -----
    #     steps = kwargs.get("steps", config.steps)
    #     max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
    #     max_length = kwargs.get("max_length", config.max_length)
    #     block_size = kwargs.get("block_size", config.block_size)
    #     temperature = kwargs.get("temperature", config.temperature)
    #     cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
    #     cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
    #     remasking = kwargs.get("remasking", config.remasking)
    #     suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
    #     stochastic_transfer = kwargs.get(
    #         "stochastic_transfer", config.stochastic_transfer
    #     )
    #     return_dict = kwargs.get("return_dict", config.return_dict)
    #     right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
    #     begin_suppress_tokens = kwargs.get(
    #         "begin_suppress_tokens", config.begin_suppress_tokens
    #     )

    #     assert 1 <= block_size
    #     assert 1 <= steps
    #     mask_id = self.tokenizer.mask_token_id
    #     bos_id = self.tokenizer.bos_token_id
    #     eos_id = self.tokenizer.eos_token_id

    #     if right_shift_logits:
    #         inputs = [
    #             [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
    #         ]

    #     if isinstance(inputs[0], list):
    #         inputs = [
    #             torch.as_tensor(p, dtype=torch.long, device=self.model.device)
    #             for p in inputs
    #         ]
    #     prompt_lens = [p.shape[0] for p in inputs]

    #     if max_new_tokens:
    #         max_length = max_new_tokens + max(prompt_lens)
    #     else:
    #         max_new_tokens = max_length - max(prompt_lens)

    #     B = len(inputs)
    #     T = max_length

    #     x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
    #     for i, p in enumerate(inputs):
    #         x[i, : prompt_lens[i]] = p  # keep original prompt tokens
    #         x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = (
    #             mask_id  
    #         )
    #     attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
    #     for i, pl in enumerate(prompt_lens):
    #         valid_end = min(pl + max_new_tokens, T)
    #         attention_mask[i, :valid_end] = 1

    #     unmasked_index = (x != mask_id) & attention_mask.bool()
    #     if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
    #         keep_mask = torch.isin(
    #             x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
    #         )
    #         unmasked_index = unmasked_index & ~keep_mask

    #     num_blocks = math.ceil(max_new_tokens / block_size)
    #     steps = math.ceil(steps / num_blocks)  # per-block step budget
    #     histories = [x.clone()] if return_dict else None


    #     for b in range(num_blocks):
    #         # Build a per-sample mask *within this block* (aligned to each prompt's tail)
    #         block_mask_index = torch.zeros(
    #             (B, block_size), dtype=torch.bool, device=x.device
    #         )
    #         widths = []
    #         block_starts = []
    #         block_ends = []
    #         for j in range(B):
    #             start = prompt_lens[j] + b * block_size
    #             end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
    #             width = max(0, end - start)
    #             widths.append(width)
    #             block_starts.append(start)
    #             block_ends.append(end)
    #             if width > 0:
    #                 block_mask_index[j, :width] = x[j, start:end] == mask_id

    #         # Decide how many tokens to reveal per step in this block
    #         num_transfer_tokens = get_num_transfer_tokens(
    #             mask_index=block_mask_index,
    #             steps=steps,
    #             scheduler=self.scheduler,
    #             stochastic=stochastic_transfer,
    #         )
    #         # print(f"----------------------num_transfer_tokens (block {b})-------------------")
    #         # print(num_transfer_tokens.shape, num_transfer_tokens) # torch.Size([1, 32]) tensor([[1,  ..., 1, 1, 1]])

    #         # Some steps may be skipped if there are no transfers
    #         effective_steps = num_transfer_tokens.size(1)
    #         # ----- Iterative reveal inside the current block -----
    #         for i in range(effective_steps):
    #             mask_index = x == mask_id 
    #             logits = self.model(x, attention_mask=attention_mask).logits  
    #             logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    #             x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, T] predicted token ids
    #             x0_raw = torch.argmax(logits_with_noise, dim=-1)  # save raw predictions BEFORE the where-clamp

    #             if remasking == "low_confidence":
    #                 p = F.softmax(logits, dim=-1)
    #                 x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # [B, T] confidence of predicted token
    #             elif remasking == "random":
    #                 x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)  # random scores
    #             else:
    #                 raise NotImplementedError(remasking)
    #             for j in range(B):
    #                 x0_p[j, prompt_lens[j] + (b + 1) * block_size :] = -np.inf
    #             x0 = torch.where(mask_index, x0, x)
    #             confidence = torch.where(mask_index, x0_p, -np.inf)
    #             transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

    #             for j in range(confidence.shape[0]):
    #                 _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
    #                 transfer_index[j, select_index] = True

    #             # Commit chosen predictions into the canvas
    #             x[transfer_index] = x0[transfer_index]
    #             if histories is not None:
    #                 histories.append(x.clone())



    #     if not return_dict:
    #         return x
    #     else:
    #         return BaseSamplerOutput(sequences=x, histories=histories)


