import argparse
import json
import time
from types import SimpleNamespace
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from model import DFlashDraftModel, sample, extract_context_feature

MTP_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@torch.inference_mode()
def dflash_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list,
    temperature: float = 0.0,
) -> SimpleNamespace:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    # Prefill stage
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True if block_size > 1 else False,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    start = input_ids.shape[1]
    acceptance_lengths = []
    draft_prefill = True

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        if block_size > 1:
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(
                model(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[
                        :, past_key_values_draft.get_seq_length() : start + block_size
                    ],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                )[:, -block_size + 1 :, :]
            )
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)
            if draft_prefill:
                draft_prefill = False

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True if block_size > 1 else False,
        )

        posterior = sample(output.logits, temperature)
        acceptance_length = (
            (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        )
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
            :, : acceptance_length + 1
        ]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length + 1)
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[
                :, : acceptance_length + 1, :
            ]

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_token_ids_t = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = (
            torch.isin(output_ids[0][num_input_tokens:], stop_token_ids_t).nonzero(as_tuple=True)[0]
        )
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        acceptance_lengths=acceptance_lengths,
    )


def eval_model(
    draft_model,
    target,
    tokenizer,
    text,
    max_new_tokens=512,
    temperature=0.0,
    block_size=None,
):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(target.device)
    result = dflash_generate(
        model=draft_model,
        target=target,
        input_ids=input_ids,
        mask_token_id=tokenizer.mask_token_id,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        stop_token_ids=[tokenizer.eos_token_id],
        temperature=temperature,
    )
    accept_length_list = [int(x) for x in result.acceptance_lengths]
    generated_ids = result.output_ids[0, result.num_input_tokens:]
    output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return accept_length_list, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="gpqa_diamond")
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--model-name-or-path", type=str, default='Qwen3-8B')
    parser.add_argument("--draft-name-or-path", type=str, default='Qwen3-8B-DFlash-b16')
    parser.add_argument(
        "--bench-data-dir",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--bench-save-dir",
        type=str,
        default="results",
    )
    parser.add_argument("--block-size", type=int, default=None)
    args = parser.parse_args()

    bench_file_name = f"{args.bench_name}.jsonl"
    suffix = f"_qwen8b_dflash_{args.max_new_tokens}.jsonl"
    bench_data_path = f"{MTP_BASE}/data/{bench_file_name}"
    bench_save_path = f"{MTP_BASE}/results/{bench_file_name.replace('.jsonl', suffix)}"

    device = torch.device("cuda:0")

    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    target = AutoModelForCausalLM.from_pretrained(
        f"{MTP_BASE}/models_from_hf/{args.model_name_or_path}",
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device).eval()

    draft_model = DFlashDraftModel.from_pretrained(
        f"{MTP_BASE}/models_from_hf/{args.draft_name_or_path}",
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    tokenizer = AutoTokenizer.from_pretrained(f"{MTP_BASE}/models_from_hf/{args.model_name_or_path}", local_files_only=True,)
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})

    bench_data = []
    with open(bench_data_path, "r") as f:
        for line in f:
            bench_data.append(json.loads(line))

    for line_data in bench_data:
        prompt = line_data["prompt"]
        messages = [{"role": "user", "content": prompt}]
        think_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        no_think_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        think_accept_length_list, think_output = eval_model(
            draft_model, target, tokenizer, think_text, args.max_new_tokens, block_size=block_size, temperature=0.6
        )
        no_think_accept_length_list, no_think_output = eval_model(
            draft_model, target, tokenizer, no_think_text, args.max_new_tokens, block_size=block_size, temperature=0.6
        )

        line_data["think_accept_length_list"] = think_accept_length_list
        line_data["no_think_accept_length_list"] = no_think_accept_length_list
        line_data["think_output"] = think_output
        line_data["no_think_output"] = no_think_output

        with open(bench_save_path, "a", encoding="utf-8") as outfile:
            json.dump(line_data, outfile, ensure_ascii=False)
            outfile.write("\n")


if __name__ == "__main__":
    main()